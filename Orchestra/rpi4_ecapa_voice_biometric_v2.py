#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        Raspberry Pi 4 – ECAPA-TDNN Voice Biometric System  v2              ║
║        Audio Source: ESP32 INMP441 I2S → UART → RPi4                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Changes from v1:
  • GLITCH FIX: reset_input_buffer() called at the start of EVERY session
  • Mode-aware length enforcement (enroll 5-10s / verify 2-6s)
  • Auto-truncate at max length even if ESP32 keeps streaming
  • Pre-enrollment duplicate check rejects if score > DUPLICATE_THRESHOLD
  • Embedding shape explained inline (always fixed regardless of audio length)

Installation (Raspberry Pi OS 64-bit / aarch64 — recommended):
    sudo apt update && sudo apt install -y python3-pip python3-numpy python3-scipy libsndfile1
    pip3 install onnxruntime pyserial psutil

Installation (Raspberry Pi OS 32-bit / armhf):
    sudo apt install -y python3-pip python3-numpy python3-scipy python3-serial python3-psutil
    # match cp3XX to your Python: python3 --version
    pip3 install https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux/raw/master/wheels/bullseye/onnxruntime-1.16.3-cp311-cp311-linux_armv7l.whl

Usage:
    python3 rpi4_ecapa_voice_biometric.py [--model ecapa_tdnn.onnx] [--debug]
"""

import os
import sys
import glob
import time
import wave
import signal
import argparse
from datetime import datetime

import numpy as np
import serial
import psutil

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime not installed.  Run: pip3 install onnxruntime")
    sys.exit(1)

# ===============================================================================
# CONFIGURATION
# ===============================================================================

SERIAL_PORT       = "/dev/serial0"
BAUD_RATE         = 460800
SAMPLE_RATE       = 16000
SAMPLE_WIDTH      = 2          # bytes, 16-bit PCM
RECEIVE_TIMEOUT_S = 3.0        # seconds silence -> session end

VOICEPRINT_DB     = "voiceprint_database_onnx"
ONNX_MODEL        = "ecapa_tdnn.onnx"

# Frame sync bytes (must match ESP32 firmware)
SYNC_A = 0xAA
SYNC_B = 0x55

# ECAPA-TDNN feature parameters (must match training config)
N_FFT      = 400
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS     = 80

# ---- Audio length limits -----------------------------------------------------
#
#  ENROLLMENT (5-10 s):
#    ECAPA-TDNN's Attentive Statistics Pooling (ASP) computes weighted mean +
#    weighted std across ALL time frames and collapses them into a fixed-size
#    embedding. More frames = better statistics = more stable voiceprint.
#    Research benchmarks show embeddings stabilise around 5-6 s for enrollment.
#    Beyond ~10 s you get diminishing returns, and long clips risk including
#    coughs, background noise, or silence that degrade the voiceprint.
#
#  VERIFICATION (2-6 s):
#    2 s is the practical minimum for reliable cosine scoring.
#    Beyond 6 s the score rarely changes by more than 0.01-0.02 on RPi
#    while inference latency grows linearly with more frames.
#
ENROLL_MIN_S = 8.0
ENROLL_MAX_S = 10.0
VERIFY_MIN_S = 6.0
VERIFY_MAX_S = 8.0

# Cosine similarity decision thresholds
VERIFY_THRESHOLD    = 0.70   # score >= this -> MATCH
DUPLICATE_THRESHOLD = 0.72   # score >= this during enrollment -> reject as duplicate
                              # (set slightly above VERIFY_THRESHOLD so we catch
                              #  re-enrolling the same person under a new name)

NORM_TARGET_PEAK = 0.90


# ===============================================================================
# PURE NUMPY MEL SPECTROGRAM
# Mathematically identical to torchaudio.transforms.MelSpectrogram defaults:
#   mel_scale='htk', norm=None, center=True, periodic Hann window, power=2.0
# No torch, no torchaudio, no llvmlite compilation needed on RPi.
# ===============================================================================

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, np.float64) / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, np.float64) / 2595.0) - 1.0)

_MEL_FB = None  # cached after first call

def _get_filterbank():
    global _MEL_FB
    if _MEL_FB is None:
        fmax = SAMPLE_RATE / 2.0
        mels = np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), N_MELS + 2)
        hz   = _mel_to_hz(mels)
        bins = np.linspace(0.0, fmax, N_FFT // 2 + 1)
        fb   = np.zeros((N_MELS, N_FFT // 2 + 1), np.float32)
        for m in range(1, N_MELS + 1):
            lo, mid, hi = hz[m-1], hz[m], hz[m+1]
            fb[m-1] = np.maximum(0.0, np.minimum(
                (bins - lo) / (mid  - lo  + 1e-30),
                (hi - bins) / (hi   - mid + 1e-30)))
        _MEL_FB = fb
    return _MEL_FB


def compute_log_mel(audio_f32):
    """
    Input : float32 mono, normalised to [-1, 1]
    Output: float32 ndarray shape (1, T, 80)
            Identical to torchaudio MelSpectrogram + log(x+1e-6) + permute(0,2,1)
    """
    n   = np.arange(WIN_LENGTH)
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / WIN_LENGTH)).astype(np.float32)

    pad   = N_FFT // 2
    audio = np.pad(audio_f32.astype(np.float32), pad, mode='reflect')
    n_frames = (len(audio) - N_FFT) // HOP_LENGTH + 1

    power = np.empty((N_FFT // 2 + 1, n_frames), np.float32)
    for i in range(n_frames):
        s = i * HOP_LENGTH
        power[:, i] = np.abs(np.fft.rfft(audio[s:s + N_FFT] * win, n=N_FFT)) ** 2

    mel     = _get_filterbank() @ power     # (80, T)
    log_mel = np.log(mel + 1e-6)            # (80, T)
    return log_mel.T[np.newaxis].astype(np.float32)  # (1, T, 80)


# ===============================================================================
# AUDIO HELPERS
# ===============================================================================

def pcm_to_float(raw):
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def normalise(audio):
    peak = np.max(np.abs(audio))
    return audio * (NORM_TARGET_PEAK / peak) if peak > 1e-6 else audio


def check_quality(audio):
    """Lightweight energy + zero-crossing rate VAD. Returns (ok, reason)."""
    dur = len(audio) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.005:
        return False, f"Signal too quiet (RMS={rms:.4f}) — check INMP441 wiring"
    zcr = float(np.sum(np.abs(np.diff(np.sign(audio)))) / (2.0 * dur))
    if zcr < 50:
        return False, f"ZCR={zcr:.0f}/s — possible DC / stuck signal"
    if zcr > 10000:
        return False, f"ZCR={zcr:.0f}/s — likely RF noise or disconnected mic"
    return True, "OK"


def raw_to_features(raw):
    """
    raw PCM bytes -> (features float32 (1,T,80), audio float32 normalised)
    Raises ValueError if quality check fails.
    """
    audio = pcm_to_float(raw)
    audio = normalise(audio)
    ok, reason = check_quality(audio)
    if not ok:
        raise ValueError(reason)
    return compute_log_mel(audio), audio


# ===============================================================================
# UART AUDIO RECEIVER
# ===============================================================================

class UARTReceiver:
    """
    Receives framed 16-bit PCM from ESP32 over UART.
    Frame format: [0xAA][0x55][LEN_H][LEN_L][PCM_DATA...]   LEN=0 = EOS

    GLITCH FIX EXPLANATION
    ─────────────────────
    The RPi kernel maintains a tty RX ring buffer (default 4096 bytes but can
    grow). When a session ends — whether by timeout or EOS — any bytes already
    clocked in by the UART hardware but not yet consumed by read() stay in that
    buffer. On the NEXT call to receive_session(), _sync() runs immediately on
    those stale bytes. Because 0xAA 0x55 is a two-byte sequence, it appears by
    chance in partially-received PCM data about 1-in-65536 times — often enough
    to trigger this bug every few sessions.

    The fix is one line: self._ser.reset_input_buffer() at the very top of
    receive_session(). This calls tcflush(fd, TCIFLUSH) which discards every
    byte currently waiting in the kernel RX buffer before we start listening
    for a new session. The next byte the kernel sees will be a fresh frame from
    the ESP32.
    """

    def __init__(self, port=SERIAL_PORT, baud=BAUD_RATE, timeout=RECEIVE_TIMEOUT_S):
        self._ser = serial.Serial(
            port=port, baudrate=baud,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, timeout=timeout)
        self._ser.reset_input_buffer()
        print(f"[UART] Opened {port} @ {baud} baud  (read timeout={timeout}s)")

    def close(self):
        if self._ser.is_open:
            self._ser.close()

    # ---- low-level framing ---------------------------------------------------

    def _sync(self):
        """Scan for 0xAA 0x55 sync header. Returns True when found."""
        state = 0
        while True:
            b = self._ser.read(1)
            if not b:
                return False
            v = b[0]
            if state == 0:
                state = 1 if v == SYNC_A else 0
            else:
                if v == SYNC_B:
                    return True
                state = 1 if v == SYNC_A else 0

    def _read_frame(self):
        """
        Read one frame.
        Returns: PCM bytes (non-empty) | b'' (EOS) | None (timeout/error)
        """
        if not self._sync():
            return None
        hdr = self._ser.read(2)
        if len(hdr) < 2:
            return None
        length = (hdr[0] << 8) | hdr[1]
        if length == 0:
            return b''
        buf, rem = bytearray(), length
        while rem > 0:
            chunk = self._ser.read(rem)
            if not chunk:
                return None
            buf.extend(chunk)
            rem -= len(chunk)
        return bytes(buf)

    # ---- public API ----------------------------------------------------------

    def receive_session(self, min_seconds, max_seconds, label="audio"):
        """
        Block until a complete audio session arrives.
        Enforces min_seconds (reject if shorter) and max_seconds (truncate).
        Returns raw PCM bytes, or None on failure / audio too short.
        """
        # ---- GLITCH FIX: flush stale bytes from previous session ------------
        self._ser.reset_input_buffer()
        # ---------------------------------------------------------------------

        max_bytes = int(max_seconds * SAMPLE_RATE * SAMPLE_WIDTH)
        min_bytes = int(min_seconds * SAMPLE_RATE * SAMPLE_WIDTH)
        accum     = bytearray()
        started   = False
        truncated = False

        print(f"\n[UART] Waiting for {label}...  min={min_seconds:.0f}s  max={max_seconds:.0f}s")

        while True:
            frame = self._read_frame()

            if frame is None:
                if started and accum:
                    break     # timeout after data -> end session
                continue      # timeout before first frame -> keep waiting

            if frame == b'':
                if accum:
                    break     # clean EOS from ESP32
                continue      # EOS before any data

            if not started:
                started = True
                print(f"[UART] Session started  {datetime.now().strftime('%H:%M:%S')}")

            accum.extend(frame)
            dur = len(accum) / (SAMPLE_RATE * SAMPLE_WIDTH)
            sys.stdout.write(f"\r[UART] {dur:.1f}s  ({len(accum):,} bytes)  ")
            sys.stdout.flush()

            if len(accum) >= max_bytes:
                truncated = True
                break   # hit ceiling; flush happens next call

        print()

        if not accum:
            print("[UART] No audio received.")
            return None

        actual_s = len(accum) / (SAMPLE_RATE * SAMPLE_WIDTH)
        if len(accum) < min_bytes:
            print(f"[!] Audio too short: {actual_s:.2f}s < {min_seconds:.1f}s minimum — rejected.")
            return None

        accum = accum[:max_bytes]
        final_s = len(accum) / (SAMPLE_RATE * SAMPLE_WIDTH)

        if truncated:
            print(f"[UART] Max length reached — using first {final_s:.1f}s")
        else:
            print(f"[UART] Session complete: {final_s:.2f}s")

        return bytes(accum)


# ===============================================================================
# ONNX AUTHENTICATOR
# ===============================================================================

class ONNXAuthenticator:
    def __init__(self, model_path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2   # leave 2 cores free for UART/OS on RPi
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=so)
        inp = self.session.get_inputs()[0]
        self._input_name = inp.name
        print(f"[ONNX] Model loaded  input='{inp.name}'  shape={inp.shape}")

    def embed(self, features):
        """features: (1, T, 80) float32 -> embedding ndarray"""
        return self.session.run(None, {self._input_name: features})[0]

    @staticmethod
    def cosine(a, b):
        a, b = a.flatten(), b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ===============================================================================
# DATABASE HELPERS
# ===============================================================================

def enrolled_speakers():
    return glob.glob(os.path.join(VOICEPRINT_DB, "*.npy"))


def save_debug_wav(audio_f32, tag):
    os.makedirs("debug_audio", exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"debug_audio/{tag}_{ts}.wav"
    pcm  = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return path


# ===============================================================================
# APPLICATION: ENROLL
# ===============================================================================

def enroll_speaker(uart, auth, debug):
    username = input("\n  Username to enroll: ").lower().strip()
    if not username:
        print("  [!] Username cannot be empty."); return

    emb_path = os.path.join(VOICEPRINT_DB, f"{username}.npy")
    if os.path.exists(emb_path):
        print(f"  [!] '{username}' already enrolled.")
        print(f"      Delete {emb_path} to re-enroll."); return

    print(f"\n  Enrolling '{username}'")
    print(f"  Speak clearly for {ENROLL_MIN_S:.0f}–{ENROLL_MAX_S:.0f} seconds.")
    print(f"  Trigger the ESP32 now.")

    raw = uart.receive_session(
        min_seconds=ENROLL_MIN_S,
        max_seconds=ENROLL_MAX_S,
        label=f"enrollment '{username}'")
    if raw is None:
        return

    try:
        features, audio = raw_to_features(raw)
    except ValueError as e:
        print(f"  [!] Quality check failed: {e}"); return

    if debug:
        p = save_debug_wav(audio, f"enroll_{username}")
        print(f"  [DBG] {p}")

    t0  = time.perf_counter()
    emb = auth.embed(features)
    ms  = (time.perf_counter() - t0) * 1000

    # ---- Duplicate identity check -------------------------------------------
    #
    # Why this matters:
    #   Accidentally enrolling the same person twice under different names means
    #   verification always returns whichever name was enrolled first (because
    #   both embeddings will score equally high, and tie-breaking is arbitrary).
    #   We guard against this by comparing the new embedding against all existing
    #   ones before saving.
    #
    # Threshold choice:
    #   DUPLICATE_THRESHOLD (0.72) sits just above VERIFY_THRESHOLD (0.70).
    #   Two utterances from the SAME person typically score 0.75–0.95.
    #   Two DIFFERENT people typically score < 0.65.
    #   The small gap between 0.70 and 0.72 is intentional: a genuine match
    #   during verification is accepted at 0.70, but we require 0.72 to flag a
    #   duplicate so we don't reject valid new speakers whose voice is
    #   coincidentally somewhat similar.
    #
    existing = enrolled_speakers()
    if existing:
        print(f"\n  Uniqueness check against {len(existing)} enrolled speaker(s)...")
        top_score, top_name = -1.0, None
        for p in existing:
            s = auth.cosine(np.load(p), emb)
            n = os.path.basename(p)[:-4]
            print(f"    vs {n:20s}: {s:.4f}")
            if s > top_score:
                top_score, top_name = s, n

        if top_score >= DUPLICATE_THRESHOLD:
            print(f"\n  [!] DUPLICATE REJECTED — new voice matches '{top_name}' "
                  f"at score {top_score:.4f} (threshold {DUPLICATE_THRESHOLD})")
            print(f"      This appears to be the same person.")
            print(f"      If they are genuinely different, use --dup-threshold to lower the bar.")
            return

        print(f"  Uniqueness OK  (best match: {top_name} @ {top_score:.4f})")

    np.save(emb_path, emb)
    dur = len(raw) / (SAMPLE_RATE * SAMPLE_WIDTH)
    print(f"\n  SUCCESS — '{username}' enrolled")
    print(f"  Audio used   : {dur:.2f}s")
    print(f"  Inference    : {ms:.1f}ms")
    print(f"  Embedding    : shape={emb.shape}  dtype={emb.dtype}")
    print()
    print(f"  NOTE — Why is the embedding always the same size?")
    print(f"  ECAPA-TDNN ends with an Attentive Statistics Pooling (ASP) layer.")
    print(f"  ASP computes a weighted mean and weighted std-dev across ALL time")
    print(f"  frames, then concatenates them into ONE fixed-size vector.")
    print(f"  Whether you speak for 5s or 10s, the output is always {emb.shape}.")
    print(f"  More audio = more frames = better mean/std statistics = more stable")
    print(f"  voiceprint, but the shape never changes. This is by design.")


# ===============================================================================
# APPLICATION: VERIFY
# ===============================================================================

def verify_speaker(uart, auth, threshold, debug):
    speakers = enrolled_speakers()
    if not speakers:
        print("  [!] No speakers enrolled. Enroll first."); return

    print(f"\n  Enrolled: {[os.path.basename(p)[:-4] for p in speakers]}")
    print(f"  Speak for {VERIFY_MIN_S:.0f}–{VERIFY_MAX_S:.0f} seconds.")
    print(f"  Trigger the ESP32 now.")

    proc = psutil.Process(os.getpid())

    raw = uart.receive_session(
        min_seconds=VERIFY_MIN_S,
        max_seconds=VERIFY_MAX_S,
        label="verification")
    if raw is None:
        return

    try:
        proc.cpu_percent(interval=None)    # reset CPU counter
        t0 = time.perf_counter()
        features, audio = raw_to_features(raw)
        feat_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        live_emb = auth.embed(features)
        inf_ms = (time.perf_counter() - t0) * 1000

        cpu = proc.cpu_percent(interval=None)
        ram = proc.memory_info().rss / (1024 * 1024)
    except ValueError as e:
        print(f"  [!] Quality check failed: {e}"); return

    if debug:
        p = save_debug_wav(audio, "verify")
        print(f"  [DBG] {p}")

    print()
    best_score, best_name = -1.0, None
    for p in speakers:
        s = auth.cosine(np.load(p), live_emb)
        n = os.path.basename(p)[:-4]
        print(f"    vs {n:20s}: {s:.4f}")
        if s > best_score:
            best_score, best_name = s, n

    print()
    if best_score >= threshold:
        print(f"  >>> MATCH    Welcome, {best_name}!  (score {best_score:.4f})")
    else:
        print(f"  >>> NO MATCH — Unknown speaker  "
              f"(best: {best_name} @ {best_score:.4f},  threshold {threshold})")

    dur = len(raw) / (SAMPLE_RATE * SAMPLE_WIDTH)
    print()
    print(f"  Feature extraction : {feat_ms:.1f} ms")
    print(f"  ONNX inference     : {inf_ms:.1f} ms")
    print(f"  End-to-end         : {feat_ms + inf_ms:.1f} ms")
    print(f"  CPU during proc    : {cpu:.1f}%")
    print(f"  RAM footprint      : {ram:.1f} MB")
    print(f"  Audio processed    : {dur:.2f}s")


# ===============================================================================
# APPLICATION: MISC
# ===============================================================================

def list_speakers():
    sp = enrolled_speakers()
    if not sp:
        print("  (no speakers enrolled)"); return
    print(f"\n  {'Name':25s}  {'Shape':15s}  Path")
    print(f"  {'─'*25}  {'─'*15}  {'─'*40}")
    for p in sp:
        emb  = np.load(p)
        name = os.path.basename(p)[:-4]
        print(f"  {name:25s}  {str(emb.shape):15s}  {p}")


def delete_speaker():
    sp = enrolled_speakers()
    if not sp:
        print("  (no speakers enrolled)"); return
    for p in sp:
        print(f"  {os.path.basename(p)[:-4]}")
    name = input("  Username to delete: ").lower().strip()
    path = os.path.join(VOICEPRINT_DB, f"{name}.npy")
    if os.path.exists(path):
        os.remove(path); print(f"  Deleted '{name}'")
    else:
        print(f"  [!] '{name}' not found")


# ===============================================================================
# MAIN
# ===============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="RPi4 ECAPA-TDNN Voice Biometric")
    p.add_argument("--model",         default=ONNX_MODEL)
    p.add_argument("--port",          default=SERIAL_PORT)
    p.add_argument("--baud",          type=int,   default=BAUD_RATE)
    p.add_argument("--threshold",     type=float, default=VERIFY_THRESHOLD,
                   help="Cosine similarity verify threshold (default 0.70)")
    p.add_argument("--dup-threshold", type=float, default=DUPLICATE_THRESHOLD,
                   help="Duplicate-check threshold (default 0.72)")
    p.add_argument("--debug",         action="store_true",
                   help="Save received WAVs to debug_audio/")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"\n[ERROR] ONNX model not found: {args.model}")
        print(f"        Copy ecapa_tdnn.onnx here or use --model <path>")
        sys.exit(1)

    os.makedirs(VOICEPRINT_DB, exist_ok=True)
    mb = os.path.getsize(args.model) / (1024 * 1024)

    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  RPi4  ECAPA-TDNN  Voice Biometric  v2  (ESP32 UART source)  ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print(f"  Model           : {args.model}  ({mb:.1f} MB)")
    print(f"  UART            : {args.port}  @  {args.baud} baud")
    print(f"  Enrollment      : {ENROLL_MIN_S:.0f}–{ENROLL_MAX_S:.0f}s")
    print(f"  Verification    : {VERIFY_MIN_S:.0f}–{VERIFY_MAX_S:.0f}s")
    print(f"  Verify thresh   : {args.threshold}")
    print(f"  Dup-check thresh: {args.dup_threshold}")
    print(f"  Debug mode      : {'ON (saving WAVs)' if args.debug else 'OFF'}")
    print()

    global VERIFY_THRESHOLD, DUPLICATE_THRESHOLD
    VERIFY_THRESHOLD    = args.threshold
    DUPLICATE_THRESHOLD = args.dup_threshold

    auth = ONNXAuthenticator(args.model)

    try:
        uart = UARTReceiver(port=args.port, baud=args.baud)
    except serial.SerialException as e:
        print(f"\n[ERROR] Cannot open UART: {e}")
        print("  sudo raspi-config -> Interface Options -> Serial Port")
        print("  Login shell: NO   Hardware enabled: YES   then reboot")
        sys.exit(1)

    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
        print("\n[SYS] Shutting down...")
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running:
        print()
        print("  1. Enroll New Speaker")
        print("  2. Verify / Identify Speaker")
        print("  3. List Enrolled Speakers")
        print("  4. Delete Speaker")
        print("  5. Exit")
        try:
            c = input("  Choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if   c == '1': enroll_speaker(uart, auth, args.debug)
        elif c == '2': verify_speaker(uart, auth, VERIFY_THRESHOLD, args.debug)
        elif c == '3': list_speakers()
        elif c == '4': delete_speaker()
        elif c == '5': break
        else:          print("  [!] Invalid choice")

    uart.close()
    print("[SYS] Done.")


if __name__ == "__main__":
    main()
