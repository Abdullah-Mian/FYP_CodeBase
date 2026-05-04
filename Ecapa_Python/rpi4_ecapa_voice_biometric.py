#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        Raspberry Pi 4 – ECAPA-TDNN Voice Biometric System                  ║
║        Audio Source: ESP32 INMP441 I2S → UART → RPi4                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Functional parity with laptop_onnx_main.py — adapted for RPi:
  • Audio received from ESP32 over UART (no microphone, no sounddevice)
  • Feature extraction: pure NumPy/SciPy (no torch, no torchaudio, no librosa)
    → Mel filterbank matches torchaudio.transforms.MelSpectrogram defaults
       (HTK scale, norm=None, center=True, periodic Hann window, power=2.0)
  • ONNX inference: onnxruntime (ARM64 native via pip)
  • Voice Quality Check before processing (energy + zero-crossing VAD)
  • Full enrollment + verification with cosine-similarity scoring
  • Performance benchmarking identical to laptop version

Frame Protocol from ESP32 (rpi4_audio_receiver.py compatible):
    [0xAA][0x55][LEN_H][LEN_L][PCM_DATA...]
    LEN=0 → end-of-stream marker

Wiring:
    ESP32 GPIO17 (TX2) ──► RPi4 GPIO15 / Pin 10 (RXD)
    ESP32 GND          ──► RPi4 GND / Pin 6 or 14

Installation (Raspberry Pi OS 64-bit / aarch64):
    sudo apt update && sudo apt install -y python3-numpy python3-scipy libsndfile1
    pip3 install onnxruntime pyserial psutil

Installation (Raspberry Pi OS 32-bit / armhf):
    sudo apt install -y python3-numpy python3-scipy libsndfile1 python3-serial python3-psutil
    # onnxruntime on 32-bit: use pre-built wheel from community repo
    pip3 install https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux/raw/master/wheels/bullseye/onnxruntime-1.16.3-cp39-cp39-linux_armv7l.whl
    # (adjust Python version suffix cp3X to match: python3 --version)

Usage:
    python3 rpi4_ecapa_voice_biometric.py [--model ecapa_tdnn.onnx] [--debug]
    # Place ecapa_tdnn.onnx in the same directory or pass --model path
"""

import os
import sys
import glob
import time
import wave
import struct
import signal
import argparse
import threading
from datetime import datetime

import numpy as np
import serial
import psutil

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime not installed.")
    print("        Run: pip3 install onnxruntime")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SERIAL_PORT    = "/dev/serial0"   # RPi4 hardware UART
BAUD_RATE      = 460800           # Match ESP32 firmware
SAMPLE_RATE    = 16000            # Hz — must match ESP32 I2S config
SAMPLE_WIDTH   = 2                # bytes (16-bit PCM)
RECEIVE_TIMEOUT_S = 3.0           # seconds of silence → session end

VOICEPRINT_DB  = "voiceprint_database_onnx"  # same dir name as laptop
ONNX_MODEL     = "ecapa_tdnn.onnx"

# Frame sync bytes
SYNC_A = 0xAA
SYNC_B = 0x55

# ECAPA-TDNN feature parameters — MUST match how the model was trained
# These replicate: torchaudio.transforms.MelSpectrogram(
#     sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80)
N_FFT      = 400
WIN_LENGTH = 400   # == n_fft (no zero-padding needed)
HOP_LENGTH = 160
N_MELS     = 80

# Speaker verification decision threshold (same as laptop)
THRESHOLD  = 0.70

# Minimum speech duration to accept for processing
MIN_SPEECH_DURATION_S = 1.5   # seconds

# Audio normalisation target peak (avoids INMP441 low-level issue)
NORM_TARGET_PEAK = 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# PURE NUMPY MEL SPECTROGRAM  (replaces torchaudio — zero ARM build issues)
# ═══════════════════════════════════════════════════════════════════════════════

def _hz_to_mel_htk(hz: np.ndarray) -> np.ndarray:
    """HTK mel scale — matches torchaudio default mel_scale='htk'."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz_htk(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int,
                           fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """
    Triangular mel filterbank matching torchaudio.transforms.MelSpectrogram defaults:
        mel_scale='htk',  norm=None  (no area normalisation)
    Returns shape (n_mels, n_fft // 2 + 1).
    """
    if fmax is None:
        fmax = sr / 2.0

    mel_min = _hz_to_mel_htk(fmin)
    mel_max = _hz_to_mel_htk(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = _mel_to_hz_htk(mel_points)

    # Uniform FFT bin frequencies
    bin_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_low  = hz_points[m - 1]
        f_mid  = hz_points[m]
        f_high = hz_points[m + 1]

        rising  = (bin_freqs - f_low)  / (f_mid  - f_low  + 1e-30)
        falling = (f_high - bin_freqs) / (f_high - f_mid  + 1e-30)
        filterbank[m - 1] = np.maximum(0.0, np.minimum(rising, falling))

    return filterbank


# Pre-compute filterbank once at startup
_MEL_FILTERBANK: np.ndarray = None

def _get_filterbank() -> np.ndarray:
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        _MEL_FILTERBANK = _build_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
    return _MEL_FILTERBANK


def compute_log_mel_spectrogram(audio_f32: np.ndarray) -> np.ndarray:
    """
    Compute Log-Mel Spectrogram matching:
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=80)
        followed by torch.log(mel + 1e-6)
        followed by .permute(0, 2, 1)   →  shape (1, T, 80)

    Input : float32 mono audio, normalised to [-1, 1]
    Output: float32 ndarray of shape (1, time_frames, 80)
    """
    # Periodic Hann window — matches torch.hann_window(N, periodic=True)
    n = np.arange(WIN_LENGTH)
    window = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / WIN_LENGTH)).astype(np.float32)

    # Zero-pad window to n_fft (no-op when WIN_LENGTH == N_FFT)
    if WIN_LENGTH < N_FFT:
        pad_l = (N_FFT - WIN_LENGTH) // 2
        pad_r = N_FFT - WIN_LENGTH - pad_l
        window = np.pad(window, (pad_l, pad_r))

    # Reflect-pad audio for centre STFT (torchaudio default center=True)
    pad = N_FFT // 2
    audio_padded = np.pad(audio_f32.astype(np.float32), pad, mode='reflect')

    n_frames = (len(audio_padded) - N_FFT) // HOP_LENGTH + 1
    power = np.empty((N_FFT // 2 + 1, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = audio_padded[start: start + N_FFT] * window
        power[:, i] = np.abs(np.fft.rfft(frame, n=N_FFT)) ** 2

    # Mel filterbank → log
    mel = _get_filterbank() @ power                   # (80, T)
    log_mel = np.log(mel + 1e-6)                      # (80, T)

    # (1, T, 80) — same as torchaudio after .permute(0, 2, 1)
    return log_mel.T[np.newaxis, :, :].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO NORMALISATION & QUALITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def pcm_bytes_to_float32(raw_pcm: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM bytes to normalised float32 in [-1, 1]."""
    samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
    return samples / 32768.0


def normalise_audio(audio_f32: np.ndarray) -> np.ndarray:
    """
    Peak-normalise to NORM_TARGET_PEAK.
    INMP441 via ESP32 often captures at ~42 % of full scale — this compensates.
    """
    peak = np.max(np.abs(audio_f32))
    if peak > 1e-6:
        return audio_f32 * (NORM_TARGET_PEAK / peak)
    return audio_f32


def check_audio_quality(audio_f32: np.ndarray, sr: int = SAMPLE_RATE) -> tuple:
    """
    Simple energy + zero-crossing voice quality check (no external VAD library).

    Returns (is_ok: bool, reason: str)
    """
    duration = len(audio_f32) / sr

    if duration < MIN_SPEECH_DURATION_S:
        return False, f"Too short ({duration:.2f}s < {MIN_SPEECH_DURATION_S}s minimum)"

    # RMS energy check
    rms = np.sqrt(np.mean(audio_f32 ** 2))
    if rms < 0.002:   # ~ -54 dBFS after normalisation — essentially silence
        return False, f"Audio too quiet (RMS={rms:.5f}). Check INMP441 wiring."

    # Zero-crossing rate — speech is typically 2–30 kHz-equivalent ZCR
    # Below this the signal is near-DC / stuck; well above = noise / RF
    zcr = np.sum(np.abs(np.diff(np.sign(audio_f32)))) / (2.0 * duration)
    if zcr < 50:
        return False, f"Very low zero-crossing rate ({zcr:.0f}/s) — possible DC/stuck signal"
    if zcr > 10000:
        return False, f"Excessive zero-crossing rate ({zcr:.0f}/s) — possible noise or RF interference"

    return True, "OK"


# ═══════════════════════════════════════════════════════════════════════════════
# UART AUDIO RECEIVER  (from rpi4_audio_receiver.py — inline, no import needed)
# ═══════════════════════════════════════════════════════════════════════════════

class UARTAudioReceiver:
    """
    Receives framed 16-bit PCM audio from ESP32 over UART.
    Frame format: [0xAA][0x55][LEN_H][LEN_L][PCM_DATA...]  LEN=0 → EOS
    """

    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE,
                 timeout: float = RECEIVE_TIMEOUT_S):
        self._port    = port
        self._baud    = baud
        self._timeout = timeout
        self._ser: serial.Serial = None

    def open(self):
        self._ser = serial.Serial(
            port      = self._port,
            baudrate  = self._baud,
            bytesize  = serial.EIGHTBITS,
            parity    = serial.PARITY_NONE,
            stopbits  = serial.STOPBITS_ONE,
            timeout   = self._timeout,
        )
        self._ser.reset_input_buffer()
        print(f"[UART] Opened {self._port} @ {self._baud} baud  timeout={self._timeout}s")

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    # ── internal frame helpers ──────────────────────────────────────────────

    def _find_sync(self) -> bool:
        """Scan stream for 0xAA 0x55 sync header. Returns True when found."""
        state = 0
        while True:
            b = self._ser.read(1)
            if not b:
                return False   # timeout
            byte = b[0]
            if state == 0:
                if byte == SYNC_A:
                    state = 1
            else:
                if byte == SYNC_B:
                    return True
                state = 1 if byte == SYNC_A else 0

    def _read_frame(self) -> bytes | None:
        """
        Read one frame.
        Returns: PCM bytes  (non-empty)
                 b''        (end-of-stream)
                 None       (timeout / error)
        """
        if not self._find_sync():
            return None

        hdr = self._ser.read(2)
        if len(hdr) < 2:
            return None

        frame_len = (hdr[0] << 8) | hdr[1]
        if frame_len == 0:
            return b''   # EOS marker

        buf = bytearray()
        remaining = frame_len
        while remaining > 0:
            chunk = self._ser.read(remaining)
            if not chunk:
                return None   # timeout mid-frame
            buf.extend(chunk)
            remaining -= len(chunk)
        return bytes(buf)

    # ── public: blocking receive of one complete audio session ──────────────

    def receive_session(self, prompt: str = "Speak now (waiting for ESP32)...") -> bytes | None:
        """
        Block until a complete audio session arrives.
        Session ends on EOS marker (ESP32-side VAD) or RECEIVE_TIMEOUT_S silence.

        Returns raw PCM bytes, or None if interrupted / empty.
        """
        print(f"\n[UART] {prompt}")
        accum = bytearray()
        session_started = False
        session_start_time = None

        while True:
            frame = self._read_frame()

            if frame is None:
                # Timeout — end session if we already have data
                if session_started and accum:
                    break
                # else: keep waiting for first frame
                continue

            if frame == b'':
                # EOS marker from ESP32 — clean end
                if accum:
                    break
                # EOS before any data → ESP32 woke up and went back to sleep
                continue

            if not session_started:
                session_started = True
                session_start_time = time.time()
                print(f"[UART] Session started at {datetime.now().strftime('%H:%M:%S')}")

            accum.extend(frame)
            duration = len(accum) / (SAMPLE_RATE * SAMPLE_WIDTH)
            sys.stdout.write(f"\r[UART] Receiving... {duration:.1f}s  ({len(accum):,} bytes)")
            sys.stdout.flush()

        if accum:
            duration = len(accum) / (SAMPLE_RATE * SAMPLE_WIDTH)
            print(f"\n[UART] Session complete: {duration:.2f}s  ({len(accum):,} bytes)")
            return bytes(accum)

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ONNX AUTHENTICATOR  (identical to laptop_onnx_main.py)
# ═══════════════════════════════════════════════════════════════════════════════

class ONNXAuthenticator:
    """Handles ECAPA-TDNN ONNX model inference."""

    def __init__(self, model_path: str):
        print(f"[ONNX] Loading model: {model_path}")
        # Disable parallelism to avoid overloading RPi cores during I/O
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=so)
        inp = self.session.get_inputs()[0]
        print(f"[ONNX] Input name : {inp.name}   shape: {inp.shape}")
        print(f"[ONNX] Model ready.")
        self._input_name = inp.name

    def extract_embedding(self, features: np.ndarray) -> np.ndarray:
        """Run inference. features: (1, T, 80) float32."""
        return self.session.run(None, {self._input_name: features})[0]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.flatten(), b.flatten()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / (denom + 1e-12))


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR  (wraps pure-numpy pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Drop-in replacement for the torchaudio-based FeatureExtractor on laptop.
    Works entirely with NumPy — no PyTorch, no librosa, no compilation required.
    """

    def __init__(self):
        # Pre-compute filterbank so first call isn't slow
        _ = _get_filterbank()
        print(f"[FEAT] NumPy mel filterbank ready  "
              f"(n_fft={N_FFT}, hop={HOP_LENGTH}, n_mels={N_MELS})")

    def extract_from_pcm_bytes(self, raw_pcm: bytes) -> tuple:
        """
        Convert raw PCM bytes → float32 → normalised → log-mel features.
        Returns (features, audio_f32)  or raises ValueError on bad audio.
        """
        audio = pcm_bytes_to_float32(raw_pcm)
        audio = normalise_audio(audio)

        ok, reason = check_audio_quality(audio)
        if not ok:
            raise ValueError(f"Audio quality check failed: {reason}")

        features = compute_log_mel_spectrogram(audio)
        return features, audio


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save_debug_wav(audio_f32: np.ndarray, tag: str = "debug") -> str:
    """Optionally save received audio for offline inspection."""
    os.makedirs("debug_audio", exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"debug_audio/{tag}_{ts}.wav"
    pcm16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    return path


def enroll_speaker(uart: UARTAudioReceiver,
                   extractor: FeatureExtractor,
                   auth: ONNXAuthenticator,
                   debug: bool = False):
    """Enrol a new speaker from ESP32 audio."""

    username = input("\nEnter username for enrollment: ").lower().strip()
    if not username:
        print("[!] Username cannot be empty.")
        return

    emb_path = os.path.join(VOICEPRINT_DB, f"{username}.npy")
    if os.path.exists(emb_path):
        print(f"[!] User '{username}' already enrolled. Delete {emb_path} to re-enrol.")
        return

    print(f"\n[ENROL] Enrolling '{username}'")
    print("       Trigger the ESP32 to send audio (speak near the mic).")

    raw = uart.receive_session(prompt="Waiting for enrollment audio from ESP32...")
    if not raw:
        print("[!] No audio received. Enrollment aborted.")
        return

    try:
        features, audio = extractor.extract_from_pcm_bytes(raw)
    except ValueError as e:
        print(f"[!] {e}")
        return

    if debug:
        path = save_debug_wav(audio, tag=f"enroll_{username}")
        print(f"[DBG] Saved enrollment audio → {path}")

    print(f"[ENROL] Extracting voiceprint for '{username}'...")
    t0 = time.perf_counter()
    embedding = auth.extract_embedding(features)
    ms = (time.perf_counter() - t0) * 1000

    np.save(emb_path, embedding)
    dur = len(raw) / (SAMPLE_RATE * SAMPLE_WIDTH)
    print(f"[ENROL] ✓ '{username}' enrolled  ({dur:.2f}s audio, inference {ms:.1f}ms)")


def verify_speaker(uart: UARTAudioReceiver,
                   extractor: FeatureExtractor,
                   auth: ONNXAuthenticator,
                   debug: bool = False):
    """Verify/identify a speaker from incoming ESP32 audio."""

    enrolled = glob.glob(os.path.join(VOICEPRINT_DB, "*.npy"))
    if not enrolled:
        print("[!] No speakers enrolled. Please enroll first.")
        return

    print(f"\n[VERIFY] Enrolled speakers: {[os.path.basename(p)[:-4] for p in enrolled]}")
    print("        Trigger the ESP32 to send audio (speak near the mic).")

    proc = psutil.Process(os.getpid())

    raw = uart.receive_session(prompt="Waiting for verification audio from ESP32...")
    if not raw:
        print("[!] No audio received. Verification aborted.")
        return

    try:
        proc.cpu_percent(interval=None)  # reset cpu counter

        t_feat_start = time.perf_counter()
        features, audio = extractor.extract_from_pcm_bytes(raw)
        feat_ms = (time.perf_counter() - t_feat_start) * 1000

        t_infer_start = time.perf_counter()
        live_emb = auth.extract_embedding(features)
        infer_ms = (time.perf_counter() - t_infer_start) * 1000

        cpu_pct  = proc.cpu_percent(interval=None)
        ram_mb   = proc.memory_info().rss / (1024 * 1024)

    except ValueError as e:
        print(f"[!] {e}")
        return

    if debug:
        path = save_debug_wav(audio, tag="verify")
        print(f"[DBG] Saved verification audio → {path}")

    # ── Compare against all enrolled voiceprints ─────────────────────────
    best_score  = -1.0
    best_match  = None

    print()
    for emb_path in enrolled:
        enrolled_emb = np.load(emb_path)
        score = auth.cosine_similarity(enrolled_emb, live_emb)
        name  = os.path.basename(emb_path)[:-4]
        print(f"  Compared with {name:20s}: score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_match = name

    print(f"\n  Best match: {best_match}  (score {best_score:.4f}  threshold {THRESHOLD})")

    if best_score >= THRESHOLD:
        print(f"\n  >>> RESULT: ✓ MATCH — Welcome, {best_match}!")
    else:
        print(f"\n  >>> RESULT: ✗ NO MATCH — Unknown speaker")

    # ── Performance metrics (same as laptop) ─────────────────────────────
    print()
    print("  ─── Performance Metrics ────────────────────────────────────")
    print(f"  Feature Extraction Latency : {feat_ms:.2f} ms")
    print(f"  ONNX Inference Latency     : {infer_ms:.2f} ms")
    print(f"  End-to-End Latency         : {feat_ms + infer_ms:.2f} ms")
    print(f"  CPU Usage during processing: {cpu_pct:.1f} %")
    print(f"  Current RAM Footprint      : {ram_mb:.1f} MB")
    print(f"  Audio received             : {len(raw)//(SAMPLE_RATE*SAMPLE_WIDTH):.1f}s  ({len(raw):,} bytes)")
    print("  ─────────────────────────────────────────────────────────────")


def list_enrolled():
    enrolled = glob.glob(os.path.join(VOICEPRINT_DB, "*.npy"))
    if not enrolled:
        print("  (no speakers enrolled)")
    else:
        for p in enrolled:
            emb = np.load(p)
            print(f"  {os.path.basename(p)[:-4]:20s}  embedding shape: {emb.shape}")


def delete_speaker():
    enrolled = glob.glob(os.path.join(VOICEPRINT_DB, "*.npy"))
    if not enrolled:
        print("  (no speakers enrolled)")
        return
    list_enrolled()
    name = input("Enter username to delete: ").lower().strip()
    path = os.path.join(VOICEPRINT_DB, f"{name}.npy")
    if os.path.exists(path):
        os.remove(path)
        print(f"  ✓ Deleted '{name}'")
    else:
        print(f"  [!] User '{name}' not found")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="RPi4 ECAPA-TDNN Voice Biometric — ESP32 UART audio source")
    p.add_argument("--model",  default=ONNX_MODEL, help="Path to ecapa_tdnn.onnx")
    p.add_argument("--port",   default=SERIAL_PORT, help="UART port (default /dev/serial0)")
    p.add_argument("--baud",   type=int, default=BAUD_RATE, help="Baud rate")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="Cosine similarity decision threshold (default 0.70)")
    p.add_argument("--debug",  action="store_true",
                   help="Save received audio WAVs to debug_audio/")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Pre-flight checks ─────────────────────────────────────────────────

    if not os.path.exists(args.model):
        print(f"\n[ERROR] ONNX model not found: {args.model}")
        print("        Copy ecapa_tdnn.onnx to this directory or use --model path/to/model.onnx")
        sys.exit(1)

    os.makedirs(VOICEPRINT_DB, exist_ok=True)

    model_mb = os.path.getsize(args.model) / (1024 * 1024)

    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║    RPi4 ECAPA-TDNN Voice Biometric  (ESP32 UART source)  ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"  ONNX model  : {args.model}  ({model_mb:.1f} MB)")
    print(f"  UART port   : {args.port}  @  {args.baud} baud")
    print(f"  Sample rate : {SAMPLE_RATE} Hz  |  16-bit mono")
    print(f"  Threshold   : {args.threshold}")
    print(f"  Debug mode  : {'ON (saving WAVs to debug_audio/)' if args.debug else 'OFF'}")
    print()

    # ── Initialise components ─────────────────────────────────────────────
    extractor = FeatureExtractor()
    auth      = ONNXAuthenticator(args.model)
    uart      = UARTAudioReceiver(port=args.port, baud=args.baud)

    # Override threshold if user passed --threshold
    global THRESHOLD
    THRESHOLD = args.threshold

    try:
        uart.open()
    except serial.SerialException as e:
        print(f"\n[ERROR] Cannot open UART: {e}")
        print("  Check: sudo raspi-config → Interface → Serial Port")
        print("         Login shell: NO   Hardware enabled: YES")
        print("         Then reboot.")
        sys.exit(1)

    # ── Graceful shutdown ─────────────────────────────────────────────────
    running = True
    def _sighandler(sig, frame):
        nonlocal running
        running = False
        print("\n[SYS] Shutting down...")
    signal.signal(signal.SIGINT,  _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    # ── Main menu loop ────────────────────────────────────────────────────
    print("\nSystem ready. Use the menu below.")
    print("─" * 50)

    while running:
        print()
        print("─── ESP32 Voice Biometric System ───────────────────")
        print("  1.  Enroll New Speaker")
        print("  2.  Verify / Identify Speaker")
        print("  3.  List Enrolled Speakers")
        print("  4.  Delete Speaker")
        print("  5.  Exit")
        print("─────────────────────────────────────────────────────")

        try:
            choice = input("  Choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if   choice == '1': enroll_speaker(uart, extractor, auth, debug=args.debug)
        elif choice == '2': verify_speaker(uart, extractor, auth, debug=args.debug)
        elif choice == '3': list_enrolled()
        elif choice == '4': delete_speaker()
        elif choice == '5': break
        else:               print("  [!] Invalid choice")

    uart.close()
    print("[SYS] Done.")


if __name__ == "__main__":
    main()
