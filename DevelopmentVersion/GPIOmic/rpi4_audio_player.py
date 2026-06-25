#!/usr/bin/env python3
"""
Raspberry Pi 4 — Real-Time UART Audio Player
=============================================
Receives framed 16-bit PCM from ESP32 over UART and plays it through the
RPi4's audio output in real time, with no buffering delay beyond one chunk.

Hardware: Raspberry Pi 4 Model B, Pi OS Light + XFCE
Tested:   Python 3.9+, pyserial, pyaudio

Install:
    sudo apt-get install -y libportaudio2 portaudio19-dev python3-dev
    pip3 install pyserial pyaudio

UART setup (do this once):
    sudo raspi-config
      → Interface Options → Serial Port
         "Login shell over serial?" → No
         "Serial port hardware enabled?" → Yes
    Reboot.

Wiring:
    ESP32 GPIO17 (TX2)  →  RPi4 GPIO15 / physical pin 10 (RXD)
    ESP32 GND           →  RPi4 GND    / physical pin  6

Usage:
    python3 rpi4_audio_player.py

    The script will sit quietly until audio arrives from the ESP32,
    play it in real time, then wait for the next session.
    Press Ctrl+C to exit.

Frame protocol (matches esp32_audio_uart_streamer_v5.ino):
    [0xAA][0x55][LEN_H][LEN_L][PCM_DATA ...]
    LEN = byte count of PCM block (= samples × 2 for int16).
    LEN = 0 → end-of-stream marker (ESP32 going to sleep).
"""

import serial
import queue
import threading
import signal
import sys
import time

try:
    import pyaudio
except ImportError:
    print("ERROR: pyaudio not found.")
    print("  sudo apt-get install libportaudio2 portaudio19-dev")
    print("  pip3 install pyaudio")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SERIAL_PORT   = "/dev/serial0"   # UART0 on RPi4 after raspi-config enable
BAUD_RATE     = 460800           # Must match ESP32 UART_BAUD

SAMPLE_RATE   = 16000            # Must match ESP32 SAMPLE_RATE
CHANNELS      = 1                # Mono
SAMPLE_WIDTH  = 2                # 16-bit = 2 bytes per sample
CHUNK_SAMPLES = 512              # Must match ESP32 CHUNK_SIZE

# PyAudio output buffer: how many frames PyAudio holds internally.
# Larger = more stable but more latency. 512 → ~32 ms at 16 kHz.
PYAUDIO_FRAMES_PER_BUFFER = CHUNK_SAMPLES

# Serial read timeout. If no byte arrives in this long, read_frame()
# returns None (timeout). Should be longer than the longest silence
# you expect between chunks during an active session.
SERIAL_TIMEOUT = 4.0

# Audio queue depth. Each slot holds one CHUNK of PCM bytes (1024 bytes).
# 10 slots = ~320 ms of audio buffered maximum before old frames are dropped.
QUEUE_MAXSIZE = 10

# Sync bytes (must match ESP32 SYNC_HEADER)
SYNC1 = 0xAA
SYNC2 = 0x55

# ══════════════════════════════════════════════════════════════════════════════
# GLOBALS
# ══════════════════════════════════════════════════════════════════════════════

audio_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
_running = True          # Set to False on Ctrl+C → all threads exit cleanly

# ══════════════════════════════════════════════════════════════════════════════
# SHUTDOWN HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def _signal_handler(sig, frame):
    global _running
    print("\n[PLAYER] Ctrl+C received — shutting down...")
    _running = False

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ══════════════════════════════════════════════════════════════════════════════
# FRAME READER  (runs in a dedicated thread)
# ══════════════════════════════════════════════════════════════════════════════

def _find_sync(ser: serial.Serial) -> bool:
    """
    Scan the byte stream until [0xAA, 0x55] is found.
    Returns True on success, False if serial times out or _running goes False.
    """
    state = 0
    while _running:
        raw = ser.read(1)
        if not raw:
            return False        # Serial timeout
        b = raw[0]
        if state == 0:
            if b == SYNC1:
                state = 1
        elif state == 1:
            if b == SYNC2:
                return True     # Sync found
            elif b == SYNC1:
                state = 1       # Consecutive 0xAA — stay in state 1
            else:
                state = 0
    return False


def _read_frame(ser: serial.Serial):
    """
    Read one complete frame from the serial port.

    Returns:
        bytes  — PCM payload (may be empty only for EOS)
        None   — timeout or shutdown
        b''    — end-of-stream marker (LEN = 0)
    """
    if not _find_sync(ser):
        return None

    # 2-byte big-endian length
    len_bytes = ser.read(2)
    if len(len_bytes) < 2:
        return None

    frame_len = (len_bytes[0] << 8) | len_bytes[1]

    if frame_len == 0:
        return b""              # EOS marker

    # Read exactly frame_len bytes of PCM
    pcm = bytearray()
    remaining = frame_len
    while remaining > 0 and _running:
        chunk = ser.read(remaining)
        if not chunk:
            return None         # Timeout mid-frame — discard
        pcm += chunk
        remaining -= len(chunk)

    return bytes(pcm)


def uart_reader_thread(ser: serial.Serial):
    """
    Continuously reads frames from UART and pushes PCM bytes onto audio_queue.
    Runs as a daemon thread.
    """
    global _running

    session_active  = False
    frames_received = 0

    print("[READER] Thread started — waiting for ESP32 audio...")

    while _running:
        frame = _read_frame(ser)

        if frame is None:
            # Timeout with no data — ESP32 is sleeping or not yet awake.
            if session_active:
                print("[READER] Stream timed out — session ended.")
                session_active  = False
                frames_received = 0
            # Quietly wait for the next session; no error spam.
            continue

        if frame == b"":
            # ESP32 sent explicit EOS (going to deep sleep).
            if session_active:
                duration_s = (frames_received * CHUNK_SAMPLES) / SAMPLE_RATE
                print(f"[READER] EOS received — session done "
                      f"({frames_received} frames, ~{duration_s:.1f} s).")
            session_active  = False
            frames_received = 0
            print("[READER] ESP32 sleeping — waiting for next wake-up...")
            continue

        # Valid PCM frame
        if not session_active:
            session_active = True
            print("[READER] ▶  Audio session started — playing...")

        frames_received += 1

        # Push to audio queue (non-blocking: drop oldest if full)
        try:
            audio_queue.put_nowait(frame)
        except queue.Full:
            try:
                audio_queue.get_nowait()    # Evict oldest frame
                audio_queue.put_nowait(frame)
            except queue.Empty:
                pass

    print("[READER] Thread exiting.")


# ══════════════════════════════════════════════════════════════════════════════
# PYAUDIO PLAYBACK  (runs in a dedicated thread)
# ══════════════════════════════════════════════════════════════════════════════

def audio_player_thread():
    """
    Pulls PCM chunks from audio_queue and writes them to the PyAudio output
    stream. Blocking write — PyAudio handles timing against the DAC clock.
    """
    global _running

    pa = pyaudio.PyAudio()

    # Open an output stream matching the ESP32 audio parameters
    try:
        stream = pa.open(
            format            = pyaudio.paInt16,
            channels          = CHANNELS,
            rate              = SAMPLE_RATE,
            output            = True,
            frames_per_buffer = PYAUDIO_FRAMES_PER_BUFFER,
        )
    except OSError as e:
        print(f"[PLAYER] ERROR opening audio output: {e}")
        print("[PLAYER] Check that an audio output device is available:")
        print("         aplay -l   (list ALSA devices)")
        print("         For HDMI:  set HDMI as default in audio settings.")
        print("         For 3.5mm: same.")
        _running = False
        pa.terminate()
        return

    print(f"[PLAYER] Audio output ready — {SAMPLE_RATE} Hz, {CHANNELS}ch, 16-bit.")

    while _running:
        try:
            pcm_bytes = audio_queue.get(timeout=0.5)
            stream.write(pcm_bytes)
        except queue.Empty:
            # No audio to play — output stays silent, no underrun error
            pass
        except OSError as e:
            # ALSA underrun or device error — log and continue
            print(f"[PLAYER] Audio write error: {e}")

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("[PLAYER] Thread exiting.")


# ══════════════════════════════════════════════════════════════════════════════
# STATUS DISPLAY  (runs on main thread)
# ══════════════════════════════════════════════════════════════════════════════

def status_loop():
    """
    Prints a one-line status every 5 seconds so you can confirm the script
    is alive even when no audio is arriving.
    """
    last_print = time.time()
    while _running:
        now = time.time()
        if now - last_print >= 5.0:
            qsize = audio_queue.qsize()
            if qsize > 0:
                print(f"[STATUS] Queue: {qsize}/{QUEUE_MAXSIZE} chunks buffered — playing.")
            else:
                print("[STATUS] Idle — waiting for ESP32 wake-up...")
            last_print = now
        time.sleep(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  ESP32 → RPi4 Real-Time UART Audio Player")
    print("=" * 60)
    print(f"  Port:        {SERIAL_PORT}  @  {BAUD_RATE} baud")
    print(f"  Audio:       {SAMPLE_RATE} Hz  mono  16-bit")
    print(f"  Chunk:       {CHUNK_SAMPLES} samples = "
          f"{1000 * CHUNK_SAMPLES / SAMPLE_RATE:.0f} ms per frame")
    print(f"  Queue depth: {QUEUE_MAXSIZE} frames = "
          f"{1000 * QUEUE_MAXSIZE * CHUNK_SAMPLES / SAMPLE_RATE:.0f} ms max buffer")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)
    print()

    # Open serial port
    try:
        ser = serial.Serial(
            port      = SERIAL_PORT,
            baudrate  = BAUD_RATE,
            bytesize  = serial.EIGHTBITS,
            parity    = serial.PARITY_NONE,
            stopbits  = serial.STOPBITS_ONE,
            timeout   = SERIAL_TIMEOUT,
        )
        ser.reset_input_buffer()
        print(f"[MAIN] Serial port {SERIAL_PORT} opened.")
    except serial.SerialException as e:
        print(f"[MAIN] Cannot open {SERIAL_PORT}: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Run raspi-config → Interface → Serial Port")
        print("     Login shell: NO   Hardware enabled: YES   then reboot.")
        print("  2. Check wiring: ESP32 GPIO17 → RPi4 pin 10 (RXD)")
        print("  3. Check permissions: sudo usermod -aG dialout $USER")
        sys.exit(1)

    # Start background threads (daemon=True → they die when main exits)
    t_reader = threading.Thread(
        target=uart_reader_thread, args=(ser,), daemon=True, name="UARTReader"
    )
    t_player = threading.Thread(
        target=audio_player_thread, daemon=True, name="AudioPlayer"
    )

    t_reader.start()
    t_player.start()

    # Run status display on main thread until Ctrl+C
    try:
        status_loop()
    except KeyboardInterrupt:
        pass  # _signal_handler already set _running = False

    # Give threads a moment to exit cleanly
    print("[MAIN] Waiting for threads to finish...")
    t_reader.join(timeout=3.0)
    t_player.join(timeout=3.0)

    ser.close()
    print("[MAIN] Done.")


if __name__ == "__main__":
    main()
