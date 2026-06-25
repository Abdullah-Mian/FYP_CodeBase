#!/usr/bin/env python3
"""
Raspberry Pi 4 - Audio Receiver from ESP32 via UART
====================================================
Receives framed 16-bit PCM audio over UART at 921600 baud,
saves each voice session as a timestamped WAV file.

Prerequisites:
    sudo raspi-config -> Interface -> Serial Port
        Login shell over serial: NO
        Serial port hardware: YES
    pip3 install pyserial numpy

Wiring:
    ESP32 GPIO17 (TX2) -> RPi4 GPIO15 (RXD, Pin 10)
    ESP32 GND          -> RPi4 GND

Frame protocol:
    [0xAA][0x55][LEN_H][LEN_L][PCM_DATA...]
    LEN=0 means end-of-stream.
"""

import serial
import wave
import struct
import time
import os
import sys
import signal
from datetime import datetime

# ==================== CONFIGURATION ====================
SERIAL_PORT = "/dev/serial0"   # RPi4 UART
BAUD_RATE = 460800
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2               # 16-bit = 2 bytes
OUTPUT_DIR = os.path.expanduser("~/audio_recordings")

# Sync bytes
SYNC_BYTE_1 = 0xAA
SYNC_BYTE_2 = 0x55

# Timeout: if no frame arrives in this many seconds, save and reset
RECEIVE_TIMEOUT = 3.0


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_filename():
    """Generate a timestamped filename for the WAV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"recording_{timestamp}.wav")


def save_wav(filename, pcm_data):
    """Save raw PCM data as a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    duration = len(pcm_data) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
    print(f"[PI] Saved: {filename} ({duration:.2f}s, {len(pcm_data)} bytes)")
    return filename


def find_sync(ser):
    """
    Scan serial stream for sync header [0xAA, 0x55].
    Returns True when found, False on timeout/error.
    """
    state = 0
    while True:
        byte = ser.read(1)
        if len(byte) == 0:
            return False  # Timeout
        b = byte[0]
        if state == 0:
            if b == SYNC_BYTE_1:
                state = 1
        elif state == 1:
            if b == SYNC_BYTE_2:
                return True
            elif b == SYNC_BYTE_1:
                state = 1  # Could be start of new sync
            else:
                state = 0


def read_frame(ser):
    """
    Read one audio frame from serial.
    Returns PCM bytes, or None on timeout, or b'' for end-of-stream.
    """
    if not find_sync(ser):
        return None  # Timeout

    # Read 2-byte length header
    len_bytes = ser.read(2)
    if len(len_bytes) < 2:
        return None

    frame_len = (len_bytes[0] << 8) | len_bytes[1]

    if frame_len == 0:
        return b''  # End-of-stream marker

    # Read PCM data
    pcm_data = b''
    remaining = frame_len
    while remaining > 0:
        chunk = ser.read(remaining)
        if len(chunk) == 0:
            return None  # Timeout mid-frame
        pcm_data += chunk
        remaining -= len(chunk)

    return pcm_data


def main():
    ensure_output_dir()

    print(f"[PI] Audio Receiver Starting")
    print(f"[PI] Serial: {SERIAL_PORT} @ {BAUD_RATE} baud")
    print(f"[PI] Output: {OUTPUT_DIR}")
    print(f"[PI] Waiting for audio from ESP32...")
    print()

    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=RECEIVE_TIMEOUT,  # Read timeout
    )
    ser.reset_input_buffer()

    # Graceful shutdown
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        print("\n[PI] Shutting down...")
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    audio_accumulator = bytearray()
    session_active = False

    while running:
        frame = read_frame(ser)

        if frame is None:
            # Timeout - if we have accumulated audio, save it
            if session_active and len(audio_accumulator) > 0:
                filename = generate_filename()
                save_wav(filename, bytes(audio_accumulator))
                audio_accumulator = bytearray()
                session_active = False
                print("[PI] Session ended (timeout). Waiting for next...")
            continue

        if frame == b'':
            # End-of-stream from ESP32
            if len(audio_accumulator) > 0:
                filename = generate_filename()
                save_wav(filename, bytes(audio_accumulator))
                audio_accumulator = bytearray()
            session_active = False
            print("[PI] ESP32 going to sleep. Waiting for next wake-up...")
            continue

        # Valid audio frame
        if not session_active:
            session_active = True
            print(f"[PI] New audio session started at {datetime.now().strftime('%H:%M:%S')}")

        audio_accumulator.extend(frame)

        # Progress indicator
        duration = len(audio_accumulator) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
        sys.stdout.write(f"\r[PI] Recording: {duration:.1f}s ({len(audio_accumulator)} bytes)")
        sys.stdout.flush()

    # Cleanup: save any remaining audio
    if len(audio_accumulator) > 0:
        filename = generate_filename()
        save_wav(filename, bytes(audio_accumulator))

    ser.close()
    print("[PI] Receiver stopped.")


if __name__ == "__main__":
    main()
