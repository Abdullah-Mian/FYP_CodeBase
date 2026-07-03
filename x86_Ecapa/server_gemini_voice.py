"""
Gemini Voice Assistant — PC Server  (v3)
=========================================
Install:  pip install google-genai numpy scipy

Fixes in this version:
  1. TTS FIXED: pass ONLY plain text to TTS model — no system prompt,
     no Contents wrapper — just a bare string. That's what caused the 400 error.
  2. Gemini STT: explicitly asks for transcription then answer
  3. All errors printed clearly, EOS always sent so ESP32 never hangs
"""

import socket
import threading
import io
import wave
import base64
import time
import traceback
import numpy as np
from math import gcd
from scipy.signal import resample_poly
from google import genai
from google.genai import types

# ─── CONFIG ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyB_ZLDE3Q3Rp3CsnBtL5mwS463Qbl54nm8"
MIC_PORT       = 5008
SPK_PORT       = 5009
MIC_RATE       = 16000
TTS_RATE       = 24000
OUT_RATE       = 16000
CHUNK_SAMPLES  = 256
# ─────────────────────────────────────────────────────────────────────────────

client = genai.Client(api_key=GEMINI_API_KEY)

EOS_FRAME = bytes([0xAA, 0x55, 0x00, 0x00])

def parse_frame(data: bytes):
    if len(data) < 4 or data[0] != 0xAA or data[1] != 0x55:
        return b"", False
    payload_len = (data[2] << 8) | data[3]
    if payload_len == 0:
        return b"", True
    return data[4: 4 + payload_len], False

def make_frame(pcm_bytes: bytes) -> bytes:
    n = len(pcm_bytes)
    return bytes([0xAA, 0x55, (n >> 8) & 0xFF, n & 0xFF]) + pcm_bytes

def resample_audio(pcm_bytes: bytes, src: int, dst: int) -> bytes:
    if src == dst:
        return pcm_bytes
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    g = gcd(src, dst)
    resampled = resample_poly(samples, dst // g, src // g)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

def build_wav(pcm_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(MIC_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

def send_eos(spk_sock, esp32_ip):
    try:
        spk_sock.sendto(EOS_FRAME, (esp32_ip, SPK_PORT))
    except Exception:
        pass

def stream_to_esp32(spk_sock, pcm_16k: bytes, esp32_ip: str):
    chunk_bytes = CHUNK_SAMPLES * 2
    sent = 0
    for i in range(0, len(pcm_16k), chunk_bytes):
        chunk = pcm_16k[i:i + chunk_bytes]
        spk_sock.sendto(make_frame(chunk), (esp32_ip, SPK_PORT))
        sent += len(chunk)
        time.sleep(0.012)
    send_eos(spk_sock, esp32_ip)
    print(f"[SEND] Done — {sent} bytes + EOS")

def process_utterance(pcm_bytes: bytes, esp32_ip: str, spk_sock: socket.socket):
    try:
        duration = len(pcm_bytes) / (MIC_RATE * 2)
        print(f"\n{'='*60}")
        print(f"[GEMINI] {len(pcm_bytes)} bytes = {duration:.2f}s → transcribing...")

        # ── Step 1: Transcribe + Answer ───────────────────────────────────
        wav_bytes = build_wav(pcm_bytes)
        audio_b64 = base64.b64encode(wav_bytes).decode()

        stt_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[
                    types.Part(text=(
                        "Listen to this audio and do two things:\n"
                        "1. Transcribe exactly what was said.\n"
                        "2. Answer the question clearly in 1-3 short sentences.\n"
                        "If the question needs current information, search the web.\n"
                        "Reply format:\n"
                        "HEARD: <transcription>\n"
                        "ANSWER: <your answer>"
                    )),
                    types.Part(inline_data=types.Blob(
                        mime_type="audio/wav",
                        data=audio_b64
                    ))
                ])
            ],
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )

        full_text = stt_response.text.strip()
        print(f"\n[GEMINI RESPONSE]\n{full_text}\n")

        # Extract the ANSWER line for TTS
        answer_text = ""
        for line in full_text.splitlines():
            if line.upper().startswith("ANSWER:"):
                answer_text = line[len("ANSWER:"):].strip()
                break
        if not answer_text:
            answer_text = full_text   # fallback: use whole response

        if not answer_text:
            print("[WARN] Empty answer — sending EOS only")
            send_eos(spk_sock, esp32_ip)
            return

        print(f"[TTS] Speaking: {answer_text[:100]}...")

        # ── Step 2: TTS ───────────────────────────────────────────────────
        # CRITICAL FIX: TTS model only accepts a plain text string as contents.
        # Do NOT use types.Content(), types.Part(), or any system prompt here.
        # Passing anything other than a plain string causes 400 INVALID_ARGUMENT.
        tts_resp = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=answer_text,          # ← plain string ONLY
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Kore"
                        )
                    )
                )
            )
        )

        raw_b64 = tts_resp.candidates[0].content.parts[0].inline_data.data
        tts_24k = base64.b64decode(raw_b64)
        tts_16k = resample_audio(tts_24k, TTS_RATE, OUT_RATE)
        print(f"[TTS] {len(tts_24k)//2} samples@24kHz → {len(tts_16k)//2}@16kHz")

        # ── Step 3: Stream to ESP32 ───────────────────────────────────────
        print(f"[SEND] Streaming to {esp32_ip}:{SPK_PORT} ...")
        stream_to_esp32(spk_sock, tts_16k, esp32_ip)
        print(f"{'='*60}\n")

    except Exception:
        print("[ERROR] Exception in process_utterance:")
        traceback.print_exc()
        send_eos(spk_sock, esp32_ip)   # always send EOS so ESP32 can sleep

# ── shared state for busy flag ─────────────────────────────────────────────
busy_lock = threading.Lock()
busy      = False

def main():
    global busy

    mic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mic_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    mic_sock.bind(("", MIC_PORT))

    spk_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    audio_buf = b""
    pkt_count = 0

    print("=" * 60)
    print("  Gemini Voice Assistant Server  (v3)")
    print(f"  Mic  UDP :{MIC_PORT}  |  Speaker UDP :{SPK_PORT}")
    print("=" * 60)
    print("  Waiting for ESP32 packets...\n")

    while True:
        try:
            data, addr = mic_sock.recvfrom(4096)
            esp32_ip   = addr[0]
            pkt_count += 1

            if pkt_count <= 5 or pkt_count % 100 == 0:
                print(f"[UDP] pkt #{pkt_count:4d}  from {esp32_ip}  len={len(data)}")

            pcm, is_eos = parse_frame(data)

            if is_eos:
                dur = len(audio_buf) / (MIC_RATE * 2)
                print(f"[EOS] buf={len(audio_buf)}B  dur={dur:.2f}s")

                with busy_lock:
                    currently_busy = busy

                if currently_busy:
                    print("[EOS] Still processing previous utterance — dropping")
                    audio_buf = b""
                elif len(audio_buf) > MIC_RATE * 2 * 0.3:
                    buf_copy  = audio_buf
                    audio_buf = b""

                    with busy_lock:
                        busy = True

                    def run(b, ip):
                        global busy
                        try:
                            process_utterance(b, ip, spk_sock)
                        finally:
                            with busy_lock:
                                busy = False

                    threading.Thread(target=run, args=(buf_copy, esp32_ip), daemon=True).start()
                else:
                    print("[EOS] Clip too short — ignored, sending EOS back")
                    audio_buf = b""
                    send_eos(spk_sock, esp32_ip)

            elif pcm:
                audio_buf += pcm

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    main()
