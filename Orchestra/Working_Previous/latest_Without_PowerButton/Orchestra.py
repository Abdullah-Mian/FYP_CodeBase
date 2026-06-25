#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Biometric Orchestrator  v3.0  –  Orchestra.py                            ║
║                                                                            ║
║   Process layout (one per physical core):                                  ║
║     Core 0  →  Orchestra.py  (this file)  WebSocket + ZeroConf             ║
║     Core 1  →  face_worker   (main.py logic)                               ║
║     Core 2  →  voice_worker  (rpi4_ecapa_voice_biometric_v2.py logic)      ║
║     Core 3  →  free for OS / camera / UART interrupt handling              ║
║                                                                            ║
║   Key behaviours:                                                          ║
║     • Face: autonomous — verifies whenever a face is in FOV                ║
║     • Voice: autonomous — listens on UART, verifies when audio arrives     ║
║       Default TFT state: "STANDBY"                                         ║
║       Audio starts arriving → "LISTENING"                                  ║
║       Audio stops → "PROCESSING AUDIO: Xs"                                 ║
║       Result → "GRANTED: name (score)" or "DENIED: name (score)"           ║
║       After cooldown → back to "STANDBY"                                   ║
║     • Every attempt logged: method, timestamp, confidence, details → CSV   ║
║     • Server never self-disconnects; self-recoverable, never hangs         ║
║     • TFT display of results is robust (try/except, time-bounded)          ║
║     • Commands from client trigger the same operations as the standalone   ║
║       terminal menus in main.py / rpi4_ecapa_voice_biometric_v2.py         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import os
import sys
import csv
import time
import signal
import socket
import logging
import traceback
import pickle
import glob
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from datetime import datetime

import websockets
from websockets.server import serve
from zeroconf import IPVersion, ServiceInfo, Zeroconf

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(processName)-22s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")

# ── CPU affinity map ──────────────────────────────────────────────────────────
CORE_SERVER = {0}
CORE_FACE   = {1}
CORE_VOICE  = {2}

# ── Autonomous face identification ───────────────────────────────────────────
AUTO_FACE_RETRY_S   = 2.0
GRANTED_IOU_THRESH  = 0.3

# ── Voice TFT status display duration ────────────────────────────────────────
VOICE_RESULT_DISPLAY_S = 4.0   # show result before returning to STANDBY

# ── Log file ─────────────────────────────────────────────────────────────────
LOG_CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biometric_log.csv")
LOG_CSV_COLUMNS = [
    "timestamp", "method", "event", "user", "confidence",
    "second_best_user", "second_best_score", "reason",
    "audio_duration_s", "inference_time_ms", "detail",
]


def _pin(cores: set):
    try:
        os.sched_setaffinity(0, cores)
        log.info("Pinned to core(s) %s", cores)
    except AttributeError:
        log.warning("sched_setaffinity not available on this OS")
    except PermissionError:
        log.warning("No permission to set CPU affinity — running unpinned.")


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED STATE + LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _make_shared_state(manager):
    return {
        "logs":         manager.list(),
        "users":        manager.dict(),
        "voice_status": manager.dict({"text": "STANDBY"}),
    }

def _set_voice_status(shared_display, text: str):
    try:
        shared_display["text"] = text
    except Exception:
        pass

def _get_voice_status(shared_display) -> str:
    try:
        return shared_display.get("text", "STANDBY")
    except Exception:
        return "STANDBY"


def _init_csv_log():
    if not os.path.exists(LOG_CSV_FILE):
        try:
            with open(LOG_CSV_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
                writer.writeheader()
        except Exception as e:
            log.error("Failed to create CSV log: %s", e)


def _log_event(shared_logs, method: str, event: str,
               user: str = "", confidence: float = 0.0,
               second_best_user: str = "", second_best_score: float = 0.0,
               reason: str = "", audio_duration_s: float = 0.0,
               inference_time_ms: float = 0.0, detail: str = ""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp":         ts,
        "method":            method,
        "event":             event,
        "user":              user,
        "confidence":        round(confidence, 4) if confidence else 0.0,
        "second_best_user":  second_best_user,
        "second_best_score": round(second_best_score, 4) if second_best_score else 0.0,
        "reason":            reason,
        "audio_duration_s":  round(audio_duration_s, 2) if audio_duration_s else 0.0,
        "inference_time_ms": round(inference_time_ms, 1) if inference_time_ms else 0.0,
        "detail":            detail,
    }
    try:
        shared_logs.append(entry)
    except Exception:
        pass
    try:
        with open(LOG_CSV_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_CSV_COLUMNS)
            writer.writerow(entry)
    except Exception as e:
        log.error("Failed to write CSV log entry: %s", e)


def _iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  FACE WORKER  (Core 1)
#
#  Uses main.py's exact functions: enroll_face, capture_probe, match_probe,
#  check_enrollment_duplicate, load_database, save_database, etc.
#  Commands from client map 1:1 to what main.py's terminal menu does:
#    "enroll" → same as pressing 'a' (ADD) in main.py
#    "verify" → same as pressing 'v' (VERIFY) in main.py
#    "delete" → same as pressing 'd' (DELETE) in main.py
#    "list"   → same as pressing 'l' (LIST) in main.py
# ─────────────────────────────────────────────────────────────────────────────

def face_worker_process(cmd_q: Queue, res_q: Queue, busy_evt: Event,
                        shared_logs, shared_users, shared_display):
    _pin(CORE_FACE)
    mp.current_process().name = "face_worker"
    log.info("[FACE] Worker started on core %s", CORE_FACE)

    try:
        import cv2
        import numpy as np
        sys.path.insert(0, os.path.dirname(__file__))
        import main as _main_mod
        from main import (
            TFTDisplay, PiCamera, BlazeFaceDetector,
            enroll_face, capture_probe, match_probe,
            check_enrollment_duplicate,
            load_database, save_database,
            l2_normalize,
            SIMILARITY_THRESHOLD, MIN_SCORE_GAP, MIN_GALLERY_SIZE,
            DATABASE_FILE,
            MAX_FACES_IN_FRAME, ENROLL_DUPLICATE_THRESHOLD,
        )
    except ImportError as e:
        log.error("[FACE] Import failed: %s", e)
        res_q.put({"worker": "face", "status": "fatal", "error": str(e)})
        return

    display = camera = detector = None
    db = {}

    try:
        display  = TFTDisplay()
        camera   = PiCamera()
        if not camera.start():
            raise RuntimeError("Camera failed to start")
        detector = BlazeFaceDetector()
        db       = load_database()
        log.info("[FACE] Hardware ready. DB: %d identities", len(db))
    except Exception as e:
        log.error("[FACE] Hardware init failed: %s", e)
        res_q.put({"worker": "face", "status": "fatal", "error": str(e)})
        return

    res_q.put({"worker": "face", "status": "ready"})

    def _voice_status_fn():
        return _get_voice_status(shared_display)

    def _safe_tft(status_msg, duration=2.0):
        """Flash a result on TFT for duration. Never hangs."""
        try:
            t_end = time.time() + duration
            while time.time() < t_end:
                try:
                    ok2, fr2 = camera.read_preview()
                    vtxt = _get_voice_status(shared_display)
                    if ok2:
                        display.show_frame(fr2, status=status_msg, fps=fps,
                                           voice_status=vtxt)
                except Exception:
                    pass
                time.sleep(0.03)
        except Exception:
            pass

    def _get_second_best(probe_emb, db_dict):
        """Compute all scores, return (sorted_list, second_name, second_score)."""
        try:
            scores = {}
            for n, template in db_dict.items():
                tmpl = l2_normalize(np.asarray(template, dtype=np.float32))
                scores[n] = float(np.dot(probe_emb, tmpl))
            ss = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            s2n = ss[1][0] if len(ss) > 1 else ""
            s2s = ss[1][1] if len(ss) > 1 else 0.0
            return s2n, s2s
        except Exception:
            return "", 0.0

    # ── command handler ───────────────────────────────────────────────────────
    def _handle_command(cmd: dict) -> dict:
        nonlocal db
        op   = cmd.get("op")
        name = cmd.get("name", "").strip()

        # ── ENROLL (same as 'a' ADD in main.py) ──────────────────────────
        if op == "enroll":
            if not name:
                return {"status": "error", "message": "Name required for enroll"}

            max_attempts = cmd.get("max_attempts", 50)
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                log.info("[FACE] Enroll attempt %d for '%s'", attempt, name)
                res_q.put({"worker": "face", "status": "progress",
                           "step": "face_enroll",
                           "message": f"Attempt {attempt} — look at camera…"})

                try:
                    embedding = enroll_face(camera, detector, display,
                                            voice_status_fn=_voice_status_fn)
                except Exception as e:
                    log.warning("[FACE] enroll_face exception: %s", e)
                    embedding = None

                if embedding is None:
                    _log_event(shared_logs, "face", "ENROLL_ATTEMPT_FAILED",
                               user=name, detail=f"attempt={attempt}")
                    # Check for cancel
                    try:
                        if not cmd_q.empty():
                            peek = cmd_q.get_nowait()
                            if peek.get("op") == "cancel_enroll":
                                _log_event(shared_logs, "face", "ENROLL_CANCELLED",
                                           user=name, detail=f"after {attempt} attempts")
                                return {"status": "error",
                                        "message": f"Cancelled after {attempt} attempts"}
                            else:
                                cmd_q.put(peek)
                    except Exception:
                        pass
                    time.sleep(0.5)
                    continue

                # Duplicate check (same as main.py does)
                dup_ok, dup_name, dup_score = check_enrollment_duplicate(embedding, db)
                if not dup_ok:
                    _log_event(shared_logs, "face", "ENROLL_REJECTED_DUPLICATE",
                               user=name, confidence=dup_score,
                               second_best_user=dup_name,
                               reason="duplicate")
                    return {"status": "error",
                            "message": f"Rejected — matches '{dup_name}' at "
                                       f"{dup_score:.4f} (threshold "
                                       f"{ENROLL_DUPLICATE_THRESHOLD})"}

                # Save (same as main.py)
                db[name] = embedding
                save_database(db)
                shared_users[name] = shared_users.get(name, "") or "face"
                _log_event(shared_logs, "face", "ENROLLED",
                           user=name, detail=f"after {attempt} attempt(s)")
                _safe_tft(f"ENROLLED: {name}", duration=2.0)
                return {"status": "success", "user": name}

            _log_event(shared_logs, "face", "ENROLL_FAILED",
                       user=name, detail=f"max {max_attempts} attempts exhausted")
            return {"status": "error",
                    "message": f"Failed after {max_attempts} attempts"}

        # ── VERIFY (same as 'v' VERIFY in main.py) ───────────────────────
        elif op == "verify":
            try:
                probe = capture_probe(camera, detector, display,
                                      voice_status_fn=_voice_status_fn)
            except Exception as e:
                log.warning("[FACE] capture_probe error: %s", e)
                probe = None

            if probe is None:
                _log_event(shared_logs, "face", "VERIFY_FAILED",
                           reason="capture_failed")
                return {"status": "error",
                        "message": "Probe capture failed"}

            granted, best_name, best_score, reason = match_probe(probe, db)
            s2n, s2s = _get_second_best(probe, db)
            event_type = "GRANTED" if granted else "DENIED"
            _log_event(shared_logs, "face", event_type,
                       user=best_name, confidence=best_score,
                       second_best_user=s2n, second_best_score=s2s,
                       reason=reason)

            if granted:
                _safe_tft(f"GRANTED:{best_name} ({best_score:.2f})", 2.0)
            else:
                _safe_tft(f"DENIED (closest:{best_name} {best_score:.2f})", 2.0)

            return {"status": "success", "granted": granted,
                    "user": best_name, "score": round(best_score, 4),
                    "reason": reason}

        # ── DELETE (same as 'd' DELETE in main.py) ────────────────────────
        elif op == "delete":
            if not name:
                return {"status": "error", "message": "Name required"}
            if name not in db:
                return {"status": "error", "message": f"'{name}' not in face DB"}
            del db[name]
            save_database(db)
            _log_event(shared_logs, "face", "DELETED", user=name)
            return {"status": "success", "deleted": name}

        # ── LIST (same as 'l' LIST in main.py) ───────────────────────────
        elif op == "list":
            return {"status": "success", "data": list(db.keys())}

        elif op == "reload_db":
            db = load_database()
            return {"status": "success", "count": len(db)}

        elif op == "cancel_enroll":
            return {"status": "success", "message": "Nothing active to cancel"}

        # ── GET THRESHOLDS ───────────────────────────────────────
        elif op == "get_thresholds":
            return {"status": "success",
                    "face_similarity":       _main_mod.SIMILARITY_THRESHOLD,
                    "face_min_score_gap":    _main_mod.MIN_SCORE_GAP,
                    "face_enroll_duplicate": _main_mod.ENROLL_DUPLICATE_THRESHOLD,
                    "face_min_gallery":      _main_mod.MIN_GALLERY_SIZE}

        # ── SET THRESHOLDS ───────────────────────────────────────
        elif op == "set_thresholds":
            changed = []
            if "face_similarity" in cmd:
                v = float(cmd["face_similarity"])
                if 0.0 < v < 1.0:
                    _main_mod.SIMILARITY_THRESHOLD = v
                    changed.append(f"face_similarity={v:.4f}")
            if "face_min_score_gap" in cmd:
                v = float(cmd["face_min_score_gap"])
                if 0.0 <= v < 1.0:
                    _main_mod.MIN_SCORE_GAP = v
                    changed.append(f"face_min_score_gap={v:.4f}")
            if "face_enroll_duplicate" in cmd:
                v = float(cmd["face_enroll_duplicate"])
                if 0.0 < v < 1.0:
                    _main_mod.ENROLL_DUPLICATE_THRESHOLD = v
                    changed.append(f"face_enroll_duplicate={v:.4f}")
            if "face_min_gallery" in cmd:
                v = int(cmd["face_min_gallery"])
                if 0 <= v <= 10:
                    _main_mod.MIN_GALLERY_SIZE = v
                    changed.append(f"face_min_gallery={v}")
            log.info("[FACE] Thresholds updated: %s", ", ".join(changed) if changed else "none")
            return {"status": "success", "changed": changed}

        else:
            return {"status": "error", "message": f"Unknown op '{op}'"}

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────
    fps_t             = time.time()
    fps               = 0.0
    fc                = 0
    last_attempt_time = 0.0
    granted_box       = None
    granted_name      = None

    try:
        while True:
            # ── Priority 1: command from server ──────────────────────────
            try:
                if not cmd_q.empty():
                    cmd = cmd_q.get_nowait()
                    if cmd.get("op") == "stop":
                        log.info("[FACE] Stop received"); break
                    busy_evt.set()
                    log.info("[FACE] Cmd: %s", cmd)
                    try:
                        result = _handle_command(cmd)
                    except Exception as exc:
                        log.error("[FACE] Cmd error: %s", traceback.format_exc())
                        result = {"status": "error", "message": str(exc)}
                        _log_event(shared_logs, "face", "ERROR", detail=str(exc)[:200])
                    res_q.put({"worker": "face", **result})
                    busy_evt.clear()
                    continue
            except Exception:
                time.sleep(0.1); continue

            # ── Priority 2: autonomous face verify ───────────────────────
            try:
                ok, frame = camera.read_preview()
            except Exception:
                time.sleep(0.05); continue
            if not ok:
                time.sleep(0.005); continue

            fc += 1
            if fc % 30 == 0:
                dt = time.time() - fps_t
                if dt > 0.001: fps = 30.0 / dt
                fps_t = time.time()

            try:
                detects   = detector.detect_with_landmarks(frame)
                faces     = [d["box"] for d in detects]
                voice_txt = _get_voice_status(shared_display)
            except Exception:
                time.sleep(0.01); continue

            # Multiple faces
            if len(detects) > MAX_FACES_IN_FRAME:
                granted_box = granted_name = None
                try: display.show_frame(frame, faces=faces,
                        status="MULTIPLE FACES", fps=fps, voice_status=voice_txt)
                except Exception: pass
                time.sleep(0.01); continue

            # No face
            if not detects:
                granted_box = granted_name = None
                try: display.show_frame(frame, faces=faces, status="WATCHING",
                        fps=fps, voice_status=voice_txt)
                except Exception: pass
                time.sleep(0.01); continue

            current_box = detects[0]["box"]
            still_granted = (granted_box is not None
                             and _iou(current_box, granted_box) >= GRANTED_IOU_THRESH)

            if still_granted:
                try: display.show_frame(frame, faces=faces,
                        status=f"GRANTED:{granted_name}", fps=fps,
                        voice_status=voice_txt)
                except Exception: pass
                time.sleep(0.01); continue

            if granted_box is not None and not still_granted:
                granted_box = granted_name = None

            now = time.time()
            ready = (len(db) >= MIN_GALLERY_SIZE
                     and not busy_evt.is_set()
                     and (now - last_attempt_time) >= AUTO_FACE_RETRY_S)

            if not ready:
                try: display.show_frame(frame, faces=faces, status="WATCHING",
                        fps=fps, voice_status=voice_txt)
                except Exception: pass
                time.sleep(0.01); continue

            # ── Attempt autonomous verify ────────────────────────────────
            try: display.show_frame(frame, faces=faces, status="IDENTIFYING…",
                    fps=fps, voice_status=voice_txt)
            except Exception: pass

            try:
                probe = capture_probe(camera, detector, display,
                                      voice_status_fn=_voice_status_fn,
                                      abort_fn=lambda: not cmd_q.empty())
            except Exception as e:
                log.warning("[FACE][AUTO] capture error: %s", e)
                probe = None

            if not cmd_q.empty():
                last_attempt_time = time.time(); continue

            if probe is not None:
                try:
                    granted, name, score, reason = match_probe(probe, db)
                except Exception:
                    last_attempt_time = time.time(); continue

                s2n, s2s = _get_second_best(probe, db)
                tag = "GRANTED" if granted else "DENIED"

                if granted:
                    status_msg = f"GRANTED:{name} ({score:.2f})"
                    granted_box, granted_name = current_box, name
                else:
                    status_msg = f"UNKNOWN (closest:{name} {score:.2f})"

                log.info("[FACE][AUTO] %s %s %.4f", tag, name, score)
                _log_event(shared_logs, "face", tag,
                           user=name, confidence=score,
                           second_best_user=s2n, second_best_score=s2s,
                           reason=reason, detail="auto")

                _safe_tft(status_msg, 2.0)
            else:
                _log_event(shared_logs, "face", "ATTEMPT_NO_CAPTURE",
                           detail="auto — probe was None")

            last_attempt_time = time.time()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error("[FACE] Main loop crash: %s\n%s", e, traceback.format_exc())
    finally:
        try: camera.stop()
        except Exception: pass
        try: detector.close()
        except Exception: pass
        try: save_database(db)
        except Exception: pass
        try: display.clear()
        except Exception: pass
        log.info("[FACE] Worker shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
#  VOICE WORKER  (Core 2)
#
#  Uses rpi4_ecapa_voice_biometric_v2.py's exact functions and classes:
#    UARTReceiver, ONNXAuthenticator, enrolled_speakers, raw_to_features,
#    save_debug_wav, etc.
#  Commands from client map 1:1 to what the standalone terminal menu does:
#    "enroll" → same as choosing '1' (Enroll New Speaker)
#    "verify" → same as choosing '2' (Verify / Identify Speaker)
#    "delete" → same as choosing '4' (Delete Speaker)
#    "list"   → same as choosing '3' (List Enrolled Speakers)
#
#  Autonomous mode:
#    Default state: STANDBY (displayed on TFT via shared_display)
#    UARTReceiver.receive_session() blocks waiting for ESP32 audio.
#    As soon as any audio frame arrives → TFT shows "LISTENING"
#    When session completes → "PROCESSING AUDIO: X.Xs"
#    After inference → result displayed, then back to STANDBY
#
#  To show "LISTENING" the instant audio starts (not after the session ends),
#  we use a modified receive loop that updates shared_display in real-time.
# ─────────────────────────────────────────────────────────────────────────────

def voice_worker_process(cmd_q: Queue, res_q: Queue, busy_evt: Event,
                         shared_logs, shared_users, shared_display):
    _pin(CORE_VOICE)
    mp.current_process().name = "voice_worker"
    log.info("[VOICE] Worker started on core %s", CORE_VOICE)

    try:
        import numpy as np
        import serial
        sys.path.insert(0, os.path.dirname(__file__))
        import rpi4_ecapa_voice_biometric_v2 as _voice_mod
        from rpi4_ecapa_voice_biometric_v2 import (
            UARTReceiver, ONNXAuthenticator,
            enrolled_speakers, raw_to_features,
            save_debug_wav,
            VOICEPRINT_DB, ONNX_MODEL,
            VERIFY_THRESHOLD, DUPLICATE_THRESHOLD,
            ENROLL_MIN_S, ENROLL_MAX_S,
            VERIFY_MIN_S, VERIFY_MAX_S,
            SAMPLE_RATE, SAMPLE_WIDTH,
        )
    except ImportError as e:
        log.error("[VOICE] Import failed: %s", e)
        res_q.put({"worker": "voice", "status": "fatal", "error": str(e)})
        return

    try:
        if not os.path.exists(ONNX_MODEL):
            raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL}")
        os.makedirs(VOICEPRINT_DB, exist_ok=True)
        auth = ONNXAuthenticator(ONNX_MODEL)
        uart = UARTReceiver()
        log.info("[VOICE] Hardware ready. Speakers: %d", len(enrolled_speakers()))
    except Exception as e:
        log.error("[VOICE] Hardware init failed: %s", e)
        res_q.put({"worker": "voice", "status": "fatal", "error": str(e)})
        return

    res_q.put({"worker": "voice", "status": "ready"})

    def _vs(text: str):
        """Set voice status on TFT (shared across processes)."""
        _set_voice_status(shared_display, text)
        log.debug("[VOICE][TFT] %s", text)

    _vs("STANDBY")

    # ── Custom receive_session that updates TFT in real time ─────────────────
    def _receive_with_status(min_s, max_s, label="audio"):
        """
        Wraps the low-level UART frame reading to give real-time TFT updates:
          1. Pre-loop: set "STANDBY" (already set)
          2. First frame arrives: switch to "LISTENING"
          3. During receive: update with elapsed time
          4. Session ends: switch to "PROCESSING AUDIO: X.Xs"

        Returns raw PCM bytes or None (same contract as UARTReceiver.receive_session).
        """
        # Flush stale bytes (same as the original)
        uart._ser.reset_input_buffer()

        max_bytes = int(max_s * SAMPLE_RATE * SAMPLE_WIDTH)
        min_bytes = int(min_s * SAMPLE_RATE * SAMPLE_WIDTH)
        accum     = bytearray()
        started   = False
        start_time = None

        log.info("[VOICE] Waiting for %s… min=%.0fs max=%.0fs", label, min_s, max_s)

        while True:
            # Check for command pre-emption
            if not cmd_q.empty():
                if accum:
                    break  # Return what we have
                return None  # No audio yet, bail

            frame = uart._read_frame()

            if frame is None:
                if started and accum:
                    break  # Timeout after data → session end
                continue   # Timeout before first frame → keep waiting

            if frame == b'':
                if accum:
                    break  # Clean EOS from ESP32
                continue   # EOS before any data

            if not started:
                started = True
                start_time = time.time()
                _vs("LISTENING")
                log.info("[VOICE] Audio session started")

            accum.extend(frame)
            dur = len(accum) / (SAMPLE_RATE * SAMPLE_WIDTH)
            _vs(f"LISTENING: {dur:.1f}s")

            if len(accum) >= max_bytes:
                break

        if not accum:
            return None

        actual_s = len(accum) / (SAMPLE_RATE * SAMPLE_WIDTH)
        if len(accum) < min_bytes:
            _vs(f"TOO SHORT: {actual_s:.1f}s < {min_s:.0f}s")
            log.warning("[VOICE] Audio too short: %.2fs", actual_s)
            time.sleep(1.5)
            _vs("STANDBY")
            return None

        accum = accum[:max_bytes]
        final_s = len(accum) / (SAMPLE_RATE * SAMPLE_WIDTH)
        _vs(f"PROCESSING AUDIO: {final_s:.1f}s")
        log.info("[VOICE] Session complete: %.2fs — processing…", final_s)
        return bytes(accum)

    # ── Embedding + matching helpers ─────────────────────────────────────────
    def _embed_audio(raw: bytes):
        features, audio = raw_to_features(raw)
        return auth.embed(features), audio

    def _best_match(live_emb):
        speakers = enrolled_speakers()
        if not speakers:
            return None, -1.0, None, -1.0
        scores = []
        for p in speakers:
            stored = np.load(p)
            s = auth.cosine(stored, live_emb)
            n = os.path.basename(p)[:-4]
            scores.append((n, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = scores[0]
        s2_name = scores[1][0] if len(scores) > 1 else ""
        s2_score = scores[1][1] if len(scores) > 1 else -1.0
        return best_name, best_score, s2_name, s2_score

    # ── command handler ───────────────────────────────────────────────────────
    def _handle_command(cmd: dict) -> dict:
        op   = cmd.get("op")
        name = (cmd.get("name") or "").lower().strip()

        # ── ENROLL (same as choosing '1' in standalone menu) ─────────────
        if op == "enroll":
            if not name:
                return {"status": "error", "message": "Name required"}
            emb_path = os.path.join(VOICEPRINT_DB, f"{name}.npy")
            if os.path.exists(emb_path):
                _vs(f"'{name}' already enrolled")
                _log_event(shared_logs, "voice", "ENROLL_REJECTED",
                           user=name, reason="already_exists")
                return {"status": "error",
                        "message": f"'{name}' already enrolled. Delete first."}

            _vs(f"ENROLLING '{name}' — speak now…")
            log.info("[VOICE] Enrolling '%s' — waiting for audio", name)

            raw = _receive_with_status(ENROLL_MIN_S, ENROLL_MAX_S,
                                       label=f"enrollment '{name}'")
            if raw is None:
                _vs(f"Enroll '{name}' FAILED — no audio")
                _log_event(shared_logs, "voice", "ENROLL_FAILED",
                           user=name, reason="no_audio")
                time.sleep(1.5); _vs("STANDBY")
                return {"status": "error", "message": "No audio received"}

            audio_dur = len(raw) / (SAMPLE_RATE * SAMPLE_WIDTH)

            try:
                t0 = time.perf_counter()
                emb, audio = _embed_audio(raw)
                inf_ms = (time.perf_counter() - t0) * 1000
            except ValueError as e:
                _vs(f"Enroll FAILED — {e}")
                _log_event(shared_logs, "voice", "ENROLL_FAILED",
                           user=name, reason="quality", audio_duration_s=audio_dur,
                           detail=str(e))
                time.sleep(1.5); _vs("STANDBY")
                return {"status": "error", "message": f"Quality check: {e}"}

            # Duplicate check (same as standalone enroll_speaker)
            best_n, best_s, _, _ = _best_match(emb)
            if best_s >= _voice_mod.DUPLICATE_THRESHOLD:
                _vs(f"DUPLICATE of '{best_n}' ({best_s:.2f})")
                _log_event(shared_logs, "voice", "ENROLL_REJECTED_DUPLICATE",
                           user=name, confidence=best_s,
                           second_best_user=best_n, reason="duplicate",
                           audio_duration_s=audio_dur, inference_time_ms=inf_ms)
                time.sleep(2.0); _vs("STANDBY")
                return {"status": "error",
                        "message": f"Duplicate — matches '{best_n}' at {best_s:.4f}"}

            np.save(emb_path, emb)
            shared_users[name] = shared_users.get(name, "") or "voice"
            _log_event(shared_logs, "voice", "ENROLLED", user=name,
                       audio_duration_s=audio_dur, inference_time_ms=inf_ms)
            _vs(f"ENROLLED: {name}")
            time.sleep(2.0); _vs("STANDBY")
            return {"status": "success", "user": name}

        # ── VERIFY (same as choosing '2' in standalone menu) ─────────────
        elif op == "verify":
            speakers = enrolled_speakers()
            if not speakers:
                _vs("No speakers enrolled")
                _log_event(shared_logs, "voice", "VERIFY_FAILED",
                           reason="no_speakers")
                time.sleep(1.5); _vs("STANDBY")
                return {"status": "error", "message": "No speakers enrolled"}

            _vs("VERIFY — speak now…")
            raw = _receive_with_status(VERIFY_MIN_S, VERIFY_MAX_S,
                                       label="verification")
            if raw is None:
                _vs("Verify FAILED — no audio")
                _log_event(shared_logs, "voice", "VERIFY_FAILED",
                           reason="no_audio")
                time.sleep(1.5); _vs("STANDBY")
                return {"status": "error", "message": "No audio received"}

            audio_dur = len(raw) / (SAMPLE_RATE * SAMPLE_WIDTH)

            try:
                t0 = time.perf_counter()
                emb, audio = _embed_audio(raw)
                inf_ms = (time.perf_counter() - t0) * 1000
            except ValueError as e:
                _vs(f"Verify FAILED — {e}")
                _log_event(shared_logs, "voice", "VERIFY_FAILED",
                           reason="quality", audio_duration_s=audio_dur,
                           detail=str(e))
                time.sleep(1.5); _vs("STANDBY")
                return {"status": "error", "message": f"Quality check: {e}"}

            best_name, best_score, s2n, s2s = _best_match(emb)
            granted = best_score >= _voice_mod.VERIFY_THRESHOLD
            tag = "GRANTED" if granted else "DENIED"

            _log_event(shared_logs, "voice", tag,
                       user=best_name or "unknown", confidence=best_score,
                       second_best_user=s2n, second_best_score=s2s,
                       reason="" if granted else "below_threshold",
                       audio_duration_s=audio_dur, inference_time_ms=inf_ms)

            _vs(f"{tag}: {best_name or 'unknown'} ({best_score:.2f})")
            time.sleep(VOICE_RESULT_DISPLAY_S); _vs("STANDBY")

            return {"status": "success", "granted": granted,
                    "user": best_name, "score": round(best_score, 4)}

        # ── DELETE (same as choosing '4' in standalone menu) ──────────────
        elif op == "delete":
            if not name:
                return {"status": "error", "message": "Name required"}
            path = os.path.join(VOICEPRINT_DB, f"{name}.npy")
            if not os.path.exists(path):
                return {"status": "error", "message": f"'{name}' not found"}
            os.remove(path)
            _log_event(shared_logs, "voice", "DELETED", user=name)
            _vs(f"Deleted '{name}'")
            time.sleep(1.0); _vs("STANDBY")
            return {"status": "success", "deleted": name}

        # ── LIST (same as choosing '3' in standalone menu) ────────────────
        elif op == "list":
            names = [os.path.basename(p)[:-4] for p in enrolled_speakers()]
            return {"status": "success", "data": names}

        # ── GET THRESHOLDS ───────────────────────────────────────
        elif op == "get_thresholds":
            return {"status": "success",
                    "voice_verify":    _voice_mod.VERIFY_THRESHOLD,
                    "voice_duplicate": _voice_mod.DUPLICATE_THRESHOLD}

        # ── SET THRESHOLDS ───────────────────────────────────────
        elif op == "set_thresholds":
            changed = []
            if "voice_verify" in cmd:
                v = float(cmd["voice_verify"])
                if 0.0 < v < 1.0:
                    _voice_mod.VERIFY_THRESHOLD = v
                    changed.append(f"voice_verify={v:.4f}")
            if "voice_duplicate" in cmd:
                v = float(cmd["voice_duplicate"])
                if 0.0 < v < 1.0:
                    _voice_mod.DUPLICATE_THRESHOLD = v
                    changed.append(f"voice_duplicate={v:.4f}")
            log.info("[VOICE] Thresholds updated: %s", ", ".join(changed) if changed else "none")
            return {"status": "success", "changed": changed}

        else:
            return {"status": "error", "message": f"Unknown op '{op}'"}

    # ── MAIN LOOP — autonomous UART listening + command pre-emption ───────────
    try:
        while True:
            # ── Check for command first (non-blocking) ───────────────────
            try:
                if not cmd_q.empty():
                    cmd = cmd_q.get_nowait()
                    if cmd.get("op") == "stop":
                        log.info("[VOICE] Stop received"); break
                    busy_evt.set()
                    log.info("[VOICE] Cmd: %s", cmd)
                    try:
                        result = _handle_command(cmd)
                    except Exception as exc:
                        log.error("[VOICE] Cmd error: %s", traceback.format_exc())
                        result = {"status": "error", "message": str(exc)}
                        _log_event(shared_logs, "voice", "ERROR",
                                   detail=str(exc)[:200])
                    res_q.put({"worker": "voice", **result})
                    busy_evt.clear()
                    continue
            except Exception as e:
                log.warning("[VOICE] Queue check error (recovering): %s", e)
                time.sleep(0.5); continue

            # ── Autonomous: listen for audio on UART ─────────────────────
            _vs("STANDBY")

            raw = _receive_with_status(VERIFY_MIN_S, VERIFY_MAX_S,
                                       label="auto-identify")

            # Command may have arrived during blocking receive
            if not cmd_q.empty():
                continue

            if raw is None:
                _vs("STANDBY")
                continue

            # Got audio — process it
            audio_dur = len(raw) / (SAMPLE_RATE * SAMPLE_WIDTH)

            speakers = enrolled_speakers()
            if not speakers:
                _vs("No speakers enrolled")
                log.debug("[VOICE][AUTO] No speakers enrolled")
                _log_event(shared_logs, "voice", "AUTO_VERIFY_SKIPPED",
                           reason="no_speakers", audio_duration_s=audio_dur)
                time.sleep(1.5); _vs("STANDBY")
                continue

            try:
                t0 = time.perf_counter()
                emb, audio = _embed_audio(raw)
                inf_ms = (time.perf_counter() - t0) * 1000
            except ValueError as e:
                log.warning("[VOICE][AUTO] Quality fail: %s", e)
                _vs(f"Audio quality issue")
                _log_event(shared_logs, "voice", "AUTO_VERIFY_FAILED",
                           reason="quality", audio_duration_s=audio_dur,
                           detail=str(e))
                time.sleep(1.5); _vs("STANDBY")
                continue

            best_name, best_score, s2n, s2s = _best_match(emb)
            granted = best_score >= _voice_mod.VERIFY_THRESHOLD
            tag = "GRANTED" if granted else "DENIED"

            log.info("[VOICE][AUTO] %s  %s  %.4f  (audio=%.1fs, inf=%.0fms)",
                     tag, best_name, best_score, audio_dur, inf_ms)
            _log_event(shared_logs, "voice", tag,
                       user=best_name or "unknown", confidence=best_score,
                       second_best_user=s2n, second_best_score=s2s,
                       reason="" if granted else "below_threshold",
                       audio_duration_s=audio_dur, inference_time_ms=inf_ms,
                       detail="auto")

            _vs(f"{tag}: {best_name or 'unknown'} ({best_score:.2f})")
            time.sleep(VOICE_RESULT_DISPLAY_S)
            _vs("STANDBY")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error("[VOICE] Main loop crash: %s\n%s", e, traceback.format_exc())
    finally:
        try: uart.close()
        except Exception: pass
        log.info("[VOICE] Worker shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
#  WEBSOCKET SERVER  (Core 0)
# ─────────────────────────────────────────────────────────────────────────────

def get_local_ip() -> str:
    candidates = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                candidates.append(ip)
        finally:
            s.close()
    except Exception:
        pass
    if not candidates:
        try:
            for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
                if not ip.startswith("127.") and not ip.startswith("169.254."):
                    candidates.append(ip)
        except Exception:
            pass
    for ip in candidates:
        if not ip.startswith("127.") and not ip.startswith("169.254."):
            return ip
    return None


async def wait_for_lan_ip(poll_interval_s: float = 60.0) -> str:
    attempt = 0
    while True:
        ip = get_local_ip()
        if ip is not None:
            if attempt > 0:
                log.info("[NET] LAN IP acquired: %s", ip)
            return ip
        attempt += 1
        log.warning("[NET] No LAN IP — retry #%d in %.0fs", attempt, poll_interval_s)
        await asyncio.sleep(poll_interval_s)


class OrchestratorServer:
    RESPONSE_TIMEOUT = 120.0

    def __init__(self, face_cmd_q, face_res_q, face_busy,
                 voice_cmd_q, voice_res_q, voice_busy,
                 shared_logs, shared_users):
        self.face_cmd_q  = face_cmd_q
        self.face_res_q  = face_res_q
        self.face_busy   = face_busy
        self.voice_cmd_q = voice_cmd_q
        self.voice_res_q = voice_res_q
        self.voice_busy  = voice_busy
        self.shared_logs = shared_logs
        self.shared_users = shared_users
        self._face_lock  = None
        self._voice_lock = None

    async def _init_locks(self):
        self._face_lock  = asyncio.Lock()
        self._voice_lock = asyncio.Lock()

    async def _dispatch(self, worker: str, cmd: dict, websocket=None) -> dict:
        if worker == "face":
            q_cmd, q_res, lock = self.face_cmd_q, self.face_res_q, self._face_lock
        else:
            q_cmd, q_res, lock = self.voice_cmd_q, self.voice_res_q, self._voice_lock

        async with lock:
            q_cmd.put(cmd)
            deadline = time.monotonic() + self.RESPONSE_TIMEOUT
            loop = asyncio.get_event_loop()
            while True:
                if time.monotonic() > deadline:
                    return {"status": "error", "message": "Worker timed out"}
                result = await loop.run_in_executor(None, _try_get, q_res)
                if result is not None:
                    if result.get("status") == "progress" and websocket:
                        try: await websocket.send(json.dumps(result))
                        except Exception: pass
                        continue
                    return result
                await asyncio.sleep(0.05)

    async def handle_client(self, websocket):
        addr = websocket.remote_address
        log.info("[WS] Client connected: %s", addr)
        _log_event(self.shared_logs, "server", "CLIENT_CONNECTED",
                   detail=str(addr))
        try:
            async for raw_msg in websocket:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps(
                        {"status": "error", "message": "Invalid JSON"}))
                    continue

                command = data.get("command", "")
                name    = data.get("name", "").strip()
                log.info("[WS] '%s' from %s", command, addr)

                try:
                    response = await self._route(command, name, data, websocket)
                except Exception as e:
                    log.error("[WS] Route error: %s", traceback.format_exc())
                    response = {"status": "error", "message": str(e)}

                try:
                    await websocket.send(json.dumps(response))
                except Exception:
                    break

        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosedError as e:
            log.warning("[WS] Connection closed: %s", e)
        except Exception as e:
            log.error("[WS] Client error: %s", e)
        finally:
            log.info("[WS] Client disconnected: %s", addr)
            _log_event(self.shared_logs, "server", "CLIENT_DISCONNECTED",
                       detail=str(addr))

    async def _route(self, command: str, name: str, data: dict,
                     websocket=None) -> dict:

        # ── Register face + voice ─────────────────────────────────────────
        if command == "register":
            if not name:
                return {"status": "error", "message": "Name required"}
            face_r  = await self._dispatch("face",
                        {"op": "enroll", "name": name}, websocket)
            voice_r = await self._dispatch("voice",
                        {"op": "enroll", "name": name}, websocket)
            ok = (face_r.get("status") == "success"
                  and voice_r.get("status") == "success")
            return {"status": "success" if ok else "partial",
                    "user": name, "face": face_r, "voice": voice_r}

        # ── Register face only ────────────────────────────────────────────
        elif command == "register_face":
            if not name:
                return {"status": "error", "message": "Name required"}
            return await self._dispatch("face",
                        {"op": "enroll", "name": name}, websocket)

        elif command == "cancel_enroll_face":
            self.face_cmd_q.put({"op": "cancel_enroll"})
            return {"status": "success", "message": "Cancel signal sent"}

        # ── Register voice only ───────────────────────────────────────────
        elif command == "register_voice":
            if not name:
                return {"status": "error", "message": "Name required"}
            return await self._dispatch("voice",
                        {"op": "enroll", "name": name}, websocket)

        # ── Auth face ─────────────────────────────────────────────────────
        elif command == "auth_face":
            return await self._dispatch("face", {"op": "verify"}, websocket)

        # ── Auth voice ────────────────────────────────────────────────────
        elif command == "auth_voice":
            return await self._dispatch("voice", {"op": "verify"}, websocket)

        # ── Delete ────────────────────────────────────────────────────────
        elif command == "delete":
            if not name:
                return {"status": "error", "message": "Name required"}
            face_r  = await self._dispatch("face",
                        {"op": "delete", "name": name})
            voice_r = await self._dispatch("voice",
                        {"op": "delete", "name": name})
            return {"status": "success", "face": face_r, "voice": voice_r}

        # ── Delete face only ──────────────────────────────────────────────
        elif command == "delete_face":
            if not name:
                return {"status": "error", "message": "Name required"}
            face_r = await self._dispatch("face",
                        {"op": "delete", "name": name})
            return {"status": "success", "face": face_r,
                    "voice": {"status": "skipped"}}

        # ── Delete voice only ─────────────────────────────────────────────
        elif command == "delete_voice":
            if not name:
                return {"status": "error", "message": "Name required"}
            voice_r = await self._dispatch("voice",
                        {"op": "delete", "name": name})
            return {"status": "success",
                    "face": {"status": "skipped"},
                    "voice": voice_r}

        # ── Get thresholds ────────────────────────────────────────────────
        elif command == "get_thresholds":
            face_r  = await self._dispatch("face",  {"op": "get_thresholds"})
            voice_r = await self._dispatch("voice", {"op": "get_thresholds"})
            merged = {"status": "success"}
            if face_r.get("status") == "success":
                for k in ("face_similarity", "face_min_score_gap",
                           "face_enroll_duplicate", "face_min_gallery"):
                    if k in face_r:
                        merged[k] = face_r[k]
            if voice_r.get("status") == "success":
                for k in ("voice_verify", "voice_duplicate"):
                    if k in voice_r:
                        merged[k] = voice_r[k]
            return merged

        # ── Set thresholds ────────────────────────────────────────────────
        elif command == "set_thresholds":
            face_keys  = {k: data[k] for k in
                          ("face_similarity", "face_min_score_gap",
                           "face_enroll_duplicate", "face_min_gallery")
                          if k in data}
            voice_keys = {k: data[k] for k in
                          ("voice_verify", "voice_duplicate")
                          if k in data}
            changed = []
            if face_keys:
                fr = await self._dispatch("face",
                        {"op": "set_thresholds", **face_keys})
                changed.extend(fr.get("changed", []))
            if voice_keys:
                vr = await self._dispatch("voice",
                        {"op": "set_thresholds", **voice_keys})
                changed.extend(vr.get("changed", []))
            return {"status": "success", "changed": changed}

        # ── List identities ───────────────────────────────────────────────
        elif command == "get_identities":
            face_r  = await self._dispatch("face",  {"op": "list"})
            voice_r = await self._dispatch("voice", {"op": "list"})
            fn = face_r.get("data", [])
            vn = voice_r.get("data", [])
            return {"status": "success",
                    "data":       sorted(set(fn) | set(vn)),
                    "face_only":  sorted(set(fn) - set(vn)),
                    "voice_only": sorted(set(vn) - set(fn))}

        # ── Logs (in-memory) ──────────────────────────────────────────────
        elif command == "get_logs":
            return {"status": "success", "data": list(self.shared_logs)}

        # ── Log file (CSV on disk) ────────────────────────────────────────
        elif command == "get_log_file":
            try:
                if os.path.exists(LOG_CSV_FILE):
                    with open(LOG_CSV_FILE, "r") as f:
                        content = f.read()
                    return {"status": "success", "csv": content,
                            "filename": os.path.basename(LOG_CSV_FILE)}
                return {"status": "success", "csv": "",
                        "filename": "biometric_log.csv"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        # ── Status ────────────────────────────────────────────────────────
        elif command == "status":
            return {"status": "success",
                    "face_busy":  self.face_busy.is_set(),
                    "voice_busy": self.voice_busy.is_set()}

        # ── Shutdown ──────────────────────────────────────────────────────
        elif command == "shutdown":
            log.info("[WS] Shutdown requested by client")
            _log_event(self.shared_logs, "server", "SHUTDOWN_REQUESTED")
            self.face_cmd_q.put({"op": "stop"})
            self.voice_cmd_q.put({"op": "stop"})
            return {"status": "success", "message": "Shutting down…"}

        else:
            return {"status": "error",
                    "message": f"Unknown command '{command}'"}


def _try_get(q: Queue):
    try:
        return q.get_nowait()
    except Exception:
        return None


async def run_server(orchestrator, ip, port, stop_evt):
    await orchestrator._init_locks()
    async with serve(
        orchestrator.handle_client, "0.0.0.0", port,
        ping_interval=30, ping_timeout=120,
        close_timeout=10, max_size=10*1024*1024,
    ):
        log.info("[WS] Server live at ws://%s:%d", ip, port)
        try: await stop_evt.wait()
        except asyncio.CancelledError: pass
    log.info("[WS] Server stopped")


async def run_zeroconf(ip, port, stop_evt):
    zc = Zeroconf(ip_version=IPVersion.V4Only)
    info = ServiceInfo(
        "_biometric-auth._tcp.local.",
        "BiometricServer._biometric-auth._tcp.local.",
        addresses=[socket.inet_aton(ip)], port=port,
        properties={"version": "3.0.0", "workers": "face+voice"},
    )
    await zc.async_register_service(info)
    log.info("[mDNS] Advertised on %s:%d", ip, port)
    try: await stop_evt.wait()
    except asyncio.CancelledError: pass
    finally:
        await zc.async_unregister_all_services()
        zc.close()


LAN_CHECK_INTERVAL_S = 60.0

async def run_network_supervisor(orchestrator, port):
    while True:
        ip = await wait_for_lan_ip(LAN_CHECK_INTERVAL_S)
        log.info("[NET] Starting on %s:%d", ip, port)
        stop_evt = asyncio.Event()
        t1 = asyncio.create_task(run_server(orchestrator, ip, port, stop_evt))
        t2 = asyncio.create_task(run_zeroconf(ip, port, stop_evt))
        try:
            while True:
                await asyncio.sleep(LAN_CHECK_INTERVAL_S)
                if get_local_ip() != ip:
                    log.warning("[NET] IP changed — restarting"); break
        finally:
            stop_evt.set()
            await asyncio.gather(t1, t2, return_exceptions=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _pin(CORE_SERVER)
    mp.current_process().name = "orchestrator"
    port = 8765

    _init_csv_log()

    log.info("=" * 70)
    log.info("  Biometric Orchestrator  v3.0")
    log.info("  Cores: server=%s  face=%s  voice=%s", CORE_SERVER, CORE_FACE, CORE_VOICE)
    log.info("  Face: autonomous verify (retry %.0fs on unknown)", AUTO_FACE_RETRY_S)
    log.info("  Voice: autonomous UART listen, STANDBY default")
    log.info("  Log: %s", LOG_CSV_FILE)
    log.info("=" * 70)

    manager      = mp.Manager()
    shared_state = _make_shared_state(manager)
    shared_logs    = shared_state["logs"]
    shared_users   = shared_state["users"]
    shared_display = shared_state["voice_status"]

    _log_event(shared_logs, "server", "STARTUP")

    face_cmd_q,  face_res_q,  face_busy  = Queue(), Queue(), Event()
    voice_cmd_q, voice_res_q, voice_busy = Queue(), Queue(), Event()

    face_proc = Process(
        target=face_worker_process,
        args=(face_cmd_q, face_res_q, face_busy,
              shared_logs, shared_users, shared_display),
        name="face_worker", daemon=True)
    voice_proc = Process(
        target=voice_worker_process,
        args=(voice_cmd_q, voice_res_q, voice_busy,
              shared_logs, shared_users, shared_display),
        name="voice_worker", daemon=True)

    face_proc.start()
    voice_proc.start()
    log.info("[MAIN] Face PID=%d  Voice PID=%d", face_proc.pid, voice_proc.pid)

    # Wait for workers to report ready
    ready = 0
    deadline = time.time() + 30.0
    while ready < 2 and time.time() < deadline:
        for q in (face_res_q, voice_res_q):
            msg = _try_get(q)
            if msg:
                w, s = msg.get("worker", "?"), msg.get("status", "?")
                log.info("[MAIN] Worker '%s': %s", w, s)
                if s in ("ready", "fatal"): ready += 1
        time.sleep(0.1)

    if ready < 2:
        log.warning("[MAIN] Not all workers ready in 30s — continuing anyway")

    orchestrator = OrchestratorServer(
        face_cmd_q, face_res_q, face_busy,
        voice_cmd_q, voice_res_q, voice_busy,
        shared_logs, shared_users)

    loop = asyncio.get_event_loop()

    def _shutdown(sig, frame):
        log.info("[MAIN] Signal %s — shutting down", sig)
        _log_event(shared_logs, "server", "SHUTDOWN")
        face_cmd_q.put({"op": "stop"})
        voice_cmd_q.put({"op": "stop"})
        face_proc.join(timeout=5)
        voice_proc.join(timeout=5)
        for t in asyncio.all_tasks(loop):
            t.cancel()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(run_network_supervisor(orchestrator, port))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        log.info("[MAIN] Goodbye.")
        manager.shutdown()


if __name__ == "__main__":
    main()
