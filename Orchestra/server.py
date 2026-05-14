#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Biometric Orchestrator  –  server.py  (Orchestrator + WebSocket + mDNS)   ║
║                                                                              ║
║   Process layout (one per physical core):                                    ║
║     Core 0  →  server.py  (this file)  WebSocket + ZeroConf                 ║
║     Core 1  →  face_worker   (main.py logic)                                 ║
║     Core 2  →  voice_worker  (rpi4_ecapa_voice_biometric_v2.py logic)        ║
║     Core 3  →  free for OS / camera / UART interrupt handling                ║
║                                                                              ║
║   IPC model:                                                                 ║
║     Server ──cmd_q──▶ Worker                                                 ║
║     Worker ──res_q──▶ Server                                                 ║
║                                                                              ║
║   Command protocol (dict serialised through Queue):                          ║
║     {"op": "verify"}                                                         ║
║     {"op": "enroll",  "name": "Alice"}                                       ║
║     {"op": "delete",  "name": "Alice"}                                       ║
║     {"op": "list"}                                                           ║
║     {"op": "stop"}                                                           ║
║                                                                              ║
║   Workers operate autonomously (continuous identify loop) when idle.         ║
║   A command from the server sets worker_busy Event, pauses autonomous        ║
║   loop, executes the command, clears Event, then resumes.                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import os
import sys
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
from logging.handlers import QueueHandler, QueueListener

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
# Core 3 is intentionally left free for the OS, UART DMA, and camera interrupts.

# ── Autonomous face identification cooldown ───────────────────────────────────
# After a successful or failed auto-identification attempt, the face worker
# will not attempt another one for this many seconds.  Without this guard the
# worker hammers the camera with a new 5-8 s capture cycle the instant the
# previous result flash ends, causing the TFT to show "GRANTED/DENIED" almost
# continuously for anyone standing in view.
#
# Tune this to taste:
#   15 s  → feels like "scan once per visit"
#   30 s  → good for an access-control door (one person at a time)
#   60 s  → suitable when the camera covers a slow-moving queue
AUTO_FACE_COOLDOWN_S = 15.0


def _pin(cores: set):
    """Pin the calling process to the given CPU core set."""
    try:
        os.sched_setaffinity(0, cores)
        log.info("Pinned to core(s) %s", cores)
    except AttributeError:
        log.warning("sched_setaffinity not available on this OS (not Linux?)")
    except PermissionError:
        log.warning("No permission to set CPU affinity — running unpinned.")

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED AUDIT LOG
#  A simple list in a multiprocessing.Manager().list() so all three processes
#  can append to it without a lock.  Server reads it on "get_logs" command.
# ─────────────────────────────────────────────────────────────────────────────

def _make_shared_state(manager):
    return {
        "logs":  manager.list(),           # list of {"time", "worker", "event"}
        "users": manager.dict(),           # {"Alice": "face"|"voice"|"both"}
    }

def _log_event(shared_logs, worker: str, event: str):
    shared_logs.append({
        "time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "worker": worker,
        "event":  event,
    })

# ─────────────────────────────────────────────────────────────────────────────
#  FACE WORKER  (Core 1)
#  Imports the functional pieces of main.py without calling main().
#  Runs an autonomous verification loop; pauses when cmd_q has work.
# ─────────────────────────────────────────────────────────────────────────────

def face_worker_process(cmd_q: Queue, res_q: Queue, busy_evt: Event,
                        shared_logs, shared_users):
    """
    Entry point for the face recognition subprocess.

    DEFAULT (autonomous) mode
    ─────────────────────────
    Runs continuously.  Whenever a face appears in the camera FOV and the DB
    has enough enrolled identities, a verification attempt is triggered
    automatically.  After each attempt a cooldown of AUTO_FACE_COOLDOWN_S
    seconds is enforced so the worker does not spam the camera.

    COMMAND mode (overrides autonomous mode)
    ────────────────────────────────────────
    Any dict placed in cmd_q by the WebSocket server immediately preempts the
    autonomous loop.  The worker executes the command, puts the result in
    res_q, then resumes autonomous mode.  This is how the remote client gains
    exclusive control (enroll / verify / delete / list).
    """
    _pin(CORE_FACE)
    proc_name = "face_worker"
    mp.current_process().name = proc_name
    log.info("[FACE] Worker started on core %s", CORE_FACE)

    # ── lazy import so the heavy libs load in this process only ──────────────
    try:
        import cv2
        import numpy as np
        from picamera2 import Picamera2
        import mediapipe as _mp_lib

        # Import the functions we need directly from face_auth.py
        # (They live in the same directory)
        sys.path.insert(0, os.path.dirname(__file__))
        from face_auth import (
            TFTDisplay, PiCamera, BlazeFaceDetector,
            enroll_face, capture_probe, match_probe,
            load_database, save_database,
            l2_normalize, AuditLog,
            SIMILARITY_THRESHOLD, MIN_GALLERY_SIZE, DATABASE_FILE,
        )
    except ImportError as e:
        log.error("[FACE] Import failed: %s", e)
        res_q.put({"worker": "face", "status": "fatal", "error": str(e)})
        return

    # ── hardware init ─────────────────────────────────────────────────────────
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

    # ── helpers ───────────────────────────────────────────────────────────────
    def _handle_command(cmd: dict) -> dict:
        """Execute one command dict; return result dict."""
        nonlocal db   # needed by the reload_db branch
        op   = cmd.get("op")
        name = cmd.get("name", "").strip()

        if op == "enroll":
            if not name:
                return {"status": "error", "message": "Name required for enroll"}
            embedding = enroll_face(camera, detector, display)
            if embedding is None:
                return {"status": "error", "message": "Enrollment capture failed"}
            db[name] = embedding
            save_database(db)
            shared_users[name] = shared_users.get(name, "") or "face"
            _log_event(shared_logs, "face", f"Enrolled '{name}'")
            return {"status": "success", "user": name}

        elif op == "verify":
            probe = capture_probe(camera, detector, display)
            if probe is None:
                return {"status": "error", "message": "Probe capture failed"}
            granted, best_name, best_score, reason = match_probe(probe, db)
            _log_event(shared_logs, "face",
                       f"{'GRANTED' if granted else 'DENIED'} {best_name} "
                       f"score={best_score:.4f}")
            return {
                "status":  "success",
                "granted": granted,
                "user":    best_name,
                "score":   round(best_score, 4),
                "reason":  reason,
            }

        elif op == "delete":
            if not name:
                return {"status": "error", "message": "Name required for delete"}
            if name not in db:
                return {"status": "error", "message": f"'{name}' not found in face DB"}
            del db[name]
            save_database(db)
            _log_event(shared_logs, "face", f"Deleted '{name}'")
            return {"status": "success", "deleted": name}

        elif op == "list":
            return {"status": "success", "data": list(db.keys())}

        elif op == "reload_db":
            db = load_database()
            return {"status": "success", "count": len(db)}

        else:
            return {"status": "error", "message": f"Unknown op '{op}'"}

    # ── main loop ─────────────────────────────────────────────────────────────
    fps_t          = time.time()
    fps            = 0.0
    fc             = 0
    last_auto_time = 0.0   # timestamp of last autonomous identify attempt

    try:
        while True:
            # ── Priority 1: check for incoming command (non-blocking) ─────────
            if not cmd_q.empty():
                cmd = cmd_q.get_nowait()
                if cmd.get("op") == "stop":
                    log.info("[FACE] Stop command received")
                    break

                busy_evt.set()
                log.info("[FACE] Executing command: %s", cmd)
                try:
                    result = _handle_command(cmd)
                except Exception as exc:
                    result = {"status": "error", "message": traceback.format_exc()}
                res_q.put({"worker": "face", **result})
                busy_evt.clear()
                continue

            # ── Priority 2: autonomous verify loop ───────────────────────────
            ok, frame = camera.read_preview()
            if not ok:
                time.sleep(0.005)
                continue

            fc += 1
            if fc % 30 == 0:
                fps   = 30.0 / (time.time() - fps_t)
                fps_t = time.time()

            detects = detector.detect_with_landmarks(frame)
            faces   = [d["box"] for d in detects]
            display.show_frame(frame, faces=faces, status="WATCHING", fps=fps)

            # Autonomous identification:
            #   • at least one face in frame
            #   • DB has enough enrolled identities
            #   • not currently executing a client command
            #   • cooldown since last attempt has elapsed
            now = time.time()
            if (detects
                    and len(db) >= MIN_GALLERY_SIZE
                    and not busy_evt.is_set()
                    and (now - last_auto_time) >= AUTO_FACE_COOLDOWN_S):

                probe = capture_probe(camera, detector, display)

                # Check cmd_q again — a command may have arrived while we were
                # in the multi-second capture_probe() call.  If so, skip the
                # inference result and let the top-of-loop command path handle
                # it on the next iteration.
                if not cmd_q.empty():
                    continue

                if probe is not None:
                    granted, name, score, reason = match_probe(probe, db)
                    tag        = "GRANTED" if granted else "DENIED"
                    status_msg = f"{tag}:{name}" if granted else "UNKNOWN"
                    log.info("[FACE][AUTO] %s  %s  %.4f", tag, name, score)
                    _log_event(shared_logs, "face",
                               f"[AUTO] {tag} {name} score={score:.4f}")

                    # Flash result on TFT for 2 s
                    t_end = time.time() + 2.0
                    while time.time() < t_end:
                        ok2, fr2 = camera.read_preview()
                        if ok2:
                            display.show_frame(fr2, status=status_msg, fps=fps)

                # Update cooldown timestamp regardless of whether probe
                # succeeded — prevents tight retry loops on dark/blurry frames.
                last_auto_time = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        detector.close()
        save_database(db)
        display.clear()
        log.info("[FACE] Worker shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
#  VOICE WORKER  (Core 2)
#  Imports the functional pieces of rpi4_ecapa_voice_biometric_v2.py.
#  Listens to UART continuously; pauses when commanded.
# ─────────────────────────────────────────────────────────────────────────────

def voice_worker_process(cmd_q: Queue, res_q: Queue, busy_evt: Event,
                         shared_logs, shared_users):
    """
    Entry point for the voice recognition subprocess.

    DEFAULT (autonomous) mode
    ─────────────────────────
    Blocks on UARTReceiver.receive_session().  Each time the ESP32 sends a
    complete audio session, the worker runs ECAPA-TDNN inference and logs
    the identification result.  It re-checks cmd_q after every session so
    commands are never delayed by more than VERIFY_MAX_S + RECEIVE_TIMEOUT_S
    seconds (~11 s worst case).

    COMMAND mode
    ────────────
    Same override mechanism as the face worker.
    """
    _pin(CORE_VOICE)
    proc_name = "voice_worker"
    mp.current_process().name = proc_name
    log.info("[VOICE] Worker started on core %s", CORE_VOICE)

    try:
        import numpy as np
        import serial
        sys.path.insert(0, os.path.dirname(__file__))
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

    # ── hardware init ─────────────────────────────────────────────────────────
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

    # ── helpers ───────────────────────────────────────────────────────────────
    def _embed_audio(raw: bytes):
        features, _ = raw_to_features(raw)
        return auth.embed(features)

    def _best_match(live_emb):
        speakers = enrolled_speakers()
        if not speakers:
            return None, -1.0
        best_name, best_score = None, -1.0
        for p in speakers:
            stored = np.load(p)
            s = auth.cosine(stored, live_emb)
            n = os.path.basename(p)[:-4]
            if s > best_score:
                best_score, best_name = s, n
        return best_name, best_score

    def _handle_command(cmd: dict) -> dict:
        op   = cmd.get("op")
        name = (cmd.get("name") or "").lower().strip()

        if op == "enroll":
            if not name:
                return {"status": "error", "message": "Name required"}
            emb_path = os.path.join(VOICEPRINT_DB, f"{name}.npy")
            if os.path.exists(emb_path):
                return {"status": "error",
                        "message": f"'{name}' already enrolled. Delete first."}
            raw = uart.receive_session(ENROLL_MIN_S, ENROLL_MAX_S,
                                       label=f"enrollment '{name}'")
            if raw is None:
                return {"status": "error", "message": "No audio received"}
            try:
                emb = _embed_audio(raw)
            except ValueError as e:
                return {"status": "error", "message": f"Quality check: {e}"}
            # Duplicate check
            best_n, best_s = _best_match(emb)
            if best_s >= DUPLICATE_THRESHOLD:
                return {"status": "error",
                        "message": f"Duplicate rejected — matches '{best_n}' "
                                   f"at {best_s:.4f}"}
            np.save(emb_path, emb)
            shared_users[name] = shared_users.get(name, "") or "voice"
            _log_event(shared_logs, "voice", f"Enrolled '{name}'")
            return {"status": "success", "user": name}

        elif op == "verify":
            speakers = enrolled_speakers()
            if not speakers:
                return {"status": "error", "message": "No speakers enrolled"}
            raw = uart.receive_session(VERIFY_MIN_S, VERIFY_MAX_S,
                                       label="verification")
            if raw is None:
                return {"status": "error", "message": "No audio received"}
            try:
                emb = _embed_audio(raw)
            except ValueError as e:
                return {"status": "error", "message": f"Quality check: {e}"}
            best_name, best_score = _best_match(emb)
            granted = best_score >= VERIFY_THRESHOLD
            _log_event(shared_logs, "voice",
                       f"{'GRANTED' if granted else 'DENIED'} {best_name} "
                       f"score={best_score:.4f}")
            return {
                "status":  "success",
                "granted": granted,
                "user":    best_name,
                "score":   round(best_score, 4),
            }

        elif op == "delete":
            if not name:
                return {"status": "error", "message": "Name required"}
            path = os.path.join(VOICEPRINT_DB, f"{name}.npy")
            if not os.path.exists(path):
                return {"status": "error", "message": f"'{name}' not found"}
            os.remove(path)
            _log_event(shared_logs, "voice", f"Deleted '{name}'")
            return {"status": "success", "deleted": name}

        elif op == "list":
            names = [os.path.basename(p)[:-4] for p in enrolled_speakers()]
            return {"status": "success", "data": names}

        else:
            return {"status": "error", "message": f"Unknown op '{op}'"}

    # ── main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            # ── Check for command first (non-blocking) ────────────────────────
            if not cmd_q.empty():
                cmd = cmd_q.get_nowait()
                if cmd.get("op") == "stop":
                    log.info("[VOICE] Stop command received")
                    break

                busy_evt.set()
                log.info("[VOICE] Executing command: %s", cmd)
                try:
                    result = _handle_command(cmd)
                except Exception as exc:
                    result = {"status": "error", "message": traceback.format_exc()}
                res_q.put({"worker": "voice", **result})
                busy_evt.clear()
                continue

            # ── Autonomous: attempt to receive an audio session ───────────────
            # UARTReceiver.receive_session() blocks for up to RECEIVE_TIMEOUT_S
            # waiting for a frame.  We use the short verify window so the loop
            # re-checks cmd_q every ~VERIFY_MAX_S seconds.
            log.debug("[VOICE][AUTO] Listening on UART…")
            raw = uart.receive_session(
                min_seconds=VERIFY_MIN_S,
                max_seconds=VERIFY_MAX_S,
                label="auto-identify")

            # If a command arrived while we were blocking, skip inference
            if not cmd_q.empty():
                continue

            if raw is None:
                continue   # timeout / short session — loop again

            speakers = enrolled_speakers()
            if not speakers:
                log.debug("[VOICE][AUTO] No speakers enrolled — skipping")
                continue

            try:
                emb = _embed_audio(raw)
            except ValueError as e:
                log.warning("[VOICE][AUTO] Quality fail: %s", e)
                continue

            best_name, best_score = _best_match(emb)
            granted = best_score >= VERIFY_THRESHOLD
            tag = "GRANTED" if granted else "DENIED"
            log.info("[VOICE][AUTO] %s  %s  %.4f", tag, best_name, best_score)
            _log_event(shared_logs, "voice",
                       f"[AUTO] {tag} {best_name} score={best_score:.4f}")

    except KeyboardInterrupt:
        pass
    finally:
        uart.close()
        log.info("[VOICE] Worker shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
#  WEBSOCKET SERVER  (Core 0 — asyncio event loop)
# ─────────────────────────────────────────────────────────────────────────────

def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class OrchestratorServer:
    """
    Owns the WebSocket server and the two worker subprocess handles.
    Dispatches client commands to the appropriate worker queue and relays
    the response back to the client.
    """

    RESPONSE_TIMEOUT = 60.0   # seconds to wait for a worker result

    def __init__(self,
                 face_cmd_q, face_res_q, face_busy,
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

        # Asyncio lock per worker — prevents two clients from issuing
        # overlapping commands to the same worker simultaneously.
        self._face_lock  = None   # created in async context
        self._voice_lock = None

    async def _init_locks(self):
        self._face_lock  = asyncio.Lock()
        self._voice_lock = asyncio.Lock()

    async def _dispatch(self, worker: str, cmd: dict) -> dict:
        """
        Send cmd to worker queue, wait (async-friendly) for response.
        Uses a thread-executor poll so the asyncio event loop stays live.
        """
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
                # Poll result queue in thread pool to avoid blocking event loop
                result = await loop.run_in_executor(None, _try_get, q_res)
                if result is not None:
                    return result
                await asyncio.sleep(0.05)

    async def handle_client(self, websocket):
        addr = websocket.remote_address
        log.info("[WS] Client connected: %s", addr)
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
                log.info("[WS] Command '%s' from %s", command, addr)

                response = await self._route(command, name, data)
                await websocket.send(json.dumps(response))

        except websockets.exceptions.ConnectionClosedOK:
            pass
        except Exception as e:
            log.error("[WS] Client %s error: %s", addr, e)
        finally:
            log.info("[WS] Client disconnected: %s", addr)

    async def _route(self, command: str, name: str, data: dict) -> dict:
        """Map WebSocket command strings to worker IPC ops."""

        # ── Registration (face + voice simultaneously) ────────────────────────
        # Used when you want to enrol a brand-new identity in both modalities
        # back-to-back in a single call.  The face capture happens first, then
        # the voice session.
        if command == "register":
            if not name:
                return {"status": "error", "message": "Name required"}
            log.info("[ROUTE] Enrolling '%s' face + voice", name)
            face_result  = await self._dispatch("face",  {"op": "enroll", "name": name})
            voice_result = await self._dispatch("voice", {"op": "enroll", "name": name})
            combined_ok  = (face_result.get("status")  == "success" and
                            voice_result.get("status") == "success")
            return {
                "status": "success" if combined_ok else "partial",
                "user":   name,
                "face":   face_result,
                "voice":  voice_result,
            }

        # ── Register face only ────────────────────────────────────────────────
        elif command == "register_face":
            if not name:
                return {"status": "error", "message": "Name required"}
            return await self._dispatch("face", {"op": "enroll", "name": name})

        # ── Register voice only ───────────────────────────────────────────────
        elif command == "register_voice":
            if not name:
                return {"status": "error", "message": "Name required"}
            return await self._dispatch("voice", {"op": "enroll", "name": name})

        # ── Auth: voice (UART-triggered) ──────────────────────────────────────
        elif command == "auth_voice":
            return await self._dispatch("voice", {"op": "verify"})

        # ── Auth: face (camera-triggered) ─────────────────────────────────────
        elif command == "auth_face":
            return await self._dispatch("face", {"op": "verify"})

        # ── Delete identity ───────────────────────────────────────────────────
        elif command == "delete":
            if not name:
                return {"status": "error", "message": "Name required"}
            face_r  = await self._dispatch("face",  {"op": "delete", "name": name})
            voice_r = await self._dispatch("voice", {"op": "delete", "name": name})
            return {"status": "success", "face": face_r, "voice": voice_r}

        # ── List identities ───────────────────────────────────────────────────
        elif command == "get_identities":
            face_r  = await self._dispatch("face",  {"op": "list"})
            voice_r = await self._dispatch("voice", {"op": "list"})
            face_names  = face_r.get("data", [])
            voice_names = voice_r.get("data", [])
            all_names   = sorted(set(face_names) | set(voice_names))
            return {"status": "success", "data": all_names,
                    "face_only":  sorted(set(face_names)  - set(voice_names)),
                    "voice_only": sorted(set(voice_names) - set(face_names))}

        # ── Audit logs ────────────────────────────────────────────────────────
        elif command == "get_logs":
            entries = list(self.shared_logs)   # snapshot
            return {"status": "success", "data": entries}

        # ── Worker status / heartbeat ─────────────────────────────────────────
        elif command == "status":
            return {
                "status": "success",
                "face_busy":  self.face_busy.is_set(),
                "voice_busy": self.voice_busy.is_set(),
            }

        else:
            return {"status": "error", "message": f"Unknown command '{command}'"}


def _try_get(q: Queue):
    """Non-blocking queue get; returns None if empty. Safe to call from executor."""
    try:
        return q.get_nowait()
    except Exception:
        return None


async def run_server(orchestrator: OrchestratorServer, ip: str, port: int):
    await orchestrator._init_locks()
    async with serve(orchestrator.handle_client, "0.0.0.0", port):
        log.info("[WS] WebSocket server live at ws://%s:%d", ip, port)
        try:
            await asyncio.Future()   # run forever
        except asyncio.CancelledError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  ZEROCONF mDNS advertisement
# ─────────────────────────────────────────────────────────────────────────────

async def run_zeroconf(ip: str, port: int):
    zc = Zeroconf(ip_version=IPVersion.V4Only)
    info = ServiceInfo(
        "_biometric-auth._tcp.local.",
        "BiometricServer._biometric-auth._tcp.local.",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={"version": "2.0.0", "workers": "face+voice"},
    )
    await zc.async_register_service(info)
    log.info("[mDNS] Advertised service on %s:%d", ip, port)
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        await zc.async_unregister_all_services()
        zc.close()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _pin(CORE_SERVER)
    mp.current_process().name = "orchestrator"

    ip   = get_local_ip()
    port = 8765

    log.info("=" * 70)
    log.info("  Biometric Orchestrator  v2.1")
    log.info("  Server IP  : %s:%d", ip, port)
    log.info("  Core map   : server=%s  face=%s  voice=%s", CORE_SERVER, CORE_FACE, CORE_VOICE)
    log.info("  Face auto-ID cooldown: %.0f s", AUTO_FACE_COOLDOWN_S)
    log.info("=" * 70)

    # ── Shared state ──────────────────────────────────────────────────────────
    manager      = mp.Manager()
    shared_state = _make_shared_state(manager)
    shared_logs  = shared_state["logs"]
    shared_users = shared_state["users"]

    _log_event(shared_logs, "server", "System startup")

    # ── IPC queues + events ───────────────────────────────────────────────────
    face_cmd_q  = Queue()
    face_res_q  = Queue()
    face_busy   = Event()

    voice_cmd_q = Queue()
    voice_res_q = Queue()
    voice_busy  = Event()

    # ── Launch workers ────────────────────────────────────────────────────────
    face_proc = Process(
        target=face_worker_process,
        args=(face_cmd_q, face_res_q, face_busy, shared_logs, shared_users),
        name="face_worker",
        daemon=True,
    )
    voice_proc = Process(
        target=voice_worker_process,
        args=(voice_cmd_q, voice_res_q, voice_busy, shared_logs, shared_users),
        name="voice_worker",
        daemon=True,
    )

    face_proc.start()
    voice_proc.start()
    log.info("[MAIN] Face  worker PID: %d", face_proc.pid)
    log.info("[MAIN] Voice worker PID: %d", voice_proc.pid)

    # Wait for both workers to report ready (or fatal) before opening WS port
    ready_count = 0
    deadline    = time.time() + 30.0
    while ready_count < 2 and time.time() < deadline:
        for q in (face_res_q, voice_res_q):
            msg = _try_get(q)
            if msg:
                w = msg.get("worker", "?")
                s = msg.get("status", "?")
                log.info("[MAIN] Worker '%s' reported: %s", w, s)
                if s in ("ready", "fatal"):
                    ready_count += 1
        time.sleep(0.1)

    if ready_count < 2:
        log.warning("[MAIN] Not all workers reported ready within 30s — continuing anyway")

    # ── Build orchestrator ────────────────────────────────────────────────────
    orchestrator = OrchestratorServer(
        face_cmd_q,  face_res_q,  face_busy,
        voice_cmd_q, voice_res_q, voice_busy,
        shared_logs, shared_users,
    )

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    loop = asyncio.get_event_loop()

    def _shutdown(sig, frame):
        log.info("[MAIN] Signal %s received — shutting down", sig)
        face_cmd_q.put({"op": "stop"})
        voice_cmd_q.put({"op": "stop"})
        face_proc.join(timeout=5)
        voice_proc.join(timeout=5)
        for t in asyncio.all_tasks(loop):
            t.cancel()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Run asyncio tasks (WS server + mDNS) ─────────────────────────────────
    async def _run_all():
        await asyncio.gather(
            run_server(orchestrator, ip, port),
            run_zeroconf(ip, port),
        )

    try:
        loop.run_until_complete(_run_all())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        log.info("[MAIN] Event loop closed. Goodbye.")
        manager.shutdown()


if __name__ == "__main__":
    main()
