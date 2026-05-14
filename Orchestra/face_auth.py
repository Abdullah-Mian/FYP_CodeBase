#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   face_auth.py  —  Continuous Face Authentication Engine                    ║
║   Raspberry Pi 4  ·  MobileFaceNet (Rust)  ·  MediaPipe BlazeFace          ║
║                                                                              ║
║   This file merges main.py + rust_wrapper.py into one self-contained        ║
║   module.  It can be imported by server.py OR run as a standalone script.   ║
║                                                                              ║
║   Key changes vs the original main.py                                        ║
║   ─────────────────────────────────────                                      ║
║   1.  Continuous authentication — faces in FOV are automatically identified  ║
║       without any keyboard command.                                          ║
║   2.  Per-face bounding-box labels rendered on the TFT:                      ║
║         GREEN   "Alice | 0.87"   →  GRANTED                                 ║
║         RED     "UNKNOWN"        →  DENIED / below threshold                 ║
║         YELLOW  "scanning…"      →  capture in progress                      ║
║   3.  Smart retry logic via lightweight face tracking:                       ║
║         • Unknown faces are re-attempted every RETRY_UNKNOWN_S seconds       ║
║           while still in FOV.                                                ║
║         • Granted faces skip re-authentication for GRANTED_REAUTH_S seconds  ║
║           (avoids hammering for a person standing at the door).              ║
║   4.  UUID-based JSONL audit log — one JSON record per attempt:              ║
║         {"uuid":"…","timestamp":"…","name":"…",                             ║
║          "confidence":0.87,"mode":"face","granted":true}                    ║
║       Consumed by server.py via get_logs() and served to the WebSocket      ║
║       client on demand.                                                      ║
║   5.  rust_wrapper.py is inlined here — zero internal project imports.      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import math
import os
import pickle
import select
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import board
import cv2
import digitalio
import numpy as np
from PIL import Image, ImageFont
from adafruit_rgb_display import ili9341

try:
    from picamera2 import Picamera2
except ImportError:
    print("❌  picamera2 not found — install via: sudo apt install python3-picamera2")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("❌  mediapipe not found — install via: sudo pip install mediapipe --break-system-packages")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — tune these values for your deployment
# ══════════════════════════════════════════════════════════════════════════════

# ── Rust binary ───────────────────────────────────────────────────────────────
# Path to the compiled MobileFaceNet Rust binary (relative to CWD when the
# script is executed, or absolute).  The binary accepts one argument (image
# path) and prints a JSON array of 128 floats to stdout.
RUST_BINARY_PATH = "./rust_face_engine/rust_face_engine"

# ── Security thresholds ───────────────────────────────────────────────────────
# Cosine similarity range is [-1, 1].  A genuine match at close range with
# good lighting typically scores 0.80–0.92.
# • Raise SIMILARITY_THRESHOLD if strangers are incorrectly granted access.
# • Lower it only if your own face is being rejected.
SIMILARITY_THRESHOLD = 0.75

# Two-identity open-set guard: if the gap between the best and second-best
# cosine score is smaller than this, the match is declared ambiguous and
# access is denied.  Prevents false grants when only one person is enrolled.
MIN_SCORE_GAP = 0.10

# Minimum gallery size before any authentication is attempted.  With just one
# enrolled identity there is no "competition" and an impostor can score
# falsely high.  Require at least 2 before the system is live.
MIN_GALLERY_SIZE = 2

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_FILE = "faces_db.pkl"

# ── Audit log ─────────────────────────────────────────────────────────────────
# JSONL file — one JSON object per line, appended on every auth attempt.
# server.py reads this file on "get_logs" WebSocket command.
AUDIT_LOG_FILE = "face_audit.jsonl"

# ── Continuous authentication timing ─────────────────────────────────────────
# How long (seconds) to wait before re-attempting a face that was NOT matched.
# 3 s gives a natural "scan-again" cadence without hammering the camera.
RETRY_UNKNOWN_S: float = 3.0

# How long (seconds) to wait before re-authenticating a face that WAS granted.
# This prevents re-scanning a person who is just standing at the door.
GRANTED_REAUTH_S: float = 30.0

# ── Face tracking ─────────────────────────────────────────────────────────────
# Minimum IoU (Intersection-over-Union) score for two bounding boxes to be
# considered the SAME face across consecutive frames.
# 0.35 is intentionally loose to handle slight head movement between frames.
TRACK_IOU_THRESHOLD: float = 0.35

# Seconds of absence before a track is dropped.  When a face leaves FOV and
# returns after this long, it gets a fresh authentication attempt.
TRACK_MAX_AGE_S: float = 2.0

# ── Camera capture ────────────────────────────────────────────────────────────
PREVIEW_WIDTH  = 320
PREVIEW_HEIGHT = 240
CAPTURE_WIDTH  = 3280
CAPTURE_HEIGHT = 2464
MIN_FACE_CONF  = 0.5      # MediaPipe minimum detection confidence

# ── Face quality guards ───────────────────────────────────────────────────────
# Crop size (px) in the HI-RES frame below which we reject and retry.
MIN_FACE_CROP_PX = 200
# Face width (px) in the 320×240 PREVIEW frame below which we show "MOVE CLOSER".
MIN_FACE_PREVIEW_PX = 60

# ── Face alignment ────────────────────────────────────────────────────────────
# Target size of the square aligned-face crop sent to Rust / MobileFaceNet.
# 112 × 112 matches the ArcFace / MobileFaceNet training protocol exactly.
ALIGN_OUTPUT_SIZE    = 112
ALIGN_EYE_Y_RATIO    = 0.35   # eyes placed at 35 % from top of output image
ALIGN_EYE_DIST_RATIO = 0.37   # inter-eye distance as fraction of output width

# ── Enrollment countdown ──────────────────────────────────────────────────────
COUNTDOWN_SECONDS = 3

# ── Display ───────────────────────────────────────────────────────────────────
DISPLAY_WIDTH  = 320
DISPLAY_HEIGHT = 240
BAUDRATE       = 64_000_000

# ── Result flash duration (used during commanded verify, not continuous mode) ─
RESULT_DISPLAY_SECONDS = 2.0


# ══════════════════════════════════════════════════════════════════════════════
#  RUST WRAPPER  (inlined from rust_wrapper.py)
#  All Rust calls go through these two functions.
# ══════════════════════════════════════════════════════════════════════════════

def get_face_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Invoke the compiled Rust binary on *image_path* and return a 128-D
    L2-normalised float32 numpy embedding, or None on any failure.

    How it works
    ────────────
    The Rust binary:
      1. Loads the JPEG / PNG at image_path.
      2. Runs a MobileFaceNet forward pass (compiled as a Rust crate).
      3. L2-normalises the 128-D output vector.
      4. Prints it to stdout as a JSON array: [0.123, -0.456, …]

    We capture stdout, parse the JSON, then defensively re-normalise in case
    of any floating-point drift during the subprocess round-trip.

    Returns None (without raising) on:
      • Binary not found
      • Non-zero exit code from Rust (binary prints its own error to stderr)
      • Rust output is not valid JSON
      • Embedding has zero norm (degenerate model output)
    """
    if not os.path.exists(RUST_BINARY_PATH):
        raise FileNotFoundError(
            f"Rust binary not found: {RUST_BINARY_PATH}\n"
            f"Run 'cargo build --release' inside rust_face_engine/ first.")

    try:
        t0     = time.time()
        result = subprocess.run(
            [RUST_BINARY_PATH, image_path],
            capture_output=True, text=True,
            check=True,   # raises CalledProcessError on non-zero exit
        )
        elapsed = time.time() - t0

        raw = result.stdout.strip()
        if not raw:
            print("❌  Rust: no output on stdout")
            return None

        # Rust's Vec<f32> Debug format   [0.1, -0.2, …]  is valid JSON.
        vec = json.loads(raw)
        emb = np.array(vec, dtype=np.float32)

        # Defensive re-normalisation
        norm = float(np.linalg.norm(emb))
        if norm < 1e-10:
            print("❌  Rust returned a zero-norm embedding")
            return None
        emb = emb / norm

        print(f"⚡  Rust inference {elapsed:.3f}s  dim={emb.shape[0]}")
        return emb

    except subprocess.CalledProcessError as exc:
        print(f"❌  Rust process error:\n{exc.stderr.strip()}")
        return None
    except json.JSONDecodeError as exc:
        print(f"❌  Rust output not valid JSON: {exc}")
        return None
    except Exception as exc:
        print(f"❌  get_face_embedding: {exc}")
        return None


def compute_similarity(embed1, embed2) -> float:
    """
    Cosine similarity between two face embeddings.  Returns a float in
    [-1.0, 1.0] where 1.0 means identical.

    Both vectors are expected to be L2-normalised (which this module always
    ensures), but we normalise defensively so external callers are also safe.
    """
    a = np.asarray(embed1, dtype=np.float32).ravel()
    b = np.asarray(embed2, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return -1.0
    return float(np.dot(a / na, b / nb))


# ══════════════════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return the L2-normalised form of vector v (float32, 1-D)."""
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-10 else v


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIT LOG
#  UUID-tagged JSONL file — one record per auth attempt (face or voice).
#  server.py calls AuditLog.get_all() to serve logs to the WebSocket client.
# ══════════════════════════════════════════════════════════════════════════════

class AuditLog:
    """
    Append-only JSONL audit log for authentication events.

    Each record has the following fields:
      uuid        — RFC-4122 v4 UUID, unique per attempt
      timestamp   — ISO-8601 local time string
      name        — matched identity name (or "UNKNOWN")
      confidence  — cosine similarity score (0.0 for unknown)
      mode        — "face" or "voice"
      granted     — true / false

    Format choice: JSONL (newline-delimited JSON) is the simplest possible
    format that is both human-readable and machine-parseable without loading
    the entire file into memory.  New records are appended with a single
    write — no locking required since only one process writes here.
    """

    def __init__(self, path: str = AUDIT_LOG_FILE):
        self._path = path

    def append(self, name: str, confidence: float,
               mode: str, granted: bool) -> str:
        """
        Write one record.  Returns the UUID string so the caller can
        reference it in its own log or print it.
        """
        record = {
            "uuid":       str(uuid.uuid4()),
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "name":       name,
            "confidence": round(float(confidence), 4),
            "mode":       mode,
            "granted":    granted,
        }
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record["uuid"]

    def get_all(self) -> List[dict]:
        """
        Return all log entries as a list of dicts (chronological order).
        Safe to call while the file is being appended to.
        """
        if not os.path.exists(self._path):
            return []
        entries = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass   # skip corrupt lines
        return entries


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def load_database() -> dict:
    """
    Load the face gallery from DATABASE_FILE.

    Backward compatible: handles three historical storage formats:
      • Raw np.ndarray  (very old format)
      • {"template": np.ndarray, …}  (intermediate multi-sample format)
      • Any array-like  (current format — just a normalised 128-D vector)

    Always returns a dict of {name: L2-normalised np.ndarray}.
    """
    if not os.path.exists(DATABASE_FILE):
        return {}
    try:
        with open(DATABASE_FILE, "rb") as fh:
            raw = pickle.load(fh)
        if not isinstance(raw, dict):
            print("⚠️  DB not a dict — starting fresh.")
            return {}
        db = {}
        for name, val in raw.items():
            try:
                if isinstance(val, np.ndarray):
                    db[name] = l2_normalize(val)
                elif isinstance(val, dict) and "template" in val:
                    db[name] = l2_normalize(
                        np.asarray(val["template"], dtype=np.float32))
                else:
                    db[name] = l2_normalize(
                        np.asarray(val, dtype=np.float32))
            except Exception:
                print(f"⚠️  Skipping corrupt DB entry '{name}'")
        print(f"✅  DB loaded: {len(db)} identities")
        return db
    except Exception as exc:
        print(f"⚠️  DB load failed ({exc}) — starting fresh.")
        return {}


def save_database(db: dict) -> None:
    """Persist the gallery dict to DATABASE_FILE (atomic write via temp file)."""
    tmp = DATABASE_FILE + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(db, fh)
    os.replace(tmp, DATABASE_FILE)   # atomic on POSIX


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY  (ILI9341 / SPI TFT)
# ══════════════════════════════════════════════════════════════════════════════

# Colour palette used for bounding-box labels and status text.
# NOTE: This display pipeline does RGB→BGR conversion then sends the buffer
# with mode="RGB", which swaps R and B on screen.  Green is symmetric so it
# is unaffected.  The colour assignments below are intentionally pre-swapped
# so THEY LOOK CORRECT on the physical panel.
_COL_GRANTED  = (0,   255, 0)    # GREEN  — access granted
_COL_DENIED   = (255, 0,   0)    # RED    — access denied / unknown
_COL_SCANNING = (0,   255, 255)  # YELLOW (pre-swapped) — capture in progress
_COL_INFO     = (255, 255, 255)  # WHITE  — neutral status
_COL_CLOSER   = (255, 0,   0)    # RED    — "MOVE CLOSER"
_COL_HOLD     = (255, 255, 0)    # CYAN (pre-swapped) — "HOLD STILL"


class TFTDisplay:
    """
    Thin wrapper around the ILI9341 SPI TFT panel.

    show_frame() accepts an optional list of face_labels:
        [{"box": (x, y, w, h), "label": "Alice | 0.87", "color": (r, g, b)}, …]

    Each face gets its own coloured rectangle and label string, enabling the
    system to show different identities simultaneously in multi-face scenes.
    """

    def __init__(self):
        self.disp = ili9341.ILI9341(
            board.SPI(),
            cs=digitalio.DigitalInOut(board.CE0),
            dc=digitalio.DigitalInOut(board.D25),
            rst=digitalio.DigitalInOut(board.D27),
            baudrate=BAUDRATE, width=240, height=320, rotation=90,
        )
        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except Exception:
            self.font = ImageFont.load_default()
        print("✅  TFT Display initialised")

    def show_frame(self, frame: np.ndarray,
                   face_labels: Optional[List[dict]] = None,
                   # Legacy keyword kept for backward compatibility with server.py
                   faces: Optional[List[tuple]] = None,
                   status: str = "",
                   fps: float = 0.0) -> None:
        """
        Render frame to TFT.

        face_labels  — preferred: list of {"box", "label", "color"}
        faces        — legacy: plain list of (x,y,w,h) boxes drawn in green
        status       — optional top-bar string
        fps          — current frame rate (shown in top bar)
        """
        # Mirror horizontally so the live preview looks like a mirror
        # (natural for a person looking at themselves in the panel).
        img = cv2.flip(frame, 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame_w = frame.shape[1]

        # ── Per-face labelled boxes (new path) ────────────────────────────────
        if face_labels:
            for fl in face_labels:
                x, y, w, h = fl["box"]
                # Flip x coordinate to match the mirrored frame
                flipped_x = frame_w - x - w
                col   = fl.get("color", _COL_INFO)
                label = fl.get("label", "")

                cv2.rectangle(img, (flipped_x, y),
                              (flipped_x + w, y + h), col, 2)

                if label:
                    # Draw a small filled rectangle behind the label text
                    # so it is readable regardless of background colour.
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                    lx, ly = flipped_x, max(0, y - 2)
                    cv2.rectangle(img, (lx, ly - th - 2),
                                  (lx + tw + 2, ly), col, cv2.FILLED)
                    cv2.putText(img, label, (lx + 1, ly - 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 0, 0),   # black text on coloured bg
                                1, cv2.LINE_AA)

        # ── Legacy plain boxes (backward-compat with server.py calls) ─────────
        elif faces:
            for (x, y, w, h) in faces:
                flipped_x = frame_w - x - w
                cv2.rectangle(img, (flipped_x, y),
                              (flipped_x + w, y + h), _COL_GRANTED, 2)

        # ── Top-bar status text ───────────────────────────────────────────────
        bar = f"FPS:{fps:.0f}" + (f" | {status}" if status else "")
        if "CLOSER"  in status: col = _COL_CLOSER
        elif "GRANTED" in status: col = _COL_GRANTED
        elif "DENIED"  in status: col = _COL_DENIED
        elif "SCANNING" in status: col = _COL_SCANNING
        elif "HOLD"    in status: col = _COL_HOLD
        else: col = _COL_INFO
        cv2.putText(img, bar, (2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, col, 1, cv2.LINE_AA)

        self.disp.image(Image.fromarray(img, mode="RGB"))

    def clear(self) -> None:
        """Blank the display (black screen)."""
        self.disp.image(Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0))


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA  (Picamera2)
# ══════════════════════════════════════════════════════════════════════════════

class PiCamera:
    """
    Manages a Picamera2 instance with two configurations:
      • preview  — 320 × 240 RGB888, fast readout, used for detection loop
      • hires    — 3280 × 2464 RGB888, used for embedding capture
    Switching between them requires a stop/configure/start cycle (~1.3 s).
    """

    def __init__(self):
        self._cam = self._preview_cfg = self._capture_cfg = None

    def start(self) -> bool:
        try:
            self._cam = Picamera2()
            self._preview_cfg = self._cam.create_preview_configuration(
                main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"},
                raw={"size": (1640, 1232)},
            )
            self._capture_cfg = self._cam.create_still_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"})
            self._cam.configure(self._preview_cfg)
            self._cam.start()
            time.sleep(2)   # allow AE/AWB to converge
            print(f"✅  Camera ready  preview={PREVIEW_WIDTH}×{PREVIEW_HEIGHT}  "
                  f"hi-res={CAPTURE_WIDTH}×{CAPTURE_HEIGHT}")
            return True
        except Exception as exc:
            print(f"❌  Camera init: {exc}")
            return False

    def read_preview(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            return True, self._cam.capture_array()
        except Exception:
            return False, None

    def _to_hires(self) -> None:
        """Switch sensor to hi-res still config (~0.8 s + 0.8 s warmup)."""
        self._cam.stop()
        self._cam.configure(self._capture_cfg)
        self._cam.start()
        time.sleep(0.8)

    def _to_preview(self) -> None:
        """Switch back to preview config (~0.5 s warmup)."""
        self._cam.stop()
        self._cam.configure(self._preview_cfg)
        self._cam.start()
        time.sleep(0.5)

    def capture_hires_frame(self) -> np.ndarray:
        return self._cam.capture_array()

    def stop(self) -> None:
        if self._cam:
            self._cam.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  FACE DETECTOR  (MediaPipe BlazeFace)
# ══════════════════════════════════════════════════════════════════════════════

class BlazeFaceDetector:
    """
    Wraps MediaPipe FaceDetection (BlazeFace short-range model).

    detect_with_landmarks() returns a list of detection dicts:
        {
          "box":       (x, y, w, h),      ← pixel coords in input frame
          "right_eye": (px, py),           ← keypoint 0  (person's right eye)
          "left_eye":  (px, py),           ← keypoint 1  (person's left eye)
          "confidence": float,             ← MediaPipe detection score
        }

    We need both eye keypoints for geometric face alignment — they anchor the
    affine warp that places eyes at canonical positions in the 112×112 crop,
    which is critical for MobileFaceNet accuracy.
    """

    def __init__(self):
        _mp = mp.solutions.face_detection
        self._det = _mp.FaceDetection(
            model_selection=0,            # 0 = short-range (≤2 m), better for door use
            min_detection_confidence=MIN_FACE_CONF)

    def detect_with_landmarks(self, frame: np.ndarray) -> List[dict]:
        res = self._det.process(frame)
        if not res.detections:
            return []
        h, w = frame.shape[:2]
        out  = []
        for d in res.detections:
            b  = d.location_data.relative_bounding_box
            kp = d.location_data.relative_keypoints
            out.append({
                "box": (
                    max(0, int(b.xmin   * w)),
                    max(0, int(b.ymin   * h)),
                    max(1, int(b.width  * w)),
                    max(1, int(b.height * h)),
                ),
                "right_eye":  (kp[0].x * w, kp[0].y * h),
                "left_eye":   (kp[1].x * w, kp[1].y * h),
                "confidence": float(d.score[0]) if d.score else 0.0,
            })
        return out

    def close(self):
        self._det.close()


# ══════════════════════════════════════════════════════════════════════════════
#  FACE ALIGNMENT  (ArcFace / MobileFaceNet protocol)
# ══════════════════════════════════════════════════════════════════════════════

def align_face(frame: np.ndarray,
               right_eye_xy: tuple,
               left_eye_xy: tuple,
               output_size: int = ALIGN_OUTPUT_SIZE) -> np.ndarray:
    """
    Affine-warp frame so that the two eye keypoints land at canonical
    positions in a square output_size × output_size crop.

    This replicates the exact preprocessing applied to every face in the
    ArcFace / MobileFaceNet training set.  Skipping alignment degrades
    recognition accuracy significantly (especially for tilted heads).

    Steps
    ─────
    1. Compute the roll angle between the two eyes.
    2. Rotate around the eye midpoint to eliminate roll.
    3. Scale so the inter-eye distance equals ALIGN_EYE_DIST_RATIO × output_size.
    4. Translate so the eye midpoint lands at the canonical y-position
       (ALIGN_EYE_Y_RATIO from top) and is horizontally centred.
    """
    rx, ry = right_eye_xy
    lx, ly = left_eye_xy

    angle    = math.degrees(math.atan2(ly - ry, lx - rx))
    eye_cx   = (rx + lx) / 2.0
    eye_cy   = (ry + ly) / 2.0
    eye_dist = math.hypot(lx - rx, ly - ry) + 1e-6

    desired_dist = output_size * ALIGN_EYE_DIST_RATIO
    scale        = desired_dist / eye_dist

    M = cv2.getRotationMatrix2D((eye_cx, eye_cy), angle, scale)
    M[0, 2] += output_size / 2.0 - eye_cx
    M[1, 2] += output_size * ALIGN_EYE_Y_RATIO - eye_cy

    return cv2.warpAffine(frame, M, (output_size, output_size),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING PIPELINE  (shared by enroll + verify paths)
# ══════════════════════════════════════════════════════════════════════════════

def _embed_aligned(aligned_rgb: np.ndarray,
                   tmp_path: str = "temp_face.jpg") -> Optional[np.ndarray]:
    """
    Write an aligned RGB crop to a JPEG temp file, hand it to the Rust
    binary, return the L2-normalised 128-D embedding.

    Why write to disk: the Rust binary is a separate process and accepts a
    file path argument.  Writing JPEG at quality 95 is essentially lossless
    at 112 × 112 px.  The temp file is always deleted on return.
    """
    bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(tmp_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    emb = get_face_embedding(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return l2_normalize(emb) if emb is not None else None


# ══════════════════════════════════════════════════════════════════════════════
#  ENROLLMENT
# ══════════════════════════════════════════════════════════════════════════════

def _wait_countdown(camera: PiCamera, detector: BlazeFaceDetector,
                    display: TFTDisplay, seconds: int = COUNTDOWN_SECONDS) -> None:
    """
    Block in preview mode until a face has been present AND close enough for
    the required number of seconds without moving away.

    The countdown timer resets if:
      • No face is detected in a frame.
      • The largest detected face is smaller than MIN_FACE_PREVIEW_PX.

    This gives the operator time to position themselves and ensures the
    high-resolution capture will hit a close, stable face.
    """
    face_since = None
    while True:
        ok, frame = camera.read_preview()
        if not ok:
            continue
        detects = detector.detect_with_landmarks(frame)
        faces   = [d["box"] for d in detects]

        if faces:
            bx, by, bw, bh = max(faces, key=lambda f: f[2] * f[3])
            if bw < MIN_FACE_PREVIEW_PX or bh < MIN_FACE_PREVIEW_PX:
                face_since = None
                status     = "MOVE CLOSER!"
            else:
                if face_since is None:
                    face_since = time.time()
                remaining = seconds - int(time.time() - face_since)
                if remaining <= 0:
                    display.show_frame(frame, faces=faces, status="CAPTURING…")
                    return
                status = f"HOLD STILL  {remaining}s…"
        else:
            face_since = None
            status     = "SHOW FACE"

        display.show_frame(frame, faces=faces, status=status)


def enroll_face(camera: PiCamera, detector: BlazeFaceDetector,
                display: TFTDisplay) -> Optional[np.ndarray]:
    """
    Capture ONE hi-res frame, align the largest face, and return a 128-D
    L2-normalised embedding for storage in the gallery.

    Why single-sample enrollment:
    ─────────────────────────────
    Averaging multiple embeddings from the same session offers negligible
    accuracy improvement over a single high-quality capture.  What matters
    most is:
      • Face close to camera (large crop → more detail for MobileFaceNet)
      • Aligned eye positions (handled by align_face())
      • Good lighting

    Returns None on failure (caller should notify the operator).
    """
    print("\n🧾  Enrollment — face the camera straight on, close up.")
    _wait_countdown(camera, detector, display, seconds=COUNTDOWN_SECONDS)

    print("  📷  Switching to hi-res…")
    camera._to_hires()
    embedding = None

    try:
        frame   = camera.capture_hires_frame()
        detects = detector.detect_with_landmarks(frame)

        if not detects:
            print("❌  No face detected in hi-res frame.")
            return None

        best = max(detects, key=lambda d: d["box"][2] * d["box"][3])
        bx, by, bw, bh = best["box"]

        if bw < MIN_FACE_CROP_PX or bh < MIN_FACE_CROP_PX:
            print(f"❌  Face too small ({bw}×{bh}px) — move closer!")
            return None

        aligned   = align_face(frame, best["right_eye"], best["left_eye"])
        embedding = _embed_aligned(aligned, "temp_enroll.jpg")

        if embedding is None:
            print("❌  Rust inference failed during enrollment.")
            return None

        print("✅  Enrollment embedding ready.")
        return embedding

    finally:
        print("  🔄  Switching back to preview…")
        camera._to_preview()


# ══════════════════════════════════════════════════════════════════════════════
#  PROBE CAPTURE  (used by commanded verify AND server.py)
# ══════════════════════════════════════════════════════════════════════════════

def capture_probe(camera: PiCamera, detector: BlazeFaceDetector,
                  display: TFTDisplay,
                  use_countdown: bool = True) -> Optional[np.ndarray]:
    """
    Capture ONE hi-res probe embedding for verification.

    use_countdown=True  → shows the "HOLD STILL N s…" countdown (used for
                          operator-commanded verify from the client TUI).
    use_countdown=False → skips countdown (used by continuous auth engine
                          which has already confirmed face size/stability
                          via the track system).

    Returns a 128-D L2-normalised embedding or None.
    """
    if use_countdown:
        _wait_countdown(camera, detector, display, seconds=2)

    print("  📷  Capturing probe (hi-res)…")
    camera._to_hires()
    embedding = None

    try:
        frame   = camera.capture_hires_frame()
        detects = detector.detect_with_landmarks(frame)

        if not detects:
            print("  ⚠️  No face in probe frame.")
            return None

        best = max(detects, key=lambda d: d["box"][2] * d["box"][3])
        bx, by, bw, bh = best["box"]

        if bw < MIN_FACE_CROP_PX or bh < MIN_FACE_CROP_PX:
            print(f"  ⚠️  Face too small in hi-res ({bw}×{bh}px)")
            return None

        aligned   = align_face(frame, best["right_eye"], best["left_eye"])
        embedding = _embed_aligned(aligned, "temp_probe.jpg")

        if embedding is None:
            print("  ❌  Rust inference failed.")

    finally:
        camera._to_preview()

    return embedding


# ══════════════════════════════════════════════════════════════════════════════
#  MATCHING  (open-set, three-layer rejection)
# ══════════════════════════════════════════════════════════════════════════════

def match_probe(probe: np.ndarray,
                db: dict) -> Tuple[bool, str, float, str]:
    """
    Compare a probe embedding against all enrolled templates.

    Returns
    ───────
    (granted: bool, best_name: str, best_score: float, reason: str)

    Three rejection layers (order matters):
    ────────────────────────────────────────
    Layer 1  Gallery size guard
             Prevents false grants when the gallery is too small to provide
             meaningful competition for an impostor embedding.

    Layer 2  Absolute threshold
             If the best cosine score is below SIMILARITY_THRESHOLD, the face
             is unknown regardless of who it most resembles.

    Layer 3  Score-gap check
             If the best and second-best scores are within MIN_SCORE_GAP of
             each other, the match is ambiguous (two people look equally
             similar) and access is denied.  This is the primary guard against
             the "one enrolled identity → impostor gets high score" attack.
    """
    # Layer 1 — gallery size
    if len(db) < MIN_GALLERY_SIZE:
        return (False, "—", 0.0,
                f"gallery has only {len(db)} identity — "
                f"enroll at least {MIN_GALLERY_SIZE} people first")

    # Compute cosine similarity against every enrolled template
    scores = {name: float(np.dot(probe, l2_normalize(np.asarray(tmpl, np.float32))))
              for name, tmpl in db.items()}

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = sorted_scores[0]

    print("\n🔍  Scores:")
    for name, sc in sorted_scores:
        marker = "◀" if name == best_name else " "
        print(f"   {marker} {name:25s}: {sc:.4f}")
    print("-" * 40)

    # Layer 2 — absolute threshold
    if best_score < SIMILARITY_THRESHOLD:
        return (False, best_name, best_score,
                f"score {best_score:.4f} < threshold {SIMILARITY_THRESHOLD}")

    # Layer 3 — gap check
    if len(sorted_scores) > 1:
        gap = best_score - sorted_scores[1][1]
        if gap < MIN_SCORE_GAP:
            return (False, best_name, best_score,
                    f"ambiguous — gap {gap:.4f} < {MIN_SCORE_GAP}")

    return (True, best_name, best_score, "ok")


# ══════════════════════════════════════════════════════════════════════════════
#  FACE TRACKING  (IoU-based, no external dependency)
#
#  Goal: remember which face box belongs to which previously-seen identity so
#  we can:
#    • Show the "Alice | 0.87" label on a GRANTED face without re-running
#      inference on every frame (expensive: ~1.3 s per hi-res capture).
#    • Retry an UNKNOWN face every RETRY_UNKNOWN_S seconds without retrying
#      a GRANTED face until GRANTED_REAUTH_S seconds have passed.
#
#  Algorithm: IoU (Intersection-over-Union) matching.
#    IoU(A, B) = area(A ∩ B) / area(A ∪ B)
#  If IoU > TRACK_IOU_THRESHOLD we consider it the same physical face.
#  This handles small movements (nodding, slight turns) without a complex
#  Kalman filter or DeepSORT tracker.  It is entirely adequate for a
#  door/kiosk scenario where faces move slowly relative to frame rate.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FaceTrack:
    """
    State for one tracked face.

    Fields
    ──────
    box           Current bounding box (x, y, w, h) in the preview frame.
    last_seen     Timestamp of last frame where this box was matched.
    last_attempt  Timestamp of the most recent auth attempt (0 = never tried).
    name          Matched identity name; None = not yet attempted.
    score         Cosine similarity from last attempt (0.0 = not attempted).
    granted       True if last attempt resulted in access granted.
    """
    box:          Tuple[int, int, int, int]
    last_seen:    float = field(default_factory=time.time)
    last_attempt: float = 0.0
    name:         Optional[str] = None      # None = never tried
    score:        float = 0.0
    granted:      bool  = False

    def label_and_color(self) -> Tuple[str, tuple]:
        """
        Return the (label_string, BGR color) pair for the TFT overlay.

        States
        ──────
        Never attempted  →  "scanning…"  YELLOW   (attempt pending)
        Granted          →  "Alice|0.87"  GREEN
        Denied           →  "UNKNOWN"     RED
        """
        if self.name is None:
            return "scanning…", _COL_SCANNING
        if self.granted:
            return f"{self.name}|{self.score:.2f}", _COL_GRANTED
        return "UNKNOWN", _COL_DENIED

    def needs_attempt(self) -> bool:
        """
        Return True if we should fire a hi-res capture+inference now.

        Logic
        ─────
        • Never attempted → always True (highest priority).
        • Previously denied/unknown → True if RETRY_UNKNOWN_S have passed.
        • Previously granted → True if GRANTED_REAUTH_S have passed.
          (Re-authentication eventually ensures the person hasn't swapped out.)
        """
        if self.last_attempt == 0.0:
            return True    # first-ever attempt for this track
        elapsed = time.time() - self.last_attempt
        if self.granted:
            return elapsed >= GRANTED_REAUTH_S
        return elapsed >= RETRY_UNKNOWN_S


def _box_iou(a: tuple, b: tuple) -> float:
    """
    Compute Intersection-over-Union for two (x, y, w, h) boxes.

    IoU = area(intersection) / area(union)

    Returns a float in [0, 1].  0 means no overlap; 1 means identical boxes.
    Used to decide whether two detections in consecutive frames are the same
    physical face.
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    union = aw * ah + bw * bh - inter
    return inter / union if union > 1e-6 else 0.0


def _match_detections_to_tracks(
        detections: List[dict],
        tracks: List[FaceTrack],
) -> Tuple[Dict[int, int], List[int]]:
    """
    Greedy IoU matching between new detections and existing tracks.

    Returns
    ───────
    matched    : {detection_index: track_index}
    unmatched_det : [detection_index]  → new tracks should be created for these
    """
    if not tracks or not detections:
        return {}, list(range(len(detections)))

    # Build IoU matrix: rows = detections, cols = tracks
    iou_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
    for i, det in enumerate(detections):
        for j, trk in enumerate(tracks):
            iou_matrix[i, j] = _box_iou(det["box"], trk.box)

    matched      = {}
    used_tracks  = set()

    # Greedy: always assign the highest IoU pair first
    for _ in range(min(len(detections), len(tracks))):
        if iou_matrix.max() < TRACK_IOU_THRESHOLD:
            break
        i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matched[int(i)] = int(j)
        used_tracks.add(int(j))
        iou_matrix[i, :] = -1   # mark row/col as used
        iou_matrix[:, j] = -1

    unmatched_det = [i for i in range(len(detections)) if i not in matched]
    return matched, unmatched_det


# ══════════════════════════════════════════════════════════════════════════════
#  CONTINUOUS AUTHENTICATION ENGINE
#
#  This is the central new class that ties everything together for the
#  autonomous face authentication loop used by server.py.
#
#  Design goals
#  ────────────
#  1. Every face in FOV gets labelled in real time (preview loop runs at
#     ~25–30 fps regardless of auth attempts).
#  2. Auth attempts are serialised (one hi-res capture at a time) but the
#     preview loop continues showing the last known label for other tracks
#     while an attempt is in progress.
#  3. The engine is interruptible: caller checks cmd_q between frames and
#     can inject a command at any time.
# ══════════════════════════════════════════════════════════════════════════════

class ContinuousAuthEngine:
    """
    Runs the autonomous face identification loop.

    Usage (from server.py face_worker_process or standalone main())
    ───────────────────────────────────────────────────────────────
        engine = ContinuousAuthEngine(audit_log)
        engine.run(camera, detector, display, db,
                   interrupt_fn=lambda: not cmd_q.empty())

    interrupt_fn  — callable that returns True to pause the autonomous loop
                    so the caller can handle a command.  When None, the loop
                    runs until KeyboardInterrupt.
    """

    def __init__(self, audit_log: AuditLog):
        self._log    = audit_log
        self._tracks: List[FaceTrack] = []

    def _prune_stale_tracks(self) -> None:
        """
        Remove tracks whose face has not been seen for TRACK_MAX_AGE_S.
        When a person leaves and re-enters the frame, they get a fresh track
        and a new authentication attempt.
        """
        now = time.time()
        self._tracks = [t for t in self._tracks
                        if now - t.last_seen < TRACK_MAX_AGE_S]

    def _attempt_auth(self, track: FaceTrack, camera: PiCamera,
                      detector: BlazeFaceDetector, display: TFTDisplay,
                      db: dict) -> None:
        """
        Fire one hi-res capture + MobileFaceNet inference for a single track.
        Updates track.name, track.score, track.granted, track.last_attempt.
        Also appends to the audit log.

        This call is BLOCKING (~1.3 s for camera switch + inference).
        The preview loop is paused during this window; the last frame is
        still visible on the TFT from the previous show_frame() call.
        """
        track.last_attempt = time.time()

        # Show "SCANNING…" label while we switch to hi-res
        probe = capture_probe(camera, detector, display, use_countdown=False)

        if probe is None:
            # Hi-res capture failed (face moved away, blurry, too small).
            # Leave name=None so it will be retried after RETRY_UNKNOWN_S.
            print("  ⚠️  [ContinuousAuth] Hi-res probe failed — will retry")
            return

        granted, name, score, reason = match_probe(probe, db)

        track.name    = name if granted else "UNKNOWN"
        track.score   = score
        track.granted = granted

        tag = "GRANTED" if granted else "DENIED"
        print(f"  {'✅' if granted else '❌'}  [AUTO] {tag}  {name}  {score:.4f}")

        # Write to audit log
        self._log.append(
            name       = name,
            confidence = score,
            mode       = "face",
            granted    = granted,
        )

    def run(self, camera: PiCamera, detector: BlazeFaceDetector,
            display: TFTDisplay, db: dict,
            interrupt_fn=None) -> None:
        """
        Main preview + identification loop.

        Frame-by-frame behaviour
        ────────────────────────
        1. Read one 320×240 preview frame.
        2. Run BlazeFace detection (~10 ms on RPi4).
        3. Match detections to existing tracks using IoU.
        4. Update track positions; create new tracks for unmatched detections;
           prune stale tracks.
        5. For the first track that needs an auth attempt (and whose face is
           large enough), run _attempt_auth().  Only ONE attempt per loop
           iteration to keep the camera reconfiguration overhead bounded.
        6. Build per-face label list and call display.show_frame().
        """
        fps_t = time.time()
        fps   = 0.0
        fc    = 0

        while True:
            # ── Interrupt hook: let caller handle commands ──────────────────
            if interrupt_fn and interrupt_fn():
                return

            # ── Grab preview frame ─────────────────────────────────────────
            ok, frame = camera.read_preview()
            if not ok:
                time.sleep(0.005)
                continue

            fc += 1
            if fc % 30 == 0:
                fps   = 30.0 / (time.time() - fps_t)
                fps_t = time.time()

            # ── Face detection ─────────────────────────────────────────────
            detections = detector.detect_with_landmarks(frame)

            # ── Track matching ─────────────────────────────────────────────
            matched, new_det_idxs = _match_detections_to_tracks(
                detections, self._tracks)

            # Update matched tracks with new bounding box and last_seen time
            for det_idx, trk_idx in matched.items():
                self._tracks[trk_idx].box       = detections[det_idx]["box"]
                self._tracks[trk_idx].last_seen = time.time()

            # Create new tracks for unmatched detections
            for i in new_det_idxs:
                self._tracks.append(FaceTrack(box=detections[i]["box"]))

            # Remove tracks not seen recently
            self._prune_stale_tracks()

            # ── Authentication: attempt on the highest-priority track ───────
            # Only attempt if DB is populated and no other attempt is running.
            if db and len(db) >= MIN_GALLERY_SIZE:
                for track in self._tracks:
                    if not track.needs_attempt():
                        continue
                    # Quality gate: face must be large enough in preview
                    bx, by, bw, bh = track.box
                    if bw < MIN_FACE_PREVIEW_PX or bh < MIN_FACE_PREVIEW_PX:
                        continue   # too far away — wait until they move closer
                    # One attempt per loop iteration to bound camera switch cost
                    self._attempt_auth(track, camera, detector, display, db)
                    break

            # ── Build per-face label list for TFT overlay ──────────────────
            face_labels = []
            for track in self._tracks:
                label, color = track.label_and_color()
                face_labels.append({"box": track.box,
                                    "label": label,
                                    "color": color})

            # ── Render frame to TFT ────────────────────────────────────────
            display.show_frame(frame, face_labels=face_labels, fps=fps)


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE MAIN  (keyboard-driven, same commands as original main.py)
#  Also activates the continuous auth engine between commands.
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 65)
    print("  Raspberry Pi 4  ·  face_auth.py  ·  MobileFaceNet + BlazeFace")
    print(f"  Threshold       : {SIMILARITY_THRESHOLD}")
    print(f"  Min score gap   : {MIN_SCORE_GAP}")
    print(f"  Min gallery     : {MIN_GALLERY_SIZE} identities")
    print(f"  Retry unknown   : {RETRY_UNKNOWN_S} s")
    print(f"  Granted reauth  : {GRANTED_REAUTH_S} s")
    print("  Commands: 'a' ADD  |  'v' VERIFY  |  'l' LIST  |  'd' DELETE  |  'q' QUIT")
    print("=" * 65)

    display  = TFTDisplay()
    camera   = PiCamera()
    if not camera.start():
        sys.exit(1)
    detector = BlazeFaceDetector()
    db       = load_database()
    audit    = AuditLog()
    engine   = ContinuousAuthEngine(audit)

    print(f"\n✅  Ready.  DB: {len(db)} identities.\n")

    def _nb_input() -> Optional[str]:
        """Non-blocking stdin read."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip()
        return None

    try:
        # Run the continuous auth engine; it returns when a keystroke arrives
        # via interrupt_fn.  Then we handle the command and re-enter the loop.
        while True:
            # Run engine until a key is pressed
            engine.run(
                camera, detector, display, db,
                interrupt_fn=lambda: bool(_nb_input_peek()))

            # Drain the actual keystroke
            cmd = _nb_input()

            # ── ADD ───────────────────────────────────────────────────────────
            if cmd == "a":
                print("\n" + "=" * 60 + "\n[ ADD FACE ]")
                embedding = enroll_face(camera, detector, display)
                if embedding is not None:
                    name = input("\n👤  Enter name: ").strip()
                    if name:
                        db[name] = embedding
                        save_database(db)
                        print(f"✨  Registered '{name}'  (DB: {len(db)} identities)")
                    else:
                        print("⚠️  Empty name — cancelled.")
                else:
                    print("❌  Enrollment failed.")
                print("=" * 60 + "\n")

            # ── VERIFY (commanded, with countdown) ────────────────────────────
            elif cmd == "v":
                print("\n" + "=" * 60 + "\n[ VERIFY ]")
                if len(db) < MIN_GALLERY_SIZE:
                    print(f"⚠️  Only {len(db)} identity enrolled. "
                          f"Need at least {MIN_GALLERY_SIZE}.")
                else:
                    probe = capture_probe(camera, detector, display,
                                          use_countdown=True)
                    if probe is not None:
                        granted, name, score, reason = match_probe(probe, db)
                        audit.append(name, score, "face", granted)
                        if granted:
                            print(f"✅  GRANTED  |  {name}  |  {score:.4f}")
                            t_end = time.time() + RESULT_DISPLAY_SECONDS
                            while time.time() < t_end:
                                ok2, fr2 = camera.read_preview()
                                if ok2:
                                    display.show_frame(fr2,
                                        face_labels=[{"box": (80, 60, 160, 120),
                                                      "label": f"GRANTED:{name}",
                                                      "color": _COL_GRANTED}])
                        else:
                            print(f"❌  DENIED  |  {reason}")
                            t_end = time.time() + RESULT_DISPLAY_SECONDS
                            while time.time() < t_end:
                                ok2, fr2 = camera.read_preview()
                                if ok2:
                                    display.show_frame(fr2, status="DENIED",
                                                       fps=0)
                print("=" * 60 + "\n")

            # ── LIST ──────────────────────────────────────────────────────────
            elif cmd == "l":
                print("\n" + "=" * 60)
                if not db:
                    print("  (database is empty)")
                else:
                    for i, name in enumerate(db, 1):
                        print(f"  {i:3d}.  {name}")
                print(f"\nTotal: {len(db)} identities")
                print("=" * 60 + "\n")

            # ── DELETE ────────────────────────────────────────────────────────
            elif cmd == "d":
                print("\n" + "=" * 60 + "\n[ DELETE ]")
                if not db:
                    print("  Database is empty.")
                else:
                    for i, name in enumerate(db, 1):
                        print(f"  {i:3d}.  {name}")
                    target = input("\n👤  Enter name to delete: ").strip()
                    if target in db:
                        del db[target]
                        save_database(db)
                        print(f"🗑️   Deleted '{target}'  (DB: {len(db)})")
                    else:
                        print(f"⚠️  '{target}' not found.")
                print("=" * 60 + "\n")

            # ── QUIT ──────────────────────────────────────────────────────────
            elif cmd == "q":
                save_database(db)
                print("👋  Goodbye!")
                break

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
    finally:
        camera.stop()
        detector.close()
        save_database(db)
        display.clear()
        print("✅  Cleanup complete.")


# ── Non-blocking stdin peek (used by interrupt_fn in standalone main) ─────────
_stdin_buf: list = []

def _nb_input_peek() -> bool:
    """
    Return True if a keystroke is waiting in stdin WITHOUT consuming it.
    Uses select() which is POSIX-only (fine for Raspberry Pi).
    """
    return bool(select.select([sys.stdin], [], [], 0)[0])


if __name__ == "__main__":
    main()
