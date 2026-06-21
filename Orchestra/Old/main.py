#!/usr/bin/env python3
"""
Raspberry Pi 4 Face Recognition System
"""

import time
import board
import digitalio
from PIL import Image, ImageFont
from adafruit_rgb_display import ili9341
import cv2
import numpy as np
import os
import pickle
import sys
import select

try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ Picamera2 not found!"); sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("❌ MediaPipe not found!"); sys.exit(1)

try:
    from rust_wrapper import get_face_embedding, compute_similarity
except ImportError:
    print("❌ rust_wrapper.py not found!"); sys.exit(1)


# ============================================================
# Configuration  — edit these values to tune the system
# ============================================================

# ── Security threshold ───────────────────────────────────────
# Genuine match at close range should score 0.80–0.92.
# Raise this if strangers are being granted.
# Lower this only if your own face is being denied.
SIMILARITY_THRESHOLD = 0.75

# ── Gap check ────────────────────────────────────────────────
# If the best score and second-best score are closer than this,
# the match is ambiguous and access is denied.
# This catches the "only one person enrolled → high stranger score" problem.
MIN_SCORE_GAP = 0.10

# ── Minimum enrolled identities ──────────────────────────────
# With only 1 person enrolled, an unknown face has no competition
# and can score falsely high. Require at least this many before
# authentication is enabled.
MIN_GALLERY_SIZE = 2

# ── Database ─────────────────────────────────────────────────
DATABASE_FILE = "faces_db.pkl"

# ── Countdown before capture ─────────────────────────────────
COUNTDOWN_SECONDS = 3

# ── Distance guard ───────────────────────────────────────────
# Minimum face crop dimensions in the HI-RES frame (pixels).
# Crops smaller than this mean the person is too far away.
MIN_FACE_CROP_PX = 200

# Minimum face width in the 320×240 PREVIEW frame.
# Countdown resets and TFT shows "MOVE CLOSER!" until this is met.
# 60px in preview ≈ 600px in hi-res ≈ 60 cm distance.
# Lower = allows further distance (but expect lower scores).
MIN_FACE_PREVIEW_PX = 60

# ── Face alignment parameters ────────────────────────────────
# Where to place eyes in the 112×112 aligned crop.
# These match the ArcFace/MobileFaceNet training protocol exactly.
ALIGN_OUTPUT_SIZE    = 112
ALIGN_EYE_Y_RATIO    = 0.35    # eyes at 35% from top
ALIGN_EYE_DIST_RATIO = 0.37    # inter-eye distance as fraction of width

# ── Display ──────────────────────────────────────────────────
DISPLAY_WIDTH  = 320
DISPLAY_HEIGHT = 240
BAUDRATE       = 64_000_000

# ── Camera ───────────────────────────────────────────────────
PREVIEW_WIDTH  = 320
PREVIEW_HEIGHT = 240
CAPTURE_WIDTH  = 3280
CAPTURE_HEIGHT = 2464
MIN_FACE_CONF  = 0.5

# ── Result flash duration ────────────────────────────────────
RESULT_DISPLAY_SECONDS = 2.0


# ============================================================
# Math helpers
# ============================================================
def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-10 else v


# ============================================================
# Database  (backward-compatible with old raw-vector format)
# ============================================================
def load_database() -> dict:
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
            if isinstance(val, np.ndarray):
                # Old format: raw embedding stored directly
                db[name] = l2_normalize(val)
            elif isinstance(val, dict) and "template" in val:
                # Previous multi-sample format
                db[name] = l2_normalize(
                    np.asarray(val["template"], dtype=np.float32))
            else:
                try:
                    db[name] = l2_normalize(
                        np.asarray(val, dtype=np.float32))
                except Exception:
                    print(f"⚠️  Skipping corrupt entry for '{name}'")
        print(f"✅ Loaded DB: {len(db)} identities")
        return db
    except Exception as exc:
        print(f"⚠️  DB load failed ({exc}) — starting fresh.")
        return {}


def save_database(db: dict) -> None:
    with open(DATABASE_FILE, "wb") as fh:
        pickle.dump(db, fh)


# ============================================================
# Display
# ============================================================
class TFTDisplay:
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
        print("✅ TFT Display initialised")

    def show_frame(self, frame, faces=None, status="", fps=0.0):
        # Fix mirror: flip the preview frame horizontally so it looks natural
        img = cv2.flip(frame, 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Bounding boxes also need to be mirrored since we flipped the frame
        if faces:
            frame_w = frame.shape[1]
            for (x, y, w, h) in faces:
                flipped_x = frame_w - x - w
                cv2.rectangle(img, (flipped_x, y),
                              (flipped_x + w, y + h), (0, 255, 0), 2)

        label = f"FPS:{fps:.0f}" + (f" | {status}" if status else "")

        # NOTE: This display pipeline does RGB→BGR then sends as mode="RGB",
        # which swaps R and B channels on screen. Green is symmetric so it
        # is unaffected. For other colors R and B are intentionally swapped here.
        if "CLOSER" in status:
            color = (255, 0, 0)      # appears RED on screen
        elif "GRANTED" in status:
            color = (0, 255, 0)      # appears GREEN on screen
        elif "DENIED" in status:
            color = (255, 0, 0)      # appears RED on screen
        elif "CAPTURING" in status:
            color = (255, 255, 0)    # appears CYAN on screen
        elif "HOLD" in status:
            color = (0, 255, 255)    # appears YELLOW on screen
        else:
            color = (255, 255, 255)  # WHITE

        cv2.putText(img, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)
        self.disp.image(Image.fromarray(img, mode="RGB"))

    def clear(self):
        self.disp.image(
            Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0))


# ============================================================
# Camera
# ============================================================
class PiCamera:
    def __init__(self):
        self._cam = self._preview_cfg = self._capture_cfg = None

    def start(self) -> bool:
        try:
            self._cam = Picamera2()
            self._preview_cfg = self._cam.create_preview_configuration(
                main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                      "format": "RGB888"},
                raw={"size": (1640, 1232)},
            )
            self._capture_cfg = self._cam.create_still_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT),
                      "format": "RGB888"})
            self._cam.configure(self._preview_cfg)
            self._cam.start()
            time.sleep(2)
            print(f"✅ Camera ready  preview={PREVIEW_WIDTH}×{PREVIEW_HEIGHT}  "
                  f"hires={CAPTURE_WIDTH}×{CAPTURE_HEIGHT}")
            return True
        except Exception as exc:
            print(f"❌ Camera init: {exc}"); return False

    def read_preview(self):
        try:    return True, self._cam.capture_array()
        except: return False, None

    def _to_hires(self):
        self._cam.stop()
        self._cam.configure(self._capture_cfg)
        self._cam.start()
        time.sleep(0.8)

    def _to_preview(self):
        self._cam.stop()
        self._cam.configure(self._preview_cfg)
        self._cam.start()
        time.sleep(0.5)

    def capture_hires_frame(self):
        return self._cam.capture_array()

    def stop(self):
        if self._cam:
            self._cam.stop()


# ============================================================
# BlazeFace Detector  (MediaPipe)
# ============================================================
class BlazeFaceDetector:
    def __init__(self):
        _mp = mp.solutions.face_detection
        self._det = _mp.FaceDetection(
            model_selection=0,
            min_detection_confidence=MIN_FACE_CONF)

    def detect_with_landmarks(self, frame):
        """
        Returns list of dicts:
          { 'box': (x, y, w, h),
            'right_eye': (px, py),   # keypoint index 0
            'left_eye':  (px, py) }  # keypoint index 1
        """
        res = self._det.process(frame)
        if not res.detections:
            return []
        h, w = frame.shape[:2]
        out = []
        for d in res.detections:
            b  = d.location_data.relative_bounding_box
            kp = d.location_data.relative_keypoints
            re = (kp[0].x * w, kp[0].y * h)
            le = (kp[1].x * w, kp[1].y * h)
            out.append({
                "box":       (max(0, int(b.xmin  * w)),
                              max(0, int(b.ymin  * h)),
                              max(1, int(b.width * w)),
                              max(1, int(b.height* h))),
                "right_eye": re,
                "left_eye":  le,
            })
        return out

    def close(self):
        self._det.close()


# ============================================================
# Face Alignment
# ============================================================
def align_face(frame: np.ndarray,
               right_eye_xy: tuple,
               left_eye_xy:  tuple,
               output_size:  int = ALIGN_OUTPUT_SIZE) -> np.ndarray:
    """
    Rotate and scale frame so eyes land at standardised positions.
    This replicates the alignment applied to ALL MobileFaceNet/ArcFace
    training data — it is the single most important pre-processing step.
    """
    rx, ry = right_eye_xy
    lx, ly = left_eye_xy

    angle    = np.degrees(np.arctan2(ly - ry, lx - rx))
    eye_cx   = (rx + lx) / 2.0
    eye_cy   = (ry + ly) / 2.0
    eye_dist = np.hypot(lx - rx, ly - ry) + 1e-6

    desired_dist = output_size * ALIGN_EYE_DIST_RATIO
    scale        = desired_dist / eye_dist

    M = cv2.getRotationMatrix2D((eye_cx, eye_cy), angle, scale)
    M[0, 2] += output_size / 2.0 - eye_cx
    M[1, 2] += output_size * ALIGN_EYE_Y_RATIO - eye_cy

    return cv2.warpAffine(frame, M, (output_size, output_size),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ============================================================
# Crop + embedding pipeline
# ============================================================
def _embed_aligned(aligned_rgb: np.ndarray,
                   tmp_path: str = "temp_face.jpg"):
    """Write aligned crop to disk, call Rust, return L2-normalised embedding."""
    bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(tmp_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    emb = get_face_embedding(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return l2_normalize(emb) if emb is not None else None


# ============================================================
# Preview countdown  (preview mode — NO camera switch)
# ============================================================
def _wait_countdown(camera, detector, display, seconds=COUNTDOWN_SECONDS):
    """
    Runs entirely in preview mode.
    Shows MOVE CLOSER if face is too small — resets countdown timer.
    Returns only when face is present, close enough, and countdown has elapsed.
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
                face_since = None   # reset countdown — too far away
                status = "MOVE CLOSER!"
            else:
                if face_since is None:
                    face_since = time.time()
                remaining = seconds - int(time.time() - face_since)
                if remaining <= 0:
                    display.show_frame(frame, faces=faces,
                                       status="CAPTURING…")
                    return
                status = f"HOLD STILL  {remaining}s…"
        else:
            face_since = None
            status = "SHOW FACE"

        display.show_frame(frame, faces=faces, status=status)


# ============================================================
# ENROLLMENT  — single capture, one embedding
# ============================================================
def enroll_face(camera, detector, display):
    """
    Capture exactly ONE hi-res frame, align it, embed it.
    Returns a single L2-normalised 128-D numpy array, or None on failure.

    Why single capture:
      - Multiple captures from the same session produced no accuracy gain
        — the distance-from-camera and alignment quality matter far more
        than averaging multiple embeddings.
      - One clean, close, front-facing, aligned capture is the best template.
    """
    print("\n🧾  Enrollment — single capture")
    print("    Stand close, face the camera straight on.")
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
            print("❌  Rust inference failed.")
            return None

        print("✅  Enrollment embedding captured.")
        return embedding

    finally:
        print("  🔄  Switching back to preview…")
        camera._to_preview()


# ============================================================
# VERIFICATION  — single capture, one embedding
# ============================================================
def capture_probe(camera, detector, display):
    """
    Capture exactly ONE hi-res frame for verification.
    Returns a single L2-normalised embedding, or None on failure.
    """
    _wait_countdown(camera, detector, display, seconds=2)
    print("  📷  Capturing probe…")
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
            print(f"  ⚠️  Face too small ({bw}×{bh}px) — move closer!")
            return None

        aligned   = align_face(frame, best["right_eye"], best["left_eye"])
        embedding = _embed_aligned(aligned, "temp_probe.jpg")

        if embedding is None:
            print("  ❌  Rust inference failed.")

    finally:
        camera._to_preview()

    return embedding


# ============================================================
# MATCHING  — open-set safe
# ============================================================
def match_probe(probe: np.ndarray, db: dict):
    """
    Compare probe embedding against all enrolled templates.

    Returns (granted: bool, best_name: str, best_score: float, reason: str)

    Three-layer rejection:
      1. Gallery too small  → always deny (not enough competition for impostor)
      2. Best score below SIMILARITY_THRESHOLD  → deny
      3. Gap between best and second-best score too small → deny (ambiguous)
    """
    # ── Layer 1: gallery size guard ─────────────────────────────
    if len(db) < MIN_GALLERY_SIZE:
        return (False, "—", 0.0,
                f"Gallery has only {len(db)} identity — "
                f"enroll at least {MIN_GALLERY_SIZE} people first")

    # ── Compute all scores ───────────────────────────────────────
    scores = {}
    for name, template in db.items():
        tmpl         = l2_normalize(np.asarray(template, dtype=np.float32))
        scores[name] = float(np.dot(probe, tmpl))

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name,  best_score  = sorted_scores[0]
    print("\n🔍  Matching:")
    for name, sc in sorted_scores:
        print(f"   {name:25s}: {sc:.4f}")
    print("-" * 40)

    # ── Layer 2: threshold ───────────────────────────────────────
    if best_score < SIMILARITY_THRESHOLD:
        return (False, best_name, best_score,
                f"score {best_score:.4f} below threshold {SIMILARITY_THRESHOLD}")

    # ── Layer 3: gap check ───────────────────────────────────────
    if len(sorted_scores) > 1:
        second_score = sorted_scores[1][1]
        gap = best_score - second_score
        if gap < MIN_SCORE_GAP:
            return (False, best_name, best_score,
                    f"ambiguous — gap {gap:.4f} < {MIN_SCORE_GAP} "
                    f"(best={best_score:.4f}, 2nd={second_score:.4f})")

    return (True, best_name, best_score, "ok")


# ============================================================
# Main loop
# ============================================================
def main():
    print("=" * 60)
    print("  Raspberry Pi 4  ·  Face Recognition  ·  MobileFaceNet")
    print(f"  Threshold     : {SIMILARITY_THRESHOLD}")
    print(f"  Min score gap : {MIN_SCORE_GAP}")
    print(f"  Min gallery   : {MIN_GALLERY_SIZE} identities")
    print(f"  Min face size : {MIN_FACE_CROP_PX}px hi-res "
          f"| {MIN_FACE_PREVIEW_PX}px preview")
    print("=" * 60)

    display  = TFTDisplay()
    camera   = PiCamera()
    if not camera.start():
        sys.exit(1)
    detector = BlazeFaceDetector()
    db       = load_database()

    print(f"\n✅  Ready.  DB: {len(db)} identities.")
    print("  'a' ADD  |  'v' VERIFY  |  'l' LIST  |  'd' DELETE  |  'q' QUIT\n")

    frame_count = 0
    fps_t       = time.time()
    fps         = 0.0

    def _nb_input():
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip()
        return None

    try:
        while True:
            ok, frame = camera.read_preview()
            if not ok:
                continue

            frame_count += 1
            if frame_count % 30 == 0:
                fps   = 30.0 / (time.time() - fps_t)
                fps_t = time.time()

            detects = detector.detect_with_landmarks(frame)
            faces   = [d["box"] for d in detects]
            display.show_frame(frame, faces=faces, status="READY", fps=fps)

            cmd = _nb_input()

            # ── ADD ──────────────────────────────────────────────────────
            if cmd == "a":
                print("\n" + "=" * 60)
                print("[ ADD FACE ]")
                print("=" * 60)
                embedding = enroll_face(camera, detector, display)
                if embedding is not None:
                    name = input("\n👤  Enter name: ").strip()
                    if name:
                        db[name] = embedding
                        save_database(db)
                        print(f"✨  Registered '{name}'")
                        print(f"📊  DB total: {len(db)} identities")
                        if len(db) < MIN_GALLERY_SIZE:
                            print(f"⚠️  Enroll at least {MIN_GALLERY_SIZE} "
                                  f"people before using VERIFY.")
                    else:
                        print("⚠️  Empty name — cancelled.")
                else:
                    print("❌  Enrollment failed.")
                print("=" * 60 + "\n")

            # ── VERIFY ───────────────────────────────────────────────────
            elif cmd == "v":
                print("\n" + "=" * 60)
                print("[ VERIFY ]")

                if not db:
                    print("❌  Database empty!")
                    continue

                if len(db) < MIN_GALLERY_SIZE:
                    print(f"⚠️  Only {len(db)} identity enrolled. "
                          f"Need at least {MIN_GALLERY_SIZE} to verify safely.")
                    continue

                probe = capture_probe(camera, detector, display)
                if probe is None:
                    print("❌  Could not capture probe.")
                    continue

                granted, best_name, best_score, reason = match_probe(probe, db)

                if granted:
                    msg = f"✅  ACCESS GRANTED  |  {best_name}  |  {best_score:.4f}"
                    print(msg)
                    # Flash result on TFT for RESULT_DISPLAY_SECONDS
                    t_end = time.time() + RESULT_DISPLAY_SECONDS
                    while time.time() < t_end:
                        ok2, fr2 = camera.read_preview()
                        if ok2:
                            display.show_frame(
                                fr2, status=f"GRANTED {best_name}", fps=fps)
                else:
                    msg = f"❌  ACCESS DENIED   |  reason: {reason}"
                    print(msg)
                    t_end = time.time() + RESULT_DISPLAY_SECONDS
                    while time.time() < t_end:
                        ok2, fr2 = camera.read_preview()
                        if ok2:
                            display.show_frame(
                                fr2, status="ACCESS DENIED", fps=fps)

                print("=" * 60 + "\n")

            # ── LIST ─────────────────────────────────────────────────────
            elif cmd == "l":
                print("\n" + "=" * 60)
                if not db:
                    print("  (database is empty)")
                else:
                    for i, name in enumerate(db, 1):
                        print(f"  {i:3d}.  {name}")
                print(f"\nTotal: {len(db)} identities")
                print("=" * 60 + "\n")

            # ── DELETE ───────────────────────────────────────────────────
            elif cmd == "d":
                print("\n" + "=" * 60)
                print("[ DELETE FACE ]")
                if not db:
                    print("  Database is empty.")
                else:
                    for i, name in enumerate(db, 1):
                        print(f"  {i:3d}.  {name}")
                    target = input("\n👤  Enter name to delete: ").strip()
                    if target in db:
                        del db[target]
                        save_database(db)
                        print(f"🗑️   Deleted '{target}'")
                        print(f"📊  DB total: {len(db)} identities")
                    else:
                        print(f"⚠️  '{target}' not found.")
                print("=" * 60 + "\n")

            # ── QUIT ─────────────────────────────────────────────────────
            elif cmd == "q":
                save_database(db)
                print("👋  Goodbye!")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
    finally:
        camera.stop()
        detector.close()
        save_database(db)
        display.clear()
        print("✅  Cleanup complete.")


if __name__ == "__main__":
    main()
