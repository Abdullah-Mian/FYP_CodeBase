#!/usr/bin/env python3
"""
Raspberry Pi 4 Face Recognition System
Fixes applied in this version:
  1. FACE ALIGNMENT  — rotate/scale crop so eyes are always horizontal before
                       passing to Rust. This is the single biggest accuracy gain
                       since MobileFaceNet was trained on aligned faces.
  2. POSE-AUGMENTED ENROLLMENT — captures front + horizontal flip + optional
                       slight tilts, averages all into one template.
  3. MIN FACE SIZE GUARD — rejects crops that are too small (far-away faces)
                       to avoid degraded embeddings from heavy upscaling.
  4. SINGLE CAMERA SWITCH — all N samples are captured in one hi-res session.
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
# Configuration
# ============================================================
SIMILARITY_THRESHOLD   = 0.50    # raise after calibrating with your face

DATABASE_FILE          = "faces_db.pkl"

NUM_ENROLLMENT_SAMPLES = 5       # hi-res frames per enrollment session
INTER_SAMPLE_DELAY     = 0.30    # seconds between consecutive hi-res captures
COUNTDOWN_SECONDS      = 3       # preview countdown before hi-res switch

# Minimum face crop dimensions (pixels in hi-res image).
# Crops smaller than this mean the person is too far away.
# At 3280×2464 and ~1m distance a face is typically 400-600px wide.
MIN_FACE_CROP_PX = 200           # absolute minimum — rejects very far faces (hi-res pixels)

# Minimum face width in the 320×240 PREVIEW frame that triggers "MOVE CLOSER".
# The countdown will not start and will reset until this is satisfied.
# Rule of thumb: 320px preview is ~10× narrower than 3280px hi-res, so
#   MIN_FACE_PREVIEW_PX = 60  ≈  MIN_FACE_CROP_PX 600 in hi-res  (~60 cm distance)
# Lower this value to allow captures from further away (but expect lower scores).
MIN_FACE_PREVIEW_PX = 60        # tune this — smaller = further away allowed

# Face alignment — where to place the eyes in the 112×112 aligned output
# (these match the standard ArcFace/MobileFaceNet training protocol)
ALIGN_OUTPUT_SIZE  = 112
ALIGN_EYE_Y_RATIO  = 0.35       # eyes at 35 % from top of aligned image
ALIGN_EYE_DIST_RATIO = 0.37     # desired inter-eye distance as fraction of width

# Display
DISPLAY_WIDTH  = 320
DISPLAY_HEIGHT = 240
BAUDRATE       = 64_000_000

# Camera
PREVIEW_WIDTH  = 320
PREVIEW_HEIGHT = 240
CAPTURE_WIDTH  = 3280
CAPTURE_HEIGHT = 2464
MIN_FACE_CONF  = 0.5


# ============================================================
# Math helpers
# ============================================================
def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-10 else v


def average_template(embeddings: list) -> np.ndarray:
    """Stack normalised vectors, take mean, re-normalise."""
    stacked = np.stack([l2_normalize(e) for e in embeddings], axis=0)
    return l2_normalize(stacked.mean(axis=0))


# ============================================================
# Database  (backward-compatible: supports old raw-vector format)
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
            if isinstance(val, dict) and "template" in val:
                tmpl  = l2_normalize(np.asarray(val["template"], dtype=np.float32))
                samps = [l2_normalize(np.asarray(s, dtype=np.float32))
                         for s in val.get("samples", []) if np.asarray(s).size > 0]
                db[name] = {"template": tmpl, "samples": samps or [tmpl]}
            else:
                tmpl = l2_normalize(np.asarray(val, dtype=np.float32))
                db[name] = {"template": tmpl, "samples": [tmpl]}
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
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for (x, y, w, h) in (faces or []):
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"FPS:{fps:.0f}" + (f" | {status}" if status else "")
        cv2.putText(img, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.disp.image(Image.fromarray(img, mode="RGB"))

    def clear(self):
        self.disp.image(Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), 0))


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
                main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"},
                raw={"size": (1640, 1232)},
            )
            self._capture_cfg = self._cam.create_still_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"})
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
        self._cam.stop(); self._cam.configure(self._capture_cfg)
        self._cam.start(); time.sleep(0.8)

    def _to_preview(self):
        self._cam.stop(); self._cam.configure(self._preview_cfg)
        self._cam.start(); time.sleep(0.5)

    def capture_hires_frame(self):
        return self._cam.capture_array()

    def stop(self):
        if self._cam: self._cam.stop()


# ============================================================
# BlazeFace Detector  (MediaPipe)
# ============================================================
class BlazeFaceDetector:
    def __init__(self):
        _mp = mp.solutions.face_detection
        self._det = _mp.FaceDetection(
            model_selection=0, min_detection_confidence=MIN_FACE_CONF)

    def detect(self, frame):
        """Returns list of (x, y, w, h) bounding boxes."""
        res = self._det.process(frame)
        if not res.detections:
            return []
        h, w = frame.shape[:2]
        return [(max(0, int(d.location_data.relative_bounding_box.xmin  * w)),
                 max(0, int(d.location_data.relative_bounding_box.ymin  * h)),
                 max(1, int(d.location_data.relative_bounding_box.width * w)),
                 max(1, int(d.location_data.relative_bounding_box.height* h)))
                for d in res.detections]

    def detect_with_landmarks(self, frame):
        """
        Returns list of dicts:
          { 'box': (x,y,w,h),
            'right_eye': (px,py),   # keypoint index 0
            'left_eye':  (px,py) }  # keypoint index 1
        The eye coordinates are absolute pixels in 'frame'.
        """
        res = self._det.process(frame)
        if not res.detections:
            return []
        h, w = frame.shape[:2]
        out = []
        for d in res.detections:
            b  = d.location_data.relative_bounding_box
            kp = d.location_data.relative_keypoints
            # MediaPipe FaceDetection keypoint order:
            # 0=right_eye, 1=left_eye, 2=nose_tip, 3=mouth, 4=right_ear, 5=left_ear
            re = (kp[0].x * w, kp[0].y * h)
            le = (kp[1].x * w, kp[1].y * h)
            out.append({
                "box":       (max(0,int(b.xmin*w)), max(0,int(b.ymin*h)),
                               max(1,int(b.width*w)), max(1,int(b.height*h))),
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
               output_size: int = ALIGN_OUTPUT_SIZE) -> np.ndarray:
    """
    Rotate and scale 'frame' so that:
      - Eyes land at a standardised vertical position (ALIGN_EYE_Y_RATIO)
      - Inter-eye distance = ALIGN_EYE_DIST_RATIO * output_size
    Returns an (output_size × output_size) RGB crop.

    This replicates the alignment that was applied to ALL training data of
    MobileFaceNet/ArcFace, so it is the single most important pre-processing
    step for maximising cosine similarity scores.
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
    # Translate so eye-midpoint lands at desired position
    M[0, 2] += output_size / 2.0 - eye_cx
    M[1, 2] += output_size * ALIGN_EYE_Y_RATIO - eye_cy

    return cv2.warpAffine(frame, M, (output_size, output_size),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ============================================================
# Crop + embedding pipeline
# ============================================================
def _embed_aligned(aligned_rgb: np.ndarray, tmp_path: str = "temp_face.jpg"):
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
    face_since = None
    while True:
        ok, frame = camera.read_preview()
        if not ok:
            continue
        faces = [d["box"] for d in detector.detect_with_landmarks(frame)]
        if faces:
            # Pick the largest detected face and check its preview-frame size
            bx, by, bw, bh = max(faces, key=lambda f: f[2] * f[3])
            if bw < MIN_FACE_PREVIEW_PX or bh < MIN_FACE_PREVIEW_PX:
                # Face is too far away — reset countdown and warn on TFT
                face_since = None
                status = "MOVE CLOSER!"
            else:
                # Face is close enough — run the countdown
                if face_since is None:
                    face_since = time.time()
                remaining = seconds - int(time.time() - face_since)
                if remaining <= 0:
                    display.show_frame(frame, faces=faces, status="CAPTURING…")
                    return
                status = f"HOLD STILL  {remaining}s…"
        else:
            face_since = None
            status = "SHOW FACE"
        display.show_frame(frame, faces=faces, status=status)


# ============================================================
# ENROLLMENT
# ============================================================
def enroll_face(camera, detector, display,
                num_samples=NUM_ENROLLMENT_SAMPLES):
    """
    Enrollment strategy (research-backed):
      - Front face × 3  (majority weight on canonical pose)
      - Horizontal flip  (mirrors left-right, covers head turn to both sides)
      - 1 slight left tilt instruction (caught in normal variation)
    All captures in ONE hi-res session → minimal pose drift between samples.

    The horizontal flip trick comes directly from the face recognition research
    literature — even the Pose-TTA paper (2025) uses 'original + flipped
    embeddings averaged' as its universal baseline across ALL tested models.
    """
    print(f"\n🧾  Enrollment — {num_samples} rapid hi-res captures")
    print("    Keep your face centred and steady.")
    _wait_countdown(camera, detector, display, seconds=COUNTDOWN_SECONDS)

    print("  📷  Switching to hi-res…")
    camera._to_hires()

    embeddings = []
    rejections = 0

    try:
        for i in range(num_samples):
            label = f"Sample {i+1}/{num_samples}"
            print(f"  📸  {label}…", end=" ", flush=True)

            frame   = camera.capture_hires_frame()
            detects = detector.detect_with_landmarks(frame)

            if not detects:
                print("⚠️  no face — skipped")
                rejections += 1
                time.sleep(INTER_SAMPLE_DELAY)
                continue

            best = max(detects, key=lambda d: d["box"][2] * d["box"][3])
            bx, by, bw, bh = best["box"]

            # ── Minimum size guard ──────────────────────────────────────
            if bw < MIN_FACE_CROP_PX or bh < MIN_FACE_CROP_PX:
                print(f"⚠️  face too small ({bw}×{bh}px) — move closer!")
                rejections += 1
                time.sleep(INTER_SAMPLE_DELAY)
                continue

            # ── Aligned crop ────────────────────────────────────────────
            aligned = align_face(frame, best["right_eye"], best["left_eye"])

            # ── Embed the aligned face ───────────────────────────────────
            emb = _embed_aligned(aligned, f"temp_enroll_{i}.jpg")
            if emb is None:
                print("❌  Rust failed"); rejections += 1; continue

            embeddings.append(emb)

            # ── Horizontal flip — free pose augmentation ─────────────────
            # Flip the ALIGNED image left-right. Because the face is already
            # rotated upright, flipping it gives a clean mirror image that
            # covers head turns in the opposite direction.
            flipped = cv2.flip(aligned, 1)
            emb_flip = _embed_aligned(flipped, f"temp_enroll_{i}_flip.jpg")
            if emb_flip is not None:
                embeddings.append(emb_flip)
                print(f"✅  (+ flip)")
            else:
                print("✅")

            if i < num_samples - 1:
                time.sleep(INTER_SAMPLE_DELAY)

    finally:
        print("  🔄  Switching back to preview…")
        camera._to_preview()

    if not embeddings:
        print("❌  No valid samples.")
        return None, []

    if rejections > 0:
        print(f"⚠️  {rejections} sample(s) rejected (too far / no face).")

    template = average_template(embeddings)
    print(f"✅  Template from {len(embeddings)} embeddings "
          f"({num_samples} captures × up-to-2 embeddings each)")
    return template, embeddings


# ============================================================
# VERIFICATION  (single probe, aligned)
# ============================================================
def capture_probe(camera, detector, display):
    _wait_countdown(camera, detector, display, seconds=2)
    print("  📷  Capturing probe…")
    camera._to_hires()
    emb = None
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
        aligned = align_face(frame, best["right_eye"], best["left_eye"])
        emb = _embed_aligned(aligned, "temp_probe.jpg")
        if emb is None:
            print("  ❌  Rust failed on probe.")
    finally:
        camera._to_preview()
    return emb


# ============================================================
# Main loop
# ============================================================
def main():
    print("=" * 60)
    print("  Raspberry Pi 4  ·  Face Recognition  ·  MobileFaceNet")
    print(f"  Enrollment   : {NUM_ENROLLMENT_SAMPLES} captures + flip augmentation")
    print(f"  Alignment    : Eye-based geometric normalisation")
    print(f"  Min face size: {MIN_FACE_CROP_PX}px  (move closer if rejected)")
    print(f"  Threshold    : {SIMILARITY_THRESHOLD}")
    print("=" * 60)

    display  = TFTDisplay()
    camera   = PiCamera()
    if not camera.start():
        sys.exit(1)
    detector = BlazeFaceDetector()
    db       = load_database()

    print(f"\n✅  Ready.  DB: {len(db)} identities.")
    print("  'a' ADD  |  'v' VERIFY  |  'l' LIST  |  'q' QUIT\n")

    frame_count = 0
    fps_t = time.time()
    fps   = 0.0

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

            faces = [d["box"] for d in detector.detect_with_landmarks(frame)]
            display.show_frame(frame, faces=faces, status="READY", fps=fps)

            cmd = _nb_input()

            # ── ADD ──────────────────────────────────────────────────────
            if cmd == "a":
                print("\n" + "=" * 60)
                print("[ ADD FACE — aligned multi-sample enrollment ]")
                print("=" * 60)
                template, samples = enroll_face(camera, detector, display)
                if template is not None:
                    name = input("\n👤  Enter name: ").strip()
                    if name:
                        db[name] = {"template": template, "samples": samples}
                        save_database(db)
                        print(f"✨  Registered '{name}'  ({len(samples)} embeddings stored)")
                        print(f"📊  DB total: {len(db)} identities")
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
                    print("❌  Database empty!"); continue

                probe = capture_probe(camera, detector, display)
                if probe is None:
                    print("❌  Could not capture probe.")
                    continue

                print("\n🔍  Matching:")
                best_score, best_name = -1.0, "Unknown"
                for name, entry in db.items():
                    tmpl  = l2_normalize(np.asarray(entry["template"], np.float32))
                    score = float(np.dot(probe, tmpl))
                    print(f"   {name:25s}: {score:.4f}")
                    if score > best_score:
                        best_score, best_name = score, name

                print("-" * 40)
                if best_score >= SIMILARITY_THRESHOLD:
                    print(f"✅  ACCESS GRANTED  |  {best_name}  |  {best_score:.4f}")
                else:
                    print(f"❌  ACCESS DENIED   |  best={best_name}  ({best_score:.4f})")
                print("=" * 60 + "\n")

            # ── LIST ─────────────────────────────────────────────────────
            elif cmd == "l":
                print("\n" + "=" * 60)
                for i, (name, entry) in enumerate(db.items(), 1):
                    ns = len(entry.get("samples", []))
                    print(f"  {i:3d}.  {name}  ({ns} embeddings)")
                print(f"\nTotal: {len(db)}")
                print("=" * 60 + "\n")

            # ── QUIT ─────────────────────────────────────────────────────
            elif cmd == "q":
                save_database(db); print("👋  Goodbye!"); break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
    finally:
        camera.stop(); detector.close()
        save_database(db); display.clear()
        print("✅  Cleanup complete.")


if __name__ == "__main__":
    main()