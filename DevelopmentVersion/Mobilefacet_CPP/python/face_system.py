#!/usr/bin/env python3
"""
Face Recognition System for Raspberry Pi 4 - TFT Display Edition
- Detection: MediaPipe BlazeFace (ultra-fast)
- Recognition: MobileFaceNet via ncnn C++ (libface_engine.so)
- Display: ILI9341 TFT 320x240 with CORRECT colors
- Camera: picamera2 RGB888, full FOV via GPU ISP
- Smart Verification: Once per face, re-verify unknown faces every 2s
"""

import ctypes
import numpy as np
import cv2
import time
import sys
import os
import select
import urllib.request

# TFT Display imports
import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import ili9341

# Camera
from picamera2 import Picamera2

# MediaPipe BlazeFace
import mediapipe as mp

# ══════════════════════════════════════════════
# COLOR FIX CONFIGURATION
# ══════════════════════════════════════════════
DISPLAY_NEEDS_BGR = True  # ILI9341 color fix (BGR labeled as RGB)

# ══════════════════════════════════════════════
# PATHS SETUP
# ══════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DB_DIR = os.path.join(PROJECT_DIR, "database")
os.makedirs(DB_DIR, exist_ok=True)

# ══════════════════════════════════════════════
# TFT DISPLAY SETUP (ILI9341 320x240)
# ══════════════════════════════════════════════
print("🖥️  Initializing TFT Display...")
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D27)

spi = board.SPI()
disp = ili9341.ILI9341(
    spi, cs=cs_pin, dc=dc_pin, rst=reset_pin,
    baudrate=64000000, width=240, height=320, rotation=90
)
print("✅ TFT Display initialized: 320×240")

# ══════════════════════════════════════════════
# MEDIAPIPE BLAZEFACE DETECTOR
# ══════════════════════════════════════════════
print("🔍 Initializing MediaPipe BlazeFace...")
mp_face = mp.solutions.face_detection
blazeface = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
print("✅ MediaPipe BlazeFace detector initialized")

# ══════════════════════════════════════════════
# NCNN MOBILEFACENET RECOGNIZER (C++ Library)
# ══════════════════════════════════════════════
LIB_PATH = os.path.join(PROJECT_DIR, "build", "libface_engine.so")
if not os.path.exists(LIB_PATH):
    print(f"❌ Cannot find {LIB_PATH}")
    print("   Build: cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4")
    sys.exit(1)

lib = ctypes.CDLL(LIB_PATH)

# Function signatures
lib.engine_create.restype = ctypes.c_void_p
lib.engine_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.engine_destroy.restype = None
lib.engine_destroy.argtypes = [ctypes.c_void_p]
lib.engine_add_face.restype = ctypes.c_int
lib.engine_add_face.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                 ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.engine_verify.restype = ctypes.c_float
lib.engine_verify.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_int,
                               ctypes.c_char_p, ctypes.c_int]
lib.engine_delete_face.restype = ctypes.c_int
lib.engine_delete_face.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.engine_list_faces.restype = ctypes.c_int
lib.engine_list_faces.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
lib.engine_save_db.restype = ctypes.c_int
lib.engine_save_db.argtypes = [ctypes.c_void_p]

DB_PATH = os.path.join(DB_DIR, "faces.db")
engine = lib.engine_create(MODEL_DIR.encode(), DB_PATH.encode())
if not engine:
    print("❌ Failed to create recognition engine")
    sys.exit(1)
print("✅ Face recognizer loaded (MobileFaceNet ncnn)")


def add_face(name, face_rgb):
    """Add face to database. Input: RGB numpy array"""
    # MobileFaceNet expects BGR, so convert RGB to BGR
    face_bgr = face_rgb[:, :, ::-1].copy()
    f = np.ascontiguousarray(face_bgr)
    h, w = f.shape[:2]
    return lib.engine_add_face(engine, name.encode(), f.ctypes.data, w, h)


def verify_face(face_rgb):
    """Verify face against database. Input: RGB numpy array. Returns (name, score)"""
    # MobileFaceNet expects BGR, so convert RGB to BGR
    face_bgr = face_rgb[:, :, ::-1].copy()
    f = np.ascontiguousarray(face_bgr)
    h, w = f.shape[:2]
    buf = ctypes.create_string_buffer(256)
    score = lib.engine_verify(engine, f.ctypes.data, w, h, buf, 256)
    return buf.value.decode(), float(score)


def delete_face(name):
    """Delete face from database"""
    return lib.engine_delete_face(engine, name.encode())


def list_faces():
    """List all faces in database"""
    buf = ctypes.create_string_buffer(4096)
    count = lib.engine_list_faces(engine, buf, 4096)
    if count == 0:
        return []
    return [n for n in buf.value.decode().split('\n') if n]


# ══════════════════════════════════════════════
# CAMERA SETUP (Full FOV via raw stream)
# ══════════════════════════════════════════════
print("📷 Initializing Camera...")
picam2 = Picamera2()

# Critical: Use raw stream to force full FOV
# Main output: 320x240 RGB888 (matches TFT resolution)
# Raw: Full sensor resolution for complete field of view
camera_config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},  # Match TFT display
    raw={"size": picam2.sensor_resolution},          # Full FOV
    buffer_count=4,
)
picam2.configure(camera_config)
picam2.start()
time.sleep(1)

print(f"✅ Camera initialized: 320×240 RGB, full FOV via GPU ISP")
print(f"   Sensor resolution: {picam2.sensor_resolution}")

# ══════════════════════════════════════════════
# MAIN PROGRAM
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("🎮 CONTROLS:")
print("   a - Add new face to database")
print("   l - List all faces in database")
print("   x - Delete face from database")
print("   q - Quit")
print("="*60 + "\n")

mode = "VERIFY"  # Default mode: always verify faces
add_name = ""
fps_smooth = 0.0
start_time = time.time()

# Face recognition cache with smart re-verification strategy
face_cache = {}  # {(x, y, w, h): (name, score, timestamp, verified)}
UNKNOWN_REVERIFY_INTERVAL = 2.0  # Re-verify unknown faces every 2 seconds
KNOWN_CACHE_PERMANENT = True     # Known faces stay cached as long as in view


def get_cached_result(bbox):
    """
    Smart caching strategy:
    - If face is RECOGNIZED: Cache permanently while in view (no re-verification)
    - If face is UNKNOWN: Re-verify every 2 seconds (might be newly added to DB)

    Returns: (name, score, should_verify)
    """
    current_time = time.time()

    # Clean up old cache entries (faces that left the view)
    to_delete = []
    for cached_bbox, (name, score, timestamp, verified) in list(face_cache.items()):
        # If cache is older than 3 seconds, face probably left view
        if current_time - timestamp > 3.0:
            to_delete.append(cached_bbox)

    for bbox_key in to_delete:
        del face_cache[bbox_key]

    # Check if current face matches any cached face
    x1, y1, w1, h1 = bbox
    for cached_bbox, (name, score, timestamp, verified) in face_cache.items():
        x2, y2, w2, h2 = cached_bbox

        # Calculate overlap between bboxes
        overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        area1 = w1 * h1

        # If 50% overlap, consider it the same face
        if overlap_area > area1 * 0.5:
            # Found matching face in cache
            if name != "Unknown":
                # RECOGNIZED FACE: Use cache, no need to re-verify
                # Update position but keep same result
                face_cache[bbox] = (name, score, current_time, True)
                if cached_bbox != bbox:
                    del face_cache[cached_bbox]
                return name, score, False  # Don't verify again
            else:
                # UNKNOWN FACE: Re-verify every 2 seconds
                time_since_verify = current_time - timestamp
                if time_since_verify >= UNKNOWN_REVERIFY_INTERVAL:
                    # Time to re-verify
                    return name, score, True  # Verify again
                else:
                    # Still within 2-second window, use cached "Unknown"
                    face_cache[bbox] = (name, score, timestamp, verified)
                    if cached_bbox != bbox:
                        del face_cache[cached_bbox]
                    return name, score, False  # Don't verify yet

    # New face, needs verification
    return None, None, True


def handle_command(cmd):
    """Handle keyboard commands"""
    global mode, add_name
    if cmd == 'a':
        add_name = input("Enter name: ").strip()
        if add_name:
            mode = "ADD"
            print(f"📸 Look at camera to add '{add_name}'...")
        else:
            print("❌ Empty name")
    elif cmd == 'l':
        names = list_faces()
        print(f"📋 Database ({len(names)}): " + ", ".join(names) if names else "📋 Database empty")
    elif cmd == 'x':
        n = input("Delete name: ").strip()
        if n:
            print("✅ Deleted" if delete_face(n) == 0 else "❌ Not found")


print("🎬 Starting... Press Ctrl+C to exit.\n")

try:
    while True:
        # Capture frame from camera (320x240 RGB)
        frame_rgb = picam2.capture_array("main")

        # ── COLOR FIX: Convert RGB to BGR for OpenCV processing ──
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Detect faces with MediaPipe BlazeFace (needs RGB)
        results = blazeface.process(frame_rgb)

        face_count = 0

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                # Map relative coordinates to 320x240 frame
                x_min = int(bbox.xmin * 320)
                y_min = int(bbox.ymin * 240)
                width = int(bbox.width * 320)
                height = int(bbox.height * 240)
                x_max = x_min + width
                y_max = y_min + height

                # Ensure bounds are within frame
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(320, x_max)
                y_max = min(240, y_max)
                width = x_max - x_min
                height = y_max - y_min

                if width < 20 or height < 20:
                    continue

                face_count += 1

                # Extract face crop for recognition (RGB for verify_face)
                face_crop = frame_rgb[y_min:y_max, x_min:x_max]

                color_bgr = (0, 255, 0)  # Green in BGR
                label = f"{detection.score[0]:.2f}"

                # ── ADD MODE ──
                if mode == "ADD" and add_name:
                    add_face(add_name, face_crop.copy())
                    print(f"✅ Added: {add_name}")
                    add_name = ""
                    mode = "VERIFY"
                    face_cache.clear()

                # ── VERIFY MODE (Default) ──
                elif mode == "VERIFY":
                    # Check cache first
                    cached_name, cached_score, should_verify = get_cached_result((x_min, y_min, width, height))

                    if should_verify:
                        # Verify face against database
                        name, score = verify_face(face_crop.copy())
                        # Cache result with verified flag
                        face_cache[(x_min, y_min, width, height)] = (name, score, time.time(), True)
                    else:
                        # Use cached result
                        name, score = cached_name, cached_score

                    if name != "Unknown":
                        color_bgr = (0, 255, 0)  # Green for recognized (BGR)
                        label = f"{name} ({score:.2f})"
                    else:
                        color_bgr = (0, 0, 255)  # Red for unknown (BGR)
                        label = f"Unknown ({score:.2f})"

                # Draw bounding box on BGR frame
                cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), color_bgr, 2)

                # Draw label with background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame_bgr, (x_min, y_min - label_h - 4), (x_min + label_w, y_min), (0, 0, 0), -1)
                cv2.putText(frame_bgr, label, (x_min, y_min - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)

        # ── FPS Calculation ──
        elapsed = time.time() - start_time
        fps_smooth = 0.9 * fps_smooth + 0.1 / max(elapsed, 0.001)
        start_time = time.time()

        # ── Draw Status Info ──
        status_text = f"FPS: {fps_smooth:.1f} | Faces: {face_count} | Mode: {mode}"
        if mode == "ADD" and add_name:
            status_text = f"ADDING: {add_name}"

        cv2.putText(frame_bgr, status_text, (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)  # Yellow in BGR

        # ── Display on TFT with COLOR FIX ──
        if DISPLAY_NEEDS_BGR:
            # Trick: Send BGR data but label it as 'RGB' for ILI9341
            pil_img = Image.fromarray(frame_bgr, mode='RGB')
        else:
            # Standard RGB conversion (if your display works differently)
            frame_rgb_display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb_display)

        disp.image(pil_img)

        # ── Handle Keyboard Input ──
        # Check for terminal input (non-blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline().strip()
            if cmd == 'q':
                break
            elif cmd in ('a', 'l', 'x'):
                handle_command(cmd)

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
finally:
    print("\n💾 Saving database...")
    lib.engine_save_db(engine)

    print("🧹 Cleaning up...")
    picam2.stop()
    lib.engine_destroy(engine)
    blazeface.close()
    disp.fill(0)

    print("✅ Cleanup complete")
    print("👋 Bye!")
