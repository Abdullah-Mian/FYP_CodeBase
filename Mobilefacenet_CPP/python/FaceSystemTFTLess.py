#!/usr/bin/env python3
"""
Face Recognition System for Raspberry Pi 4
- Detection: MediaPipe BlazeFace (GPU-accelerated on supported platforms)
- Recognition: MobileFaceNet via ncnn C++ shared library (CPU-based, 128-dim embedding)
- Display: OpenCV window on monitor
- Camera: picamera2 RGB888, full FOV via GPU ISP
- Smart Caching: Verify once per recognized face, re-verify unknown every 2s
"""

import ctypes
import numpy as np
import cv2
import time
import sys
import os
import select
import urllib.request

# Camera
from picamera2 import Picamera2

# MediaPipe BlazeFace
import mediapipe as mp

# ══════════════════════════════════════════════
# PATHS SETUP
# ══════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DB_DIR = os.path.join(PROJECT_DIR, "database")
os.makedirs(DB_DIR, exist_ok=True)

# ══════════════════════════════════════════════
# MEDIAPIPE BLAZEFACE DETECTOR
# ══════════════════════════════════════════════
print("🔍 Initializing MediaPipe BlazeFace...")
mp_face = mp.solutions.face_detection
blazeface = mp_face.FaceDetection(
    model_selection=0,                    # 0 = short-range model (0-2m), 1 = full-range (0-5m)
    min_detection_confidence=0.5          # Minimum confidence threshold
)
print("✅ MediaPipe BlazeFace detector initialized")

# ══════════════════════════════════════════════
# NCNN MOBILEFACENET RECOGNIZER (C++ Library)
# ══════════════════════════════════════════════
LIB_PATH = os.path.join(PROJECT_DIR, "build", "libface_engine.so")
if not os.path.exists(LIB_PATH):
    print(f"❌ Cannot find {LIB_PATH}")
    print("   Build: cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4")
    sys.exit(1)

# Load shared library
lib = ctypes.CDLL(LIB_PATH)

# Define C function signatures for Python-C++ interface
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

# Initialize face recognition engine
DB_PATH = os.path.join(DB_DIR, "faces.db")
engine = lib.engine_create(MODEL_DIR.encode(), DB_PATH.encode())
if not engine:
    print("❌ Failed to create recognition engine")
    sys.exit(1)
print("✅ Face recognizer loaded (MobileFaceNet ncnn)")


def add_face(name, face_bgr):
    """
    Add face to database.
    Input: BGR numpy array (from camera/OpenCV)
    MobileFaceNet C++ already expects BGR - no conversion needed!
    """
    f = np.ascontiguousarray(face_bgr)
    h, w = f.shape[:2]
    # Pass pointer to C++ via ctypes
    return lib.engine_add_face(engine, name.encode(), f.ctypes.data, w, h)


def verify_face(face_bgr):
    """
    Verify face against database.
    Input: BGR numpy array (from camera/OpenCV)
    Returns: (name, similarity_score)
    MobileFaceNet C++ already expects BGR - no conversion needed!
    """
    f = np.ascontiguousarray(face_bgr)
    h, w = f.shape[:2]
    buf = ctypes.create_string_buffer(256)
    # Call C++ function via ctypes
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
# CAMERA SETUP (Full FOV via GPU ISP)
# ══════════════════════════════════════════════
print("📷 Initializing Camera...")
picam2 = Picamera2()

# Critical: Use raw stream to force full field of view
# The GPU ISP processes the raw sensor data and outputs the main stream
# Main output: 640x480 BGR888 (OpenCV native format - correct colors!)
# Raw: Full sensor resolution ensures no cropping
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"},  # BGR for OpenCV
    raw={"size": picam2.sensor_resolution},          # Raw ensures full FOV
    buffer_count=4,                                  # Number of buffers for performance
)
picam2.configure(camera_config)
picam2.start()
time.sleep(1.5)

print(f"✅ Camera initialized: 640×480 BGR, full FOV via GPU ISP")
print(f"   Sensor resolution: {picam2.sensor_resolution}")

# ══════════════════════════════════════════════
# SMART FACE CACHING SYSTEM
# ══════════════════════════════════════════════
"""
Face cache structure: {bbox_key: (name, score, timestamp, verified)}

bbox_key: (x, y, w, h) - Bounding box position and size
name: Recognition result ("Unknown" or person's name)
score: Similarity score from MobileFaceNet
timestamp: When this face was last verified
verified: Boolean flag indicating if verification was performed

Strategy:
- RECOGNIZED faces: Cached permanently while in view (no re-verification)
- UNKNOWN faces: Re-verified every 2 seconds (might be added to DB later)
"""

face_cache = {}
UNKNOWN_REVERIFY_INTERVAL = 2.0  # Seconds before re-verifying unknown faces
BBOX_OVERLAP_THRESHOLD = 0.5      # 50% overlap to consider same face


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Used to track if a face is the same across frames.

    Args:
        bbox1, bbox2: (x, y, w, h) tuples

    Returns:
        float: IoU score (0.0 to 1.0)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def get_cached_result(bbox):
    """
    Smart caching strategy for face recognition.

    Returns: (name, score, should_verify)
    - name: Cached recognition result (or None if not cached)
    - score: Cached similarity score (or None)
    - should_verify: Boolean indicating if verification should be performed

    Logic:
    1. Search cache for matching face (by position overlap)
    2. If RECOGNIZED: Return cached result, don't verify again
    3. If UNKNOWN and <2s passed: Return cached "Unknown", don't verify
    4. If UNKNOWN and ≥2s passed: Return cached, but flag for re-verification
    5. If not in cache: Flag for verification
    """
    current_time = time.time()

    # Clean up stale cache entries (faces that left the view)
    to_delete = []
    for cached_bbox, (name, score, timestamp, verified) in list(face_cache.items()):
        # If cache is older than 3 seconds, face probably left the frame
        if current_time - timestamp > 3.0:
            to_delete.append(cached_bbox)

    for bbox_key in to_delete:
        del face_cache[bbox_key]

    # Search for matching face in cache
    for cached_bbox, (name, score, timestamp, verified) in face_cache.items():
        # Calculate overlap with cached face
        iou = calculate_iou(bbox, cached_bbox)

        # If significant overlap (>50%), consider it the same face
        if iou > BBOX_OVERLAP_THRESHOLD:
            # Found matching face in cache

            if name != "Unknown":
                # ★ RECOGNIZED FACE: Use cache indefinitely (no re-verification)
                # Update position but keep same recognition result
                face_cache[bbox] = (name, score, current_time, True)
                if cached_bbox != bbox:
                    del face_cache[cached_bbox]
                return name, score, False  # Don't verify again

            else:
                # ★ UNKNOWN FACE: Check if it's time to re-verify
                time_since_verify = current_time - timestamp

                if time_since_verify >= UNKNOWN_REVERIFY_INTERVAL:
                    # Time to re-verify (might have been added to database)
                    return name, score, True
                else:
                    # Still within 2-second window, use cached "Unknown"
                    face_cache[bbox] = (name, score, timestamp, verified)
                    if cached_bbox != bbox:
                        del face_cache[cached_bbox]
                    return name, score, False

    # New face not in cache - needs verification
    return None, None, True


# ══════════════════════════════════════════════
# COMMAND HANDLER
# ══════════════════════════════════════════════
mode = "VERIFY"  # Default mode: continuous verification
add_name = ""


def handle_command(cmd):
    """Handle keyboard commands"""
    global mode, add_name

    if cmd == 'd':
        mode = "DETECT"
        print("🔍 Detection mode (no recognition)")

    elif cmd == 'v':
        mode = "VERIFY"
        print("🔐 Verify mode (continuous recognition)")

    elif cmd == 'a':
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
            result = delete_face(n)
            print("✅ Deleted" if result == 0 else "❌ Not found")


# ══════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("🎮 CONTROLS (press in window OR type in terminal):")
print("   d - Detection mode (no recognition)")
print("   v - Verify mode (continuous recognition)")
print("   a - Add face to database")
print("   l - List database")
print("   x - Delete face")
print("   q - Quit")
print("="*60 + "\n")

verify_result = ("", 0.0)
verify_timer = 0
fps_smooth = 0.0
t_prev = time.time()

try:
    while True:
        # ── Capture Frame ──
        # picamera2 outputs BGR888 directly (GPU ISP configured for BGR)
        frame_bgr = picam2.capture_array("main")

        # ── Convert to RGB for MediaPipe and Display ──
        # Single conversion: BGR → RGB (used by both MediaPipe and monitor display)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── Detect Faces with MediaPipe BlazeFace ──
        # MediaPipe expects RGB
        results = blazeface.process(frame_rgb)

        faces = []
        if results.detections:
            h, w = frame_rgb.shape[:2]
            for detection in results.detections:
                # Extract bounding box (MediaPipe uses relative coordinates)
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Ensure bounds are valid
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_min + width)
                y_max = min(h, y_min + height)

                if x_max > x_min and y_max > y_min:
                    faces.append((x_min, y_min, x_max - x_min, y_max - y_min,
                                float(detection.score[0])))

        # ── Process Each Face Independently ──
        for (x1, y1, w, h, score) in faces:
            x2, y2 = x1 + w, y1 + h

            # Extract face crop from BGR frame (MobileFaceNet needs BGR)
            face_crop_bgr = frame_bgr[y1:y2, x1:x2]

            if face_crop_bgr.shape[0] < 20 or face_crop_bgr.shape[1] < 20:
                continue

            color = (255, 0, 0)  # Red in RGB (default)
            label = f"{score:.2f}"

            # ── ADD MODE ──
            if mode == "ADD" and add_name:
                add_face(add_name, face_crop_bgr.copy())
                print(f"✅ Added: {add_name}")
                add_name = ""
                mode = "VERIFY"
                face_cache.clear()  # Clear cache after adding new face

            # ── VERIFY MODE ──
            elif mode == "VERIFY":
                # Check smart cache using bounding box position
                # Each face tracked independently by its (x, y, w, h) position
                cached_name, cached_score, should_verify = get_cached_result((x1, y1, w, h))

                if should_verify:
                    # Perform verification (calls C++ MobileFaceNet)
                    name, sim = verify_face(face_crop_bgr.copy())
                    # Store in cache with THIS face's unique position
                    face_cache[(x1, y1, w, h)] = (name, sim, time.time(), True)
                    verify_result = (name, sim)
                    verify_timer = 30
                else:
                    # Use cached result (no C++ call)
                    name, sim = cached_name, cached_score
                    verify_result = (name, sim)
                    verify_timer = 30

                # Set color and label based on recognition result
                if name != "Unknown":
                    color = (0, 255, 0)  # Green for recognized (RGB)
                    label = f"{name} ({sim:.2f})"
                else:
                    color = (255, 0, 0)  # Red for unknown (RGB)
                    label = f"Unknown ({sim:.2f})"

            # ── Draw on RGB frame for correct monitor colors ──
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_rgb, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # ── Calculate FPS ──
        now = time.time()
        fps_smooth = 0.9 * fps_smooth + 0.1 / max(now - t_prev, 0.001)
        t_prev = now

        # ── Draw Status Info on RGB frame ──
        cv2.putText(frame_rgb, f"FPS: {fps_smooth:.1f}  Faces: {len(faces)}", (8, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Yellow in RGB
        cv2.putText(frame_rgb, f"Mode: {mode}", (8, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)  # Cyan in RGB

        if verify_timer > 0:
            n, s = verify_result
            c = (0, 255, 0) if n != "Unknown" else (255, 0, 0)  # RGB colors
            cv2.putText(frame_rgb, f"{n} ({s:.3f})", (8, 68),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
            verify_timer -= 1

        if mode == "ADD" and add_name:
            cv2.putText(frame_rgb, f"Adding: {add_name}", (8, frame_rgb.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)  # Orange in RGB

        # ── Display Frame with CORRECT COLORS ──
        # OpenCV imshow with RGB frame shows correct colors on monitor!
        cv2.imshow("Face Recognition - RPi4", frame_rgb)

        # ── Handle Keyboard Input ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in (ord('d'), ord('v'), ord('a'), ord('l'), ord('x')):
            handle_command(chr(key))

        # Terminal input (non-blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline().strip()
            if cmd == 'q':
                break
            elif cmd in ('d', 'v', 'a', 'l', 'x'):
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
    cv2.destroyAllWindows()

    print("✅ Cleanup complete")
    print("👋 Bye!")
