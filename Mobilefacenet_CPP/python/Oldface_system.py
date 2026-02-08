#!/usr/bin/env python3
"""
Face Recognition System for Raspberry Pi 4
- Detection: OpenCV DNN (Python, fast on ARM)
- Recognition: MobileFaceNet via ncnn C++ shared library (128-dim embedding)
- Camera: picamera2 BGR888, full FOV via GPU ISP
"""

import ctypes
import numpy as np
import cv2
import time
import sys
import os
import select
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DB_DIR = os.path.join(PROJECT_DIR, "database")
os.makedirs(DB_DIR, exist_ok=True)

# ══════════════════════════════════════════════
# Face Detector — pure Python, OpenCV DNN
# ══════════════════════════════════════════════
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(PROTOTXT):
    print("📥 Downloading deploy.prototxt...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        PROTOTXT)

if not os.path.exists(CAFFEMODEL):
    print("📥 Downloading caffemodel (10MB)...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        CAFFEMODEL)

detector = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
print("✅ Face detector loaded (OpenCV DNN)")

CONF_THRESHOLD = 0.5


def detect_faces(frame):
    """Detect faces. Input/output in BGR. Returns list of (x1,y1,x2,y2,score)."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    dets = detector.forward()
    faces = []
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf < CONF_THRESHOLD:
            continue
        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            faces.append((x1, y1, x2, y2, float(conf)))
    return faces


# ══════════════════════════════════════════════
# Face Recognizer — C++ ncnn MobileFaceNet only
# ══════════════════════════════════════════════
LIB_PATH = os.path.join(PROJECT_DIR, "build", "libface_engine.so")
if not os.path.exists(LIB_PATH):
    print(f"❌ Cannot find {LIB_PATH}")
    print("   Build: cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4")
    sys.exit(1)

lib = ctypes.CDLL(LIB_PATH)

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


def add_face(name, face_bgr):
    f = np.ascontiguousarray(face_bgr)
    h, w = f.shape[:2]
    return lib.engine_add_face(engine, name.encode(), f.ctypes.data, w, h)


def verify_face(face_bgr):
    f = np.ascontiguousarray(face_bgr)
    h, w = f.shape[:2]
    buf = ctypes.create_string_buffer(256)
    score = lib.engine_verify(engine, f.ctypes.data, w, h, buf, 256)
    return buf.value.decode(), float(score)


def delete_face(name):
    return lib.engine_delete_face(engine, name.encode())


def list_faces():
    buf = ctypes.create_string_buffer(4096)
    count = lib.engine_list_faces(engine, buf, 4096)
    if count == 0:
        return []
    return [n for n in buf.value.decode().split('\n') if n]


# ══════════════════════════════════════════════
# Camera
# ══════════════════════════════════════════════
from picamera2 import Picamera2

cam = Picamera2()
cam.configure(cam.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"},
    raw={"size": (1640, 1232)},  # full FOV
    buffer_count=4,
))
cam.start()
time.sleep(1.5)
print("📷 Camera: 640x480 BGR, full FOV (raw 1640x1232)")

# ══════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════
mode = "DETECT"
add_name = ""
verify_result = ("", 0.0)
verify_timer = 0
t_prev = time.time()
fps_smooth = 0.0

print("\n🎮 Controls (press in camera window OR type in terminal):")
print("   d - Detection mode")
print("   v - Verify mode (continuous)")
print("   a - Add face")
print("   l - List database")
print("   x - Delete face")
print("   q - Quit\n")


def handle_command(cmd):
    global mode, add_name
    if cmd == 'd':
        mode = "DETECT"
        print("🔍 Detection mode")
    elif cmd == 'v':
        mode = "VERIFY"
        print("🔐 Verify mode")
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
            print("✅ Deleted" if delete_face(n) == 0 else "❌ Not found")


while True:
    frame = cam.capture_array("main")  # BGR888 directly

    # ── Detect ──
    faces = detect_faces(frame)

    # ── Process each face ──
    for (x1, y1, x2, y2, score) in faces:
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            continue

        color = (0, 255, 0)
        label = f"{score:.2f}"

        if mode == "ADD" and add_name:
            add_face(add_name, face_crop.copy())
            print(f"✅ Added: {add_name}")
            add_name = ""
            mode = "DETECT"

        elif mode == "VERIFY":
            name, sim = verify_face(face_crop.copy())
            verify_result = (name, sim)
            verify_timer = 30
            if name != "Unknown":
                color = (0, 255, 0)
                label = f"{name} ({sim:.2f})"
            else:
                color = (0, 0, 255)
                label = f"Unknown ({sim:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ── FPS ──
    now = time.time()
    fps_smooth = 0.9 * fps_smooth + 0.1 / max(now - t_prev, 0.001)
    t_prev = now

    cv2.putText(frame, f"FPS: {fps_smooth:.1f}  Faces: {len(faces)}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Mode: {mode}", (8, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    if verify_timer > 0:
        n, s = verify_result
        c = (0, 255, 0) if n != "Unknown" else (0, 0, 255)
        cv2.putText(frame, f"{n} ({s:.3f})", (8, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        verify_timer -= 1

    if mode == "ADD" and add_name:
        cv2.putText(frame, f"Adding: {add_name}", (8, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    cv2.imshow("Face Recognition - RPi4", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in (ord('d'), ord('v'), ord('a'), ord('l'), ord('x')):
        handle_command(chr(key))

    # Terminal input
    if select.select([sys.stdin], [], [], 0)[0]:
        cmd = sys.stdin.readline().strip()
        if cmd == 'q':
            break
        elif cmd in ('d', 'v', 'a', 'l', 'x'):
            handle_command(cmd)

cam.stop()
lib.engine_destroy(engine)
cv2.destroyAllWindows()
print("👋 Bye!")
