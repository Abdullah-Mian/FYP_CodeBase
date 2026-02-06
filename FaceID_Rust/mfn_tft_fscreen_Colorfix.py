
# Configuration
DISPLAY_NEEDS_BGR = True  # Set to True if colors still wrong
#!/usr/bin/env python3
"""
Raspberry Pi 4 Face Recognition System - Color Fixed
Fixed: RGB/BGR channel swap for ILI9341 display
"""

import time
import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import ili9341
import cv2
import numpy as np
import os
import pickle
import sys

# Picamera2
try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ Error: Picamera2 not found!")
    sys.exit(1)

# MediaPipe
try:
    import mediapipe as mp
except ImportError:
    print("❌ Error: MediaPipe not found!")
    sys.exit(1)

# Rust wrapper
try:
    from rust_wrapper import get_face_embedding, compute_similarity
except ImportError:
    print("❌ Error: rust_wrapper.py not found!")
    sys.exit(1)

# Configuration
SIMILARITY_THRESHOLD = 0.4
DATABASE_FILE = 'faces_db.pkl'

# Display Configuration
DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240
BAUDRATE = 64000000

# Camera Configuration
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 240
CAPTURE_WIDTH = 3280
CAPTURE_HEIGHT = 2464
MIN_DETECTION_CONFIDENCE = 0.5

# UI Colors (RGB format)
COLOR_BOX = (0, 255, 0)      # Green
COLOR_TEXT = (255, 255, 255)  # White

# ========================================
# Display Setup
# ========================================
class TFTDisplay:
    """ILI9341 TFT Display - Full Screen Mode with Color Fix"""

    def __init__(self):
        cs_pin = digitalio.DigitalInOut(board.CE0)
        dc_pin = digitalio.DigitalInOut(board.D25)
        reset_pin = digitalio.DigitalInOut(board.D27)

        spi = board.SPI()
        self.disp = ili9341.ILI9341(
            spi, cs=cs_pin, dc=dc_pin, rst=reset_pin,
            baudrate=BAUDRATE, width=240, height=320, rotation=90
        )

        try:
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            self.font_small = ImageFont.load_default()

        print("✅ TFT Display initialized: 320×240 (Full Screen)")

    def show_frame(self, frame, faces=[], status=None, fps=0):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if status or fps > 0:
            text = f"FPS:{fps:.0f}"
            if status:
                text += f" | {status}"
            cv2.putText(frame_bgr, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Try BGR mode if RGB doesn't work
        if DISPLAY_NEEDS_BGR:
            pil_img = Image.fromarray(frame_bgr, mode='RGB')  # Trick: send BGR as "RGB"
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
        
        self.disp.image(pil_img)

    def clear(self):
        """Clear display to black"""
        image = Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0))
        self.disp.image(image)

# ========================================
# Camera
# ========================================
class OptimizedPiCamera:
    """Camera with FULL resolution capture"""

    def __init__(self):
        self.picam2 = None
        self.preview_config = None
        self.capture_config = None

    def start(self):
        try:
            self.picam2 = Picamera2()

            self.preview_config = self.picam2.create_preview_configuration(
                main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"},
                raw={"size": (1640, 1232)}
            )

            self.capture_config = self.picam2.create_still_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
            )

            self.picam2.configure(self.preview_config)
            self.picam2.start()
            time.sleep(2)

            print(f"✅ Camera initialized: Preview {PREVIEW_WIDTH}×{PREVIEW_HEIGHT}, Capture {CAPTURE_WIDTH}×{CAPTURE_HEIGHT}")
            return True

        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False

    def read_preview(self):
        try:
            frame = self.picam2.capture_array()
            return True, frame
        except:
            return False, None

    def capture_high_res(self):
        try:
            print(f"  📷 Switching to {CAPTURE_WIDTH}×{CAPTURE_HEIGHT}...")

            self.picam2.stop()
            self.picam2.configure(self.capture_config)
            self.picam2.start()
            time.sleep(0.8)

            frame = self.picam2.capture_array()

            self.picam2.stop()
            self.picam2.configure(self.preview_config)
            self.picam2.start()
            time.sleep(0.5)

            print(f"  ✅ Captured {frame.shape[1]}×{frame.shape[0]} image")
            return frame

        except Exception as e:
            print(f"  ❌ High-res capture failed: {e}")
            return None

    def stop(self):
        if self.picam2:
            self.picam2.stop()

# ========================================
# BlazeFace Detector
# ========================================
class BlazeFaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE
        )

    def detect(self, frame):
        results = self.detector.process(frame)
        if not results.detections:
            return []

        h, w, _ = frame.shape
        faces = []
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            faces.append((max(0, x), max(0, y), width, height))
        return faces

    def close(self):
        self.detector.close()

# ========================================
# Database
# ========================================
def load_database():
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            print("⚠️  Database corrupted")
    return {}

def save_database(db):
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(db, f)

# ========================================
# Face Capture
# ========================================
def capture_face_embedding(camera, detector, display):
    print("\n📸 Position your face... Auto-capture in 3 seconds")

    countdown = 3
    start_time = time.time()

    while countdown > 0:
        ret, frame = camera.read_preview()
        if not ret:
            continue

        faces = detector.detect(frame)

        if len(faces) > 0:
            elapsed = time.time() - start_time
            countdown = 3 - int(elapsed)
            status = f"CAPTURE IN {countdown}..." if countdown > 0 else "PROCESSING..."
        else:
            status = "NO FACE"
            start_time = time.time()
            countdown = 3

        display.show_frame(frame, faces=faces, status=status, fps=0)

        if len(faces) > 0 and countdown <= 0:
            hires_frame = camera.capture_high_res()
            if hires_frame is None:
                print("  ❌ Capture failed!")
                return None

            hires_faces = detector.detect(hires_frame)
            if len(hires_faces) == 0:
                print("  ⚠️  Face lost in high-res capture!")
                return None

            (x, y, w, h) = max(hires_faces, key=lambda f: f[2] * f[3])
            margin = int(max(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(CAPTURE_WIDTH, x + w + margin)
            y2 = min(CAPTURE_HEIGHT, y + h + margin)

            face_crop = hires_frame[y1:y2, x1:x2]
            print(f"  📐 Face crop size: {face_crop.shape[1]}×{face_crop.shape[0]}")

            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            temp_file = "temp_capture.jpg"
            cv2.imwrite(temp_file, face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            print("  🦀 Processing with Rust engine...")
            embedding = get_face_embedding(temp_file)

            if os.path.exists(temp_file):
                os.remove(temp_file)

            if embedding is not None:
                print("  ✅ Embedding extracted!")
                return embedding
            else:
                print("  ❌ Rust engine failed")
                return None

    return None

# ========================================
# Main Application
# ========================================
def main():
    print("="*60)
    print("  Raspberry Pi 4 Face Recognition System")
    print("  Full Screen TFT Edition (Color Fixed)")
    print(f"  Camera: {CAPTURE_WIDTH}×{CAPTURE_HEIGHT} capture")
    print(f"  Display: {DISPLAY_WIDTH}×{DISPLAY_HEIGHT} full screen")
    print("="*60)

    display = TFTDisplay()
    camera = OptimizedPiCamera()

    if not camera.start():
        sys.exit(1)

    detector = BlazeFaceDetector()
    db = load_database()

    print(f"✅ System ready! Database: {len(db)} faces")
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  'a' - ADD new face")
    print("  'v' - VERIFY face")
    print("  'l' - LIST faces")
    print("  'q' - QUIT")
    print("="*60 + "\n")

    frame_count = 0
    fps_start = time.time()
    fps = 0

    import select

    def non_blocking_input():
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip()
        return None

    try:
        while True:
            ret, frame = camera.read_preview()
            if not ret:
                continue

            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()

            faces = detector.detect(frame)
            display.show_frame(frame, faces=faces, status="Ready", fps=fps)

            cmd = non_blocking_input()

            if cmd == 'a':
                print("\n" + "="*60)
                print("[ ADD FACE MODE ]")
                print("="*60)

                embedding = capture_face_embedding(camera, detector, display)
                if embedding is not None:
                    name = input("\n👤 Enter name: ").strip()
                    if name:
                        db[name] = embedding
                        save_database(db)
                        print(f"✨ Successfully registered: {name}")
                        print(f"📊 Total faces in database: {len(db)}")
                print("="*60 + "\n")

            elif cmd == 'v':
                print("\n" + "="*60)
                print("[ VERIFY FACE MODE ]")
                print("="*60)

                if len(db) == 0:
                    print("❌ Database is empty!")
                    print("="*60 + "\n")
                    continue

                embedding = capture_face_embedding(camera, detector, display)
                if embedding is not None:
                    print("\n🔍 Comparing with database:")

                    best_score = -1.0
                    best_name = "Unknown"

                    for name, stored in db.items():
                        score = compute_similarity(embedding, stored)
                        print(f"   {name:20s} : {score:.3f}")
                        if score > best_score:
                            best_score = score
                            best_name = name

                    print("\n" + "-"*60)
                    if best_score > SIMILARITY_THRESHOLD:
                        print(f"✅ ACCESS GRANTED")
                        print(f"   Identity: {best_name}")
                        print(f"   Confidence: {best_score:.3f}")
                    else:
                        print(f"❌ ACCESS DENIED")
                        print(f"   Best match: {best_name}")
                        print(f"   Score: {best_score:.3f}")
                    print("-"*60)
                print("="*60 + "\n")

            elif cmd == 'l':
                print("\n" + "="*60)
                print("[ REGISTERED FACES ]")
                print("="*60)
                if len(db) == 0:
                    print("  (empty)")
                else:
                    for i, name in enumerate(db.keys(), 1):
                        print(f"  {i:2d}. {name}")
                print(f"\nTotal: {len(db)} faces")
                print("="*60 + "\n")

            elif cmd == 'q':
                print("\n💾 Saving database...")
                save_database(db)
                print("👋 Goodbye!")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    finally:
        camera.stop()
        detector.close()
        save_database(db)
        display.clear()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
