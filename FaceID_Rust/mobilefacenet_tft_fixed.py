#!/usr/bin/env python3
"""
Raspberry Pi 4 Face Recognition System with GPIO TFT Display
Fixed: Removed autofocus for IMX219 (fixed-focus camera)
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

# Picamera2 imports
try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ Error: Picamera2 not found!")
    sys.exit(1)

# MediaPipe for BlazeFace
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
CAPTURE_WIDTH = 1640
CAPTURE_HEIGHT = 1232
MIN_DETECTION_CONFIDENCE = 0.5

# UI Colors
COLOR_BG = (20, 20, 30)
COLOR_TEXT = (255, 255, 255)
COLOR_BTN_ADD = (50, 200, 50)
COLOR_BTN_VERIFY = (50, 150, 255)
COLOR_BOX = (0, 255, 0)

# Display Setup
class TFTDisplay:
    """ILI9341 TFT Display Manager"""
    
    def __init__(self):
        cs_pin = digitalio.DigitalInOut(board.CE0)
        dc_pin = digitalio.DigitalInOut(board.D25)
        reset_pin = digitalio.DigitalInOut(board.D27)
        
        spi = board.SPI()
        self.disp = ili9341.ILI9341(
            spi, cs=cs_pin, dc=dc_pin, rst=reset_pin,
            baudrate=BAUDRATE, width=240, height=320, rotation=90
        )
        
        self.image = Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        self.draw = ImageDraw.Draw(self.image)
        
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
        
        print("✅ TFT Display initialized: 320×240")
    
    def draw_ui(self, camera_frame=None, status="Ready", fps=0, num_faces=0):
        """Draw UI with camera preview"""
        self.draw.rectangle((0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT), fill=COLOR_BG)
        
        if camera_frame is not None:
            if len(camera_frame.shape) == 3 and camera_frame.shape[2] == 3:
                camera_frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            else:
                camera_frame_rgb = camera_frame
            
            preview_img = Image.fromarray(camera_frame_rgb)
            preview_img = preview_img.resize((240, 180), Image.BILINEAR)
            self.image.paste(preview_img, (40, 5))
            self.draw.rectangle((40, 5, 280, 185), outline=(100, 100, 100), width=1)
        
        # Status text
        self.draw.text((5, 5), f"FPS: {fps:.1f}", font=self.font_small, fill=COLOR_TEXT)
        self.draw.text((5, 190), f"Faces: {num_faces} | {status}", font=self.font_small, fill=COLOR_TEXT)
        
        # Instructions
        self.draw.text((5, 220), "Press 'a' to ADD, 'v' to VERIFY", font=self.font_small, fill=COLOR_TEXT)
        
        self.disp.image(self.image)
    
    def draw_face_boxes(self, camera_frame, faces):
        """Draw bounding boxes"""
        if camera_frame is None or len(faces) == 0:
            return camera_frame
        
        frame_copy = camera_frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), COLOR_BOX, 2)
        return frame_copy

# Camera with Fixed Focus Support
class OptimizedPiCamera:
    """Camera for IMX219 (fixed-focus)"""
    
    def __init__(self):
        self.picam2 = None
        self.preview_config = None
        self.capture_config = None
        
    def start(self):
        """Initialize camera"""
        try:
            self.picam2 = Picamera2()
            
            # Preview config - NO autofocus for IMX219!
            self.preview_config = self.picam2.create_preview_configuration(
                main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"},
                raw={"size": (1640, 1232)}  # Mode 2: Full FOV
            )
            
            # Capture config
            self.capture_config = self.picam2.create_still_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
            )
            
            self.picam2.configure(self.preview_config)
            self.picam2.start()
            
            time.sleep(2)
            print("✅ Camera initialized: IMX219 @ 320×240 (fixed-focus)")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def read_preview(self):
        """Capture preview frame"""
        try:
            frame = self.picam2.capture_array()
            return True, frame
        except:
            return False, None
    
    def capture_high_res(self):
        """Capture high-res frame"""
        try:
            self.picam2.stop()
            self.picam2.configure(self.capture_config)
            self.picam2.start()
            time.sleep(0.5)
            
            frame = self.picam2.capture_array()
            
            self.picam2.stop()
            self.picam2.configure(self.preview_config)
            self.picam2.start()
            time.sleep(0.3)
            
            return frame
        except Exception as e:
            print(f"❌ High-res capture failed: {e}")
            return None
    
    def stop(self):
        if self.picam2:
            self.picam2.stop()

# BlazeFace Detector
class BlazeFaceDetector:
    """MediaPipe BlazeFace"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE
        )
    
    def detect(self, frame):
        """Detect faces, returns [(x,y,w,h), ...]"""
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

# Database
def load_database():
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            print("⚠️ Database corrupted")
    return {}

def save_database(db):
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(db, f)

# Face Capture
def capture_face_embedding(camera, detector, display):
    """Capture face and extract embedding"""
    print("\n📸 Position your face... capturing in 3 seconds")
    
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
            status = f"CAPTURING IN {countdown}..." if countdown > 0 else "PROCESSING..."
        else:
            status = "No face detected!"
            start_time = time.time()
            countdown = 3
        
        display_frame = display.draw_face_boxes(frame, faces)
        display.draw_ui(display_frame, status=status, fps=0, num_faces=len(faces))
        
        if len(faces) > 0 and countdown <= 0:
            print("📸 Capturing high-res...")
            
            hires_frame = camera.capture_high_res()
            if hires_frame is None:
                print("❌ Capture failed!")
                return None
            
            hires_faces = detector.detect(hires_frame)
            if len(hires_faces) == 0:
                print("⚠️ Face lost!")
                return None
            
            (x, y, w, h) = max(hires_faces, key=lambda f: f[2] * f[3])
            margin = int(max(w, h) * 0.2)
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(CAPTURE_WIDTH, x + w + margin), min(CAPTURE_HEIGHT, y + h + margin)
            
            face_crop = hires_frame[y1:y2, x1:x2]
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            
            temp_file = "temp_capture.jpg"
            cv2.imwrite(temp_file, face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print("🦀 Processing with Rust...")
            embedding = get_face_embedding(temp_file)
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            if embedding is not None:
                print("✅ Embedding extracted!")
                return embedding
            else:
                print("❌ Rust engine failed")
                return None
    
    return None

# Main Application
def main():
    print("="*60)
    print("  Raspberry Pi 4 Face Recognition System")
    print("  TFT Display Edition (Fixed-Focus)")
    print("="*60)
    
    display = TFTDisplay()
    camera = OptimizedPiCamera()
    
    if not camera.start():
        sys.exit(1)
    
    detector = BlazeFaceDetector()
    db = load_database()
    
    print(f"✅ System ready! Database: {len(db)} faces")
    print("\nControls:")
    print("  Press 'a' to ADD face")
    print("  Press 'v' to VERIFY face")
    print("  Press 'q' to QUIT")
    
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
            display_frame = display.draw_face_boxes(frame, faces)
            display.draw_ui(display_frame, status="Ready", fps=fps, num_faces=len(faces))
            
            cmd = non_blocking_input()
            
            if cmd == 'a':
                print("\n[ADD FACE]")
                embedding = capture_face_embedding(camera, detector, display)
                if embedding is not None:
                    name = input("Enter name: ").strip()
                    if name:
                        db[name] = embedding
                        save_database(db)
                        print(f"✨ Registered: {name}")
            
            elif cmd == 'v':
                print("\n[VERIFY FACE]")
                if len(db) == 0:
                    print("❌ Database empty!")
                    continue
                
                embedding = capture_face_embedding(camera, detector, display)
                if embedding is not None:
                    best_score = -1.0
                    best_name = "Unknown"
                    
                    for name, stored in db.items():
                        score = compute_similarity(embedding, stored)
                        print(f"  {name}: {score:.3f}")
                        if score > best_score:
                            best_score = score
                            best_name = name
                    
                    if best_score > SIMILARITY_THRESHOLD:
                        print(f"🔓 ACCESS GRANTED: {best_name} ({best_score:.3f})")
                    else:
                        print(f"🔒 ACCESS DENIED (Best: {best_name}, {best_score:.3f})")
            
            elif cmd == 'q':
                print("Exiting...")
                break
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    finally:
        camera.stop()
        detector.close()
        save_database(db)
        
        display.draw.rectangle((0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT), fill=(0, 0, 0))
        display.disp.image(display.image)
        
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
