#!/usr/bin/env python3
"""
Raspberry Pi 4 Face Recognition System with GPIO TFT Display
Integrated: Picamera2 + BlazeFace + MobileFaceNet + ILI9341 TFT + Touch

Hardware:
- IMX219 Camera Module v2 (8MP)
- ILI9341 3.2" TFT Display (320×240, SPI)
- XPT2046 Touch Controller (optional)
- Raspberry Pi 4 Model B

Author: FYP - Face Recognition System
"""

import time
import board
import digitalio
import spidev
import RPi.GPIO as GPIO
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
    from libcamera import controls
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

# ========================================
# Configuration
# ========================================
SIMILARITY_THRESHOLD = 0.4
DATABASE_FILE = 'faces_db.pkl'

# Display Configuration
DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240
BAUDRATE = 64000000

# Camera Configuration (Optimized for speed + full FOV)
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 240
CAPTURE_WIDTH = 1640   # High-res for embedding extraction
CAPTURE_HEIGHT = 1232
MIN_DETECTION_CONFIDENCE = 0.5

# Touch Configuration (Optional)
USE_TOUCH = False  # Set to True if you have touch screen working
TOUCH_CS_PIN = 7
TOUCH_IRQ_PIN = 24
X_MIN, X_MAX = 300, 3800
Y_MIN, Y_MAX = 200, 3700

# UI Colors
COLOR_BG = (20, 20, 30)
COLOR_TEXT = (255, 255, 255)
COLOR_BTN_ADD = (50, 200, 50)
COLOR_BTN_VERIFY = (50, 150, 255)
COLOR_BTN_PRESSED = (255, 255, 50)
COLOR_BOX = (0, 255, 0)

# ========================================
# Display Setup
# ========================================
class TFTDisplay:
    """ILI9341 TFT Display Manager"""
    
    def __init__(self):
        # Display pins
        cs_pin = digitalio.DigitalInOut(board.CE0)
        dc_pin = digitalio.DigitalInOut(board.D25)
        reset_pin = digitalio.DigitalInOut(board.D27)
        
        # Initialize display
        spi = board.SPI()
        self.disp = ili9341.ILI9341(
            spi,
            cs=cs_pin,
            dc=dc_pin,
            rst=reset_pin,
            baudrate=BAUDRATE,
            width=240,
            height=320,
            rotation=90  # Landscape mode
        )
        
        # Create drawing canvas
        self.image = Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        self.draw = ImageDraw.Draw(self.image)
        
        # Load font
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
        
        # Button definitions
        self.btn_add = (10, 190, 150, 230)      # (x1, y1, x2, y2)
        self.btn_verify = (170, 190, 310, 230)
        
        print("✅ TFT Display initialized: 320×240")
    
    def draw_ui(self, camera_frame=None, status="Ready", fps=0, num_faces=0, mode=None):
        """Draw complete UI with camera preview and controls"""
        
        # 1. Background
        self.draw.rectangle((0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT), fill=COLOR_BG)
        
        # 2. Camera preview area (240×180)
        if camera_frame is not None:
            # Convert BGR to RGB if needed
            if len(camera_frame.shape) == 3 and camera_frame.shape[2] == 3:
                camera_frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            else:
                camera_frame_rgb = camera_frame
            
            # Resize to preview area
            preview_img = Image.fromarray(camera_frame_rgb)
            preview_img = preview_img.resize((240, 180), Image.BILINEAR)
            
            # Paste onto canvas
            self.image.paste(preview_img, (40, 5))
            
            # Draw border around preview
            self.draw.rectangle((40, 5, 280, 185), outline=(100, 100, 100), width=1)
        
        # 3. Buttons
        # Add button
        btn_color_add = COLOR_BTN_PRESSED if mode == 'add' else COLOR_BTN_ADD
        self.draw.rectangle(self.btn_add, fill=btn_color_add)
        self.draw.text((35, 203), "ADD FACE", font=self.font, fill=COLOR_TEXT)
        
        # Verify button
        btn_color_verify = COLOR_BTN_PRESSED if mode == 'verify' else COLOR_BTN_VERIFY
        self.draw.rectangle(self.btn_verify, fill=btn_color_verify)
        self.draw.text((195, 203), "VERIFY", font=self.font, fill=COLOR_TEXT)
        
        # 4. Status bar (top)
        self.draw.text((5, 5), f"FPS: {fps:.1f}", font=self.font_small, fill=COLOR_TEXT)
        self.draw.text((5, 170), f"Faces: {num_faces} | {status}", font=self.font_small, fill=COLOR_TEXT)
        
        # 5. Push to display
        self.disp.image(self.image)
    
    def draw_face_boxes(self, camera_frame, faces):
        """Draw bounding boxes on camera frame"""
        if camera_frame is None or len(faces) == 0:
            return camera_frame
        
        # Scale factors (camera is 320×240, preview area is 240×180)
        scale_x = 240 / PREVIEW_WIDTH
        scale_y = 180 / PREVIEW_HEIGHT
        
        frame_copy = camera_frame.copy()
        for (x, y, w, h) in faces:
            # Draw on camera frame
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), COLOR_BOX, 2)
        
        return frame_copy

# ========================================
# Touch Screen (Optional)
# ========================================
class TouchScreen:
    """XPT2046 Touch Controller"""
    
    def __init__(self):
        if not USE_TOUCH:
            return
        
        self.touch_spi = spidev.SpiDev()
        self.touch_spi.open(0, 1)  # Bus 0, Device 1 (CE1)
        self.touch_spi.max_speed_hz = 1000000
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TOUCH_IRQ_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        self.last_touch_time = 0
        print("✅ Touch screen initialized")
    
    def read_touch(self):
        """Returns (x, y) or None"""
        if not USE_TOUCH:
            return None
        
        if GPIO.input(TOUCH_IRQ_PIN):
            return None  # No touch
        
        try:
            # Read X
            resp_x = self.touch_spi.xfer2([0xD0, 0, 0])
            x_raw = ((resp_x[1] << 8) | resp_x[2]) >> 3
            
            # Read Y
            resp_y = self.touch_spi.xfer2([0x90, 0, 0])
            y_raw = ((resp_y[1] << 8) | resp_y[2]) >> 3
            
            # Map to screen coordinates
            sx = int(((x_raw - X_MIN) / (X_MAX - X_MIN)) * DISPLAY_WIDTH)
            sy = int(((y_raw - Y_MIN) / (Y_MAX - Y_MIN)) * DISPLAY_HEIGHT)
            
            # Clamp
            sx = max(0, min(DISPLAY_WIDTH, sx))
            sy = max(0, min(DISPLAY_HEIGHT, sy))
            
            # Debounce
            current_time = time.time()
            if current_time - self.last_touch_time > 0.3:
                self.last_touch_time = current_time
                return (sx, sy)
        except:
            pass
        
        return None
    
    def check_button(self, touch_pos, button_rect):
        """Check if touch is inside button"""
        if touch_pos is None:
            return False
        
        tx, ty = touch_pos
        x1, y1, x2, y2 = button_rect
        return (x1 <= tx <= x2) and (y1 <= ty <= y2)

# ========================================
# Camera with Dual Resolution
# ========================================
class OptimizedPiCamera:
    """Hardware-accelerated camera with dual resolution support"""
    
    def __init__(self):
        self.picam2 = None
        self.preview_config = None
        self.capture_config = None
        
    def start(self):
        """Initialize dual-resolution pipeline"""
        try:
            self.picam2 = Picamera2()
            
            # Preview config: 320×240 @ ~40 FPS (fast, for detection)
            # Using Mode 2 (1640×1232) for full FOV at higher FPS
            self.preview_config = self.picam2.create_preview_configuration(
                main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"},
                raw={"size": (1640, 1232)},  # Mode 2: Full FOV, ~40 FPS
                controls={
                    "AfMode": controls.AfModeEnum.Continuous,
                    "AwbEnable": True,
                    "AeEnable": True,
                    "FrameRate": 30
                }
            )
            
            # Capture config: 1640×1232 (high-res for embeddings)
            self.capture_config = self.picam2.create_still_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
            )
            
            # Start with preview config
            self.picam2.configure(self.preview_config)
            self.picam2.start()
            
            time.sleep(2)  # Camera warm-up
            print("✅ Camera initialized: IMX219 @ 320×240 (preview), 1640×1232 (capture)")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def read_preview(self):
        """Capture preview frame (fast, for detection)"""
        try:
            frame = self.picam2.capture_array()
            return True, frame
        except:
            return False, None
    
    def capture_high_res(self):
        """Switch to full resolution and capture"""
        try:
            # Switch to capture config
            self.picam2.stop()
            self.picam2.configure(self.capture_config)
            self.picam2.start()
            time.sleep(0.5)
            
            # Capture
            frame = self.picam2.capture_array()
            
            # Switch back to preview
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

# ========================================
# BlazeFace Detector
# ========================================
class BlazeFaceDetector:
    """MediaPipe BlazeFace detector"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE
        )
    
    def detect(self, frame):
        """Detect faces in RGB frame, returns list of (x, y, w, h)"""
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
            
            # Clamp
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            faces.append((x, y, width, height))
        
        return faces
    
    def close(self):
        self.detector.close()

# ========================================
# Database Management
# ========================================
def load_database():
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            print("⚠️  Database corrupted, creating new one.")
    return {}

def save_database(db):
    try:
        with open(DATABASE_FILE, 'wb') as f:
            pickle.dump(db, f)
        print("💾 Database saved.")
    except Exception as e:
        print(f"❌ Save failed: {e}")

# ========================================
# Face Capture & Processing
# ========================================
def capture_face_embedding(camera, detector, display, mode='add'):
    """
    Capture face and extract embedding.
    Returns: embedding vector or None
    """
    print(f"\n[{mode.upper()}] Mode: Align face in preview, auto-capture in 3s")
    
    countdown = 3
    start_time = time.time()
    embedding = None
    
    while countdown > 0:
        ret, frame = camera.read_preview()
        if not ret:
            time.sleep(0.01)
            continue
        
        # Detect faces
        faces = detector.detect(frame)
        
        # Draw UI with countdown
        if len(faces) > 0:
            elapsed = time.time() - start_time
            countdown = 3 - int(elapsed)
            if countdown < 0:
                countdown = 0
            
            status = f"CAPTURING IN {countdown}..." if countdown > 0 else "PROCESSING..."
        else:
            status = "No face detected!"
            start_time = time.time()  # Reset timer if no face
            countdown = 3
        
        # Draw boxes on frame
        display_frame = display.draw_face_boxes(frame, faces)
        display.draw_ui(display_frame, status=status, fps=0, num_faces=len(faces), mode=mode)
        
        # Auto-capture after 3 seconds
        if len(faces) > 0 and countdown == 0:
            print("📸 Capturing high-resolution image...")
            
            # Switch to high-res
            hires_frame_rgb = camera.capture_high_res()
            if hires_frame_rgb is None:
                print("❌ High-res capture failed!")
                return None
            
            # Re-detect on high-res
            hires_faces = detector.detect(hires_frame_rgb)
            if len(hires_faces) == 0:
                print("⚠️  Face lost in high-res capture!")
                return None
            
            # Use largest face
            (x, y, w, h) = max(hires_faces, key=lambda f: f[2] * f[3])
            
            # Add margin
            margin = int(max(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(CAPTURE_WIDTH, x + w + margin)
            y2 = min(CAPTURE_HEIGHT, y + h + margin)
            
            face_crop_rgb = hires_frame_rgb[y1:y2, x1:x2]
            
            # Convert to BGR for saving
            face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
            
            # Save and process with Rust
            temp_file = "temp_capture.jpg"
            cv2.imwrite(temp_file, face_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print("🦀 Processing with Rust engine...")
            embedding = get_face_embedding(temp_file)
            
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            if embedding is not None:
                print("✅ Embedding extracted!")
                return embedding
            else:
                print("❌ Rust engine failed.")
                return None
    
    return None

# ========================================
# Main Application
# ========================================
def main():
    print("=" * 60)
    print("  Raspberry Pi 4 Face Recognition System")
    print("  TFT Display Edition")
    print("=" * 60)
    
    # Initialize components
    display = TFTDisplay()
    touch = TouchScreen() if USE_TOUCH else None
    camera = OptimizedPiCamera()
    
    if not camera.start():
        sys.exit(1)
    
    detector = BlazeFaceDetector()
    db = load_database()
    
    print(f"✅ System ready! Database: {len(db)} faces")
    print("\nControls:")
    if USE_TOUCH:
        print("  - Touch 'ADD FACE' to register new face")
        print("  - Touch 'VERIFY' to authenticate")
    else:
        print("  - Press 'a' in terminal for ADD FACE")
        print("  - Press 'v' in terminal for VERIFY")
    
    # Main loop
    mode = None  # 'add', 'verify', or None
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    try:
        while True:
            # Read camera frame
            ret, frame = camera.read_preview()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Detect faces
            faces = detector.detect(frame)
            
            # Draw UI
            display_frame = display.draw_face_boxes(frame, faces)
            status = "Ready" if mode is None else f"Mode: {mode.upper()}"
            display.draw_ui(display_frame, status=status, fps=fps, num_faces=len(faces), mode=mode)
            
            # Handle touch input
            if USE_TOUCH and touch:
                touch_pos = touch.read_touch()
                
                if touch.check_button(touch_pos, display.btn_add):
                    mode = 'add'
                    print("\n[ADD FACE] Starting...")
                    
                    # Get name from user (terminal input for now)
                    # TODO: Implement on-screen keyboard for production
                    print("Enter name in terminal:")
                    # For now, just capture without name prompt
                    
                    embedding = capture_face_embedding(camera, detector, display, 'add')
                    if embedding is not None:
                        # In production, get name from on-screen keyboard
                        # For now, use timestamp as name
                        name = f"Person_{int(time.time())}"
                        db[name] = embedding
                        save_database(db)
                        print(f"✨ Registered: {name}")
                    
                    mode = None
                
                elif touch.check_button(touch_pos, display.btn_verify):
                    mode = 'verify'
                    print("\n[VERIFY] Starting...")
                    
                    if len(db) == 0:
                        print("⚠️  Database empty!")
                        mode = None
                        continue
                    
                    embedding = capture_face_embedding(camera, detector, display, 'verify')
                    if embedding is not None:
                        # Search database
                        best_score = -1.0
                        best_name = "Unknown"
                        
                        for name, stored_vec in db.items():
                            score = compute_similarity(embedding, stored_vec)
                            if score > best_score:
                                best_score = score
                                best_name = name
                        
                        if best_score > SIMILARITY_THRESHOLD:
                            print(f"🔓 ACCESS GRANTED: {best_name} ({best_score:.3f})")
                        else:
                            print(f"🔒 ACCESS DENIED (Best: {best_name}, {best_score:.3f})")
                    
                    mode = None
            
            # Keyboard input fallback (if no touch)
            if not USE_TOUCH:
                # Non-blocking keyboard input would require threading
                # For simplicity, touch is recommended for production
                pass
            
            time.sleep(0.01)  # Small delay
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    finally:
        camera.stop()
        detector.close()
        save_database(db)
        
        # Clear display
        display.draw.rectangle((0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT), fill=(0, 0, 0))
        display.disp.image(display.image)
        
        if USE_TOUCH:
            GPIO.cleanup()
        
        print("✅ Cleanup complete.")

if __name__ == "__main__":
    main()