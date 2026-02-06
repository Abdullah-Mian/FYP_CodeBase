import cv2
import numpy as np
import os
import pickle
import time
import subprocess
import threading
import sys

# Import the wrapper for our new Rust Engine
# Ensure rust_wrapper.py is in the same folder
try:
    from rust_wrapper import get_face_embedding
except ImportError:
    print("❌ Error: rust_wrapper.py not found!")
    sys.exit(1)

# Configuration
SIMILARITY_THRESHOLD = 0.4  # Adjust based on testing (0.3 - 0.6)
DATABASE_FILE = 'faces_db.pkl'
FPS_CAP = 15

# ---------------------------------------------------------
# Camera Handling
# ---------------------------------------------------------

class RPiCamera:
    """Wrapper for rpicam-vid (libcamera) - Optimized for Pi"""
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.process = None
        self.running = False
        self.current_frame = None
        self.thread = None

    def start(self):
        try:
            # rpicam-vid command for raw MJPEG stream
            cmd = [
                'rpicam-vid',
                '--inline', '--nopreview',
                '-t', '0',
                '--width', str(self.width),
                '--height', str(self.height),
                '--codec', 'mjpeg',
                '-o', '-',
                '--framerate', str(FPS_CAP)
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            
            self.running = True
            self.thread = threading.Thread(target=self._read_frames, daemon=True)
            self.thread.start()
            
            time.sleep(1.5) # Allow camera to warm up
            print("✅  RPi Camera (libcamera) started!")
            return True
            
        except Exception as e:
            print(f"❌  Error starting RPi camera: {e}")
            return False
    
    def _read_frames(self):
        """Reads MJPEG stream from stdout"""
        # Simple buffer reader for MJPEG (looks for JPEG start/end bytes)
        stream_bytes = b''
        while self.running and self.process.poll() is None:
            try:
                stream_bytes += self.process.stdout.read(4096)
                a = stream_bytes.find(b'\xff\xd8') # JPEG Start
                b = stream_bytes.find(b'\xff\xd9') # JPEG End
                
                if a != -1 and b != -1:
                    jpg = stream_bytes[a:b+2]
                    stream_bytes = stream_bytes[b+2:]
                    
                    # Decode to OpenCV image
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        self.current_frame = img
            except Exception:
                pass
    
    def read(self):
        if self.current_frame is not None:
            return True, self.current_frame.copy()
        return False, None
    
    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()

def init_camera():
    """Tries RPi Camera first, then OpenCV USB, then Dummy."""
    print("🎥  Initializing camera...")
    
    # 1. Try Raspberry Pi Camera (libcamera)
    try:
        cam = RPiCamera()
        if cam.start():
            return cam, 'rpicam'
    except:
        pass
    
    # 2. Try Standard USB Camera (OpenCV)
    print("⚠️  RPi Camera failed, trying USB/OpenCV...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0) # Fallback without V4L2
            
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("✅  USB Camera initialized!")
            return cap, 'opencv'
    except:
        pass
    
    # 3. Dummy Camera (for testing without hardware)
    print("❌  No camera found. Using DUMMY mode (Black screen).")
    return DummyCamera(), 'dummy'

class DummyCamera:
    def read(self):
        # Return black image with some text
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "NO CAMERA", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return True, img
    def release(self): pass
    def stop(self): pass

# ---------------------------------------------------------
# Database Handling
# ---------------------------------------------------------

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
        print("💾  Database saved.")
    except Exception as e:
        print(f"❌  Save failed: {e}")

# ---------------------------------------------------------
# Face Detection & Capture
# ---------------------------------------------------------

def detect_face(frame):
    """Uses OpenCV Haar Cascade for fast detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    # ScaleFactor 1.1, MinNeighbors 5
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return faces

def capture_face(camera, mode='add'):
    """GUI Loop to capture a face."""
    print(f"\n[{mode.upper()}] Mode: Press SPACE to capture, ESC to quit.")
    
    embedding = None
    window_name = f'FaceID - {mode.upper()}'
    
    # Check if we can show a window (Headless vs Desktop)
    has_display = 'DISPLAY' in os.environ
    if has_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            display_frame = frame.copy()
            faces = detect_face(frame)
            
            # Draw UI
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if mode == 'add' else (255, 200, 0)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # Handle Input
            if has_display:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(int(1000/FPS_CAP)) & 0xFF
            else:
                # Headless: Auto-capture if face detected
                key = 32 if len(faces) > 0 else -1
                time.sleep(0.1)

            if key == 27:  # ESC
                print("Cancelled.")
                break
            
            elif key == 32:  # SPACE
                if len(faces) == 0:
                    if has_display: print("❌ No face seen!")
                    continue
                
                # Use the largest face
                (x, y, w, h) = faces[0]
                
                # Add padding to crop (MobileFaceNet likes loose crops)
                margin = 20
                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)
                
                face_crop = frame[y1:y2, x1:x2]
                
                # --- RUST BRIDGE ---
                print("📸  Processing...")
                temp_file = "temp_capture.jpg"
                cv2.imwrite(temp_file, face_crop)
                
                print("🦀  Analyzing with Rust...")
                embedding = get_face_embedding(temp_file)
                
                # Cleanup
                if os.path.exists(temp_file): os.remove(temp_file)
                
                if embedding is not None:
                    print("✅  Face Vector Generated!")
                    break
                else:
                    print("❌  Rust Engine failed to process image.")
                # -------------------

    except KeyboardInterrupt:
        pass
    finally:
        if has_display:
            cv2.destroyAllWindows()
            
    return embedding

# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------

def add_user(camera, db):
    name = input("Enter Name: ").strip()
    if not name: return
    
    vec = capture_face(camera, 'add')
    if vec is not None:
        db[name] = vec
        save_database(db)
        print(f"✨ Registered: {name}")

def verify_user(camera, db):
    if not db:
        print("⚠️  Database is empty.")
        return
        
    vec = capture_face(camera, 'verify')
    if vec is None: return
    
    print("\n🔍  Searching Database...")
    best_score = -1.0
    best_name = "Unknown"
    
    for name, stored_vec in db.items():
        # Cosine Similarity: Dot product of normalized vectors
        score = np.dot(vec, stored_vec)
        print(f"   vs {name}: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_name = name
            
    print("-" * 30)
    if best_score > SIMILARITY_THRESHOLD:
        print(f"🔓  ACCESS GRANTED: {best_name} ({best_score:.2f})")
    else:
        print(f"🔒  ACCESS DENIED. Best match: {best_name} ({best_score:.2f})")

def main():
    print("\n=======================================")
    print("   Rust-Powered FaceID (MobileFaceNet)")
    print("=======================================")
    
    db = load_database()
    camera, cam_type = init_camera()
    
    try:
        while True:
            print("\n-----------------------")
            print(f"Faces in DB: {len(db)}")
            cmd = input("[A]dd Face, [V]erify, [L]ist, [Q]uit: ").lower().strip()
            
            if cmd == 'a':
                add_user(camera, db)
            elif cmd == 'v':
                verify_user(camera, db)
            elif cmd == 'l':
                print("Faces:", list(db.keys()))
            elif cmd == 'q':
                break
            else:
                print("Unknown command.")
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if hasattr(camera, 'stop'): camera.stop()
        elif hasattr(camera, 'release'): camera.release()
        save_database(db)

if __name__ == "__main__":
    main()
