# Import necessary libraries
import torch
import torchvision. transforms as transforms
import cv2
import numpy as np
import pickle
import os
import sys
import requests
import subprocess
import threading
from io import BytesIO
from PIL import Image

# Constants
MODEL_URL = 'https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet_scripted.pt'
MODEL_PATH = './models/mobilefacenet_scripted.pt'
DB_FILE = 'faces_db.pkl'
EMBED_SIZE = 128
SIMILARITY_THRESHOLD = 0.4
CAPTURE_DELAY = 2000
FPS_CAP = 15

# Preprocessing transform (ArcFace standard)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def download_model():
    """Downloads PTH if not cached."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MobileFaceNet model from {MODEL_URL}...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!  Cached locally.")
    else:
        print("Model already cached.")
    return MODEL_PATH

def load_model():
    """Loads MobileFaceNet model."""
    try:
        pth_path = download_model()
        model = torch.jit.load(pth_path, map_location='cpu')
        model.eval()
        print(f"MobileFaceNet loaded!  Embedding size: {EMBED_SIZE}")
        return model
    except Exception as e: 
        print(f"Error loading model: {e}")
        print("Check PTH download; ensure torch installed.")
        sys.exit(1)

def load_database():
    """Load face database from pickle file."""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'rb') as f:
            db = pickle. load(f)
        print(f"Loaded DB with {len(db)} faces.")
    else:
        db = {}
        print("Starting fresh DB.")
    return db

def save_database(db):
    """Save face database to pickle file."""
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db, f)
    print(f"Saved DB with {len(db)} faces.")

class RPiCamera:
    """Wrapper for Raspberry Pi camera using rpicam-vid."""
    def __init__(self):
        self.width = 640
        self.height = 480
        self.process = None
        self.current_frame = None
        self. running = False
        
    def start(self):
        """Start camera stream using rpicam-vid."""
        try:
            # Start rpicam-vid with stdout output
            cmd = [
                'rpicam-vid',
                '--inline',
                '--nopreview',
                '-t', '0',  # Run indefinitely
                '--width', str(self.width),
                '--height', str(self. height),
                '--codec', 'mjpeg',
                '-o', '-',  # Output to stdout
                '--framerate', '15'
            ]
            
            self.process = subprocess. Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            
            self.running = True
            # Start frame reading thread
            self.thread = threading.Thread(target=self._read_frames, daemon=True)
            self.thread.start()
            
            import time
            time.sleep(2)  # Camera warm-up
            print("✅ RPi Camera initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            print("Make sure rpicam-vid is installed:  sudo apt install rpicam-apps")
            return False
    
    def _read_frames(self):
        """Background thread to read MJPEG frames."""
        bytes_data = b''
        while self.running:
            try:
                chunk = self.process.stdout.read(1024)
                if not chunk: 
                    break
                    
                bytes_data += chunk
                
                # Find JPEG boundaries
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    # Decode JPEG to numpy array
                    img = cv2.imdecode(np. frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        self.current_frame = img
                        
            except Exception as e:
                print(f"Frame read error: {e}")
                break
    
    def read(self):
        """Read current frame."""
        if self.current_frame is not None:
            return True, self.current_frame. copy()
        return False, None
    
    def stop(self):
        """Stop camera."""
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()

def init_camera():
    """Initialize camera with fallback options."""
    print("🎥 Initializing camera...")
    
    # Try rpicam first
    try:
        cam = RPiCamera()
        if cam.start():
            return cam, 'rpicam'
    except:
        pass
    
    # Fallback to OpenCV with libcamera backend
    print("Trying OpenCV with libcamera backend...")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)
            print("✅ Camera initialized with OpenCV!")
            return cap, 'opencv'
    except:
        pass
    
    print("❌ Failed to initialize camera!")
    print("Please enable camera:  sudo raspi-config -> Interface Options -> Camera")
    sys.exit(1)

def detect_face(frame):
    """Simple face detection with Haar cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def extract_embedding(model, face_crop):
    """Extracts 128-dim embedding from cropped face."""
    try:
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        input_tensor = transform(face_rgb).unsqueeze(0)
        with torch.no_grad():
            embedding = model(input_tensor)
        embedding = embedding. squeeze().cpu().numpy()
        return embedding / np.linalg.norm(embedding)
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def capture_face(model, camera, cam_type, mode='add'):
    """Capture face using camera."""
    print(f"\n{mode. capitalize()} mode:  Press SPACE to capture, ESC to quit.")
    print("Position your face in front of the camera...")
    
    embedding = None
    window_name = f'Face Capture - {mode. upper()}'
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    
    delay = int(1000 / FPS_CAP)
    
    try:
        while True:
            ret, frame = camera.read()
            
            if not ret or frame is None:
                print("❌ Failed to read frame")
                continue
            
            display_frame = frame.copy()
            
            # Detect faces
            faces = detect_face(frame)
            
            # Draw boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{w}x{h}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Instructions
            instruction_text = "SPACE: Capture | ESC: Quit"
            cv2.putText(display_frame, instruction_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if len(faces) > 0:
                cv2.putText(display_frame, f"Faces detected: {len(faces)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No face detected!", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(delay) & 0xFF
            
            if key == 27:  # ESC
                print("Cancelled.")
                break
            elif key == 32:  # SPACE
                if len(faces) == 0:
                    print("❌ No face detected!  Try again.")
                    continue
                
                (x, y, w, h) = faces[0]
                face_crop = frame[y:y+h, x:x+w]
                
                print("📸 Hold still...")
                # Countdown
                for i in range(3, 0, -1):
                    temp_frame = display_frame.copy()
                    cv2.putText(temp_frame, str(i), (display_frame.shape[1]//2 - 30, display_frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
                    cv2.imshow(window_name, temp_frame)
                    cv2.waitKey(1000)
                
                embedding = extract_embedding(model, face_crop)
                if embedding is not None:
                    print(f"✅ Captured! Embedding shape: {embedding.shape}")
                    break
                else:
                    print("❌ Failed to extract embedding.  Try again.")
    
    except KeyboardInterrupt: 
        print("\nInterrupted by user.")
    finally:
        cv2.destroyAllWindows()
    
    return embedding

def add_face(model, camera, cam_type, db):
    """Enroll new face."""
    name = input("Enter name:  ").strip()
    if not name:
        print("❌ Name cannot be empty!")
        return
    
    if name in db:
        overwrite = input(f"⚠️  '{name}' already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Cancelled.")
            return
    
    embedding = capture_face(model, camera, cam_type, 'add')
    if embedding is not None:
        db[name] = embedding
        save_database(db)
        print(f"✅ Successfully added '{name}' to database!")
    else:
        print("❌ Failed to add face.")

def verify_face(model, camera, cam_type, db):
    """Verify face against database."""
    if not db:
        print("❌ Database is empty!  Add faces first.")
        return
    
    embedding = capture_face(model, camera, cam_type, 'verify')
    if embedding is None:
        print("❌ Failed to capture face.")
        return
    
    print("\n🔍 Comparing with database...")
    best_score, best_name = -1, None
    
    for name, stored in db.items():
        score = np.dot(embedding, stored)
        print(f"  {name}: {score:.3f}")
        if score > best_score:
            best_score, best_name = score, name
    
    print("-" * 40)
    if best_score > SIMILARITY_THRESHOLD:
        print(f"✅ MATCH:  {best_name} (confidence: {best_score:.3f})")
    else:
        print(f"❌ NO MATCH (best: {best_name}, score: {best_score:.3f})")
        print(f"   Threshold: {SIMILARITY_THRESHOLD}")

def list_faces(db):
    """List all registered faces."""
    if not db:
        print("❌ Database is empty!")
        return
    
    print(f"\n📋 Registered faces ({len(db)}):")
    for i, name in enumerate(db.keys(), 1):
        print(f"  {i}. {name}")

def delete_face(db):
    """Delete a face from database."""
    if not db:
        print("❌ Database is empty!")
        return
    
    list_faces(db)
    name = input("\nEnter name to delete: ").strip()
    
    if name in db:
        confirm = input(f"⚠️  Delete '{name}'? (y/n): ").lower()
        if confirm == 'y':
            del db[name]
            save_database(db)
            print(f"✅ Deleted '{name}'")
    else:
        print(f"❌ '{name}' not found in database!")

def main():
    """Main application loop."""
    print("=" * 50)
    print("  MobileFaceNet Face Recognition System")
    print("  Optimized for Raspberry Pi")
    print("=" * 50)
    
    # Initialize
    model = load_model()
    db = load_database()
    camera, cam_type = init_camera()
    
    print(f"\n✅ System ready! (Camera: {cam_type})")
    print("\nCommands:")
    print("  add     - Register a new face")
    print("  verify  - Verify a face")
    print("  list    - List all registered faces")
    print("  delete  - Delete a face")
    print("  quit    - Exit")
    
    try:
        while True:
            cmd = input("\n>>> Command: ").strip().lower()
            
            if cmd == 'add':
                add_face(model, camera, cam_type, db)
            elif cmd == 'verify':
                verify_face(model, camera, cam_type, db)
            elif cmd == 'list': 
                list_faces(db)
            elif cmd == 'delete': 
                delete_face(db)
            elif cmd == 'quit' or cmd == 'exit': 
                print("Saving database...")
                save_database(db)
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid command!  Use:  add, verify, list, delete, quit")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    finally:
        if cam_type == 'rpicam':
            camera.stop()
        else:
            camera.release()
        cv2.destroyAllWindows()
        save_database(db)
        print("✅ Cleanup complete.")

if __name__ == "__main__":
    main()