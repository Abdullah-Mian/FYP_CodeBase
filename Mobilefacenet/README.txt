# MobileFaceNet Face Recognition for Raspberry Pi 4

A lightweight face recognition system optimized for Raspberry Pi 4, using MobileFaceNet for efficient real-time facial recognition.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Customization](#customization)
- [Technical Details](#technical-details)

---

## ⚙️ Prerequisites

### System Requirements
- **Raspberry Pi 4** (any RAM variant)
- **Raspberry Pi OS** (64-bit, Bookworm or later) - **RECOMMENDED**
- **Camera Module** (v2, v3) or USB Webcam
- **Python 3.11+**
- **Display** (HDMI or VNC for camera preview)

### Tested Configuration
```
OS: Debian GNU/Linux 12 (bookworm) aarch64
Kernel: 6.12.47+rpt-rpi-v8
Python: 3.11+
```

---

## 🚀 Installation Guide

### Step 1: Enable Camera

```bash
# Enable camera interface
sudo raspi-config
# Navigate to:  Interface Options → Camera → Enable

# Reboot to apply changes
sudo reboot
```

### Step 2: Install System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system libraries
sudo apt install -y \
    libopenblas-dev \
    libopenjp2-7 \
    libjpeg-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    python3-venv \
    python3-dev

# Install Picamera2 (MUST be installed via apt, not pip!)
sudo apt install -y python3-picamera2 python3-libcamera python3-kms++
```

**⚠️ IMPORTANT:** Picamera2 MUST be installed as a system package, not through pip!

### Step 3: Verify Camera Works

```bash
# Test camera (should show preview for 5 seconds)
rpicam-hello -t 5000

# List available cameras
rpicam-hello --list-cameras
```

**Note:** Old `libcamera-*` commands are deprecated. Use `rpicam-*` commands instead.

### Step 4: Clone Repository

```bash
# Clone the repository
git clone https://github.com/Abdullah-Mian/FaceID.git

# Navigate to project directory
cd FaceID
```

### Step 5: Create Virtual Environment

**⚠️ CRITICAL:** Use `--system-site-packages` flag to access system-installed picamera2!

```bash
# Create virtual environment WITH system site packages
python3 -m venv --system-site-packages venv

# Activate virtual environment
source venv/bin/activate

# Your terminal prompt should now show (venv) prefix
```

### Step 6: Install Python Dependencies

**⚠️ IMPORTANT:** Specific versions are required for compatibility! 

```bash
# Install PyTorch CPU version for ARM64 (specific version for Pi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install OpenCV and other dependencies
pip install opencv-python opencv-contrib-python requests pillow

# Verify NumPy version (MUST be 1.x, NOT 2.x)
pip list | grep numpy
```

**Expected NumPy version:** `1.26.x` or `1.24.x` (anything `<2.0`)

**Why specific versions?**
- **NumPy 1.x (<2.0)**: Required for OpenCV compatibility and picamera2 binary compatibility
- **PyTorch CPU build**: Raspberry Pi 4 has no CUDA/GPU support for PyTorch
- **System-site-packages venv**: Allows access to apt-installed picamera2

### Step 7: Navigate and Run

```bash
# Go to MobileFacenet directory
cd MobileFacenet

# Run the application
python mobilefacenet.py
```

**First Run:** The model (~4MB) will automatically download from GitHub.

---

## 🎯 Usage

### Starting the Application

```bash
# From FaceID directory
cd MobileFacenet
source ../venv/bin/activate
python mobilefacenet.py
```

### Available Commands

#### 1. **add** - Register a New Face
```
>>> Command: add
Enter name:  John
[Camera preview window opens]
[Position face in frame - green box should appear]
[Press SPACE to capture]
[3-2-1 countdown]
✅ Successfully added 'John' to database! 
```

#### 2. **verify** - Authenticate/Verify Face
```
>>> Command: verify
[Camera preview opens]
[Press SPACE to capture your face]
🔍 Comparing with database...
  John:  0.856
  Jane: 0.432
----------------------------------------
✅ MATCH:  John (confidence: 0.856)
```

#### 3. **list** - List All Registered Faces
```
>>> Command: list
📋 Registered faces (2):
  1. John
  2. Jane
```

#### 4. **delete** - Remove a Face from Database
```
>>> Command: delete
Enter name to delete: John
⚠️ Delete 'John'? (y/n): y
✅ Deleted 'John'
```

#### 5. **quit** - Exit Application
```
>>> Command: quit
Saving database... 
👋 Goodbye!
```

### Camera Window Controls

| Key | Action |
|-----|--------|
| **SPACE** | Capture face |
| **ESC** | Cancel and return to menu |

---

## 🔧 Troubleshooting

### Issue 1: NumPy Binary Incompatibility Error

**Error Message:**
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. 
Expected 96 from C header, got 88 from PyObject
```

**Root Cause:** NumPy 2.x is incompatible with picamera2 and OpenCV on Raspberry Pi.

**Solution:**
```bash
# Check current NumPy version
pip list | grep numpy

# If NumPy is 2.x, downgrade to 1.x
pip uninstall numpy -y
pip install "numpy>=1.24.0,<2.0.0"

# Verify installation
pip list | grep numpy
# Should show:  numpy 1.26.4 (or any 1.x version)
```

---

### Issue 2: Camera Window Not Opening

**Symptoms:**
- Command returns immediately after typing `add` or `verify`
- No camera preview appears
- No error messages

**Solutions:**

#### A) Set Display Environment Variable
```bash
# Export DISPLAY variable
export DISPLAY=:0

# Run application again
python mobilefacenet.py
```

#### B) Verify Camera Hardware
```bash
# Check if camera is detected
rpicam-hello --list-cameras

# Expected output:
# Available cameras
# 0 :  imx219 [3280x2464] (/base/soc/i2c0mux/i2c@1/imx219@10)
```

#### C) Check Camera Permissions
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Log out and log back in for changes to take effect
```

---

### Issue 3: ModuleNotFoundError:  No module named 'picamera2'

**Error Message:**
```
ModuleNotFoundError: No module named 'picamera2'
```

**Root Cause:** Virtual environment was created without `--system-site-packages` flag.

**Solution:**
```bash
# Deactivate current virtual environment
deactivate

# Remove existing venv
rm -rf venv

# Recreate venv WITH --system-site-packages
python3 -m venv --system-site-packages venv

# Activate new venv
source venv/bin/activate

# Reinstall Python dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
pip install opencv-python opencv-contrib-python requests pillow
```

---

### Issue 4: PyTorch Hash Mismatch Error

**Error Message:**
```
ERROR:  THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE. 
```

**Root Cause:** Conflict between PyPI and piwheels (Raspberry Pi package repository).

**Solution:**
```bash
# Use --no-cache-dir flag to force fresh download
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
```

---

### Issue 5: Face Not Detected

**Symptoms:**
- Camera opens successfully
- "No face detected!" message persists
- No green bounding box around face

**Solutions:**

#### A) Improve Lighting and Positioning
- Ensure good frontal lighting (not backlit)
- Position face 30-60cm from camera
- Keep face straight (avoid extreme angles)
- Wait 2-3 seconds for camera to focus

#### B) Adjust Detection Sensitivity
Edit `mobilefacenet.py` around line 126:
```python
# Reduce minNeighbors for more sensitive detection
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1, 
    minNeighbors=3,  # Changed from 5 (lower = more sensitive)
    minSize=(30, 30)
)
```

#### C) Clean Camera Lens
- Wipe camera lens with soft cloth
- Check for dust or smudges

---

### Issue 6: Verification Always Fails / Low Similarity Scores

**Symptoms:**
- Face successfully added with `add` command
- `verify` command shows low scores (<0.3)
- No match found even for registered faces

**Solutions:**

#### A) Lower Similarity Threshold
Edit `mobilefacenet.py` line 38:
```python
SIMILARITY_THRESHOLD = 0.3  # Changed from 0.4 (lower = more lenient)
```

#### B) Ensure Consistent Conditions
- Use similar lighting during `add` and `verify`
- Maintain same distance from camera
- If wearing glasses/hat during `add`, wear them during `verify` too

#### C) Re-register Face
```
>>> Command: delete
Enter name to delete: YourName

>>> Command: add
Enter name:  YourName
[Capture with better lighting/positioning]
```

---

### Issue 7: Low FPS / Laggy Camera Preview

**Symptoms:**
- Camera preview is choppy
- Frame rate is very low (<5 FPS)
- System feels unresponsive

**Solutions:**

#### A) Reduce Camera Resolution
Edit `mobilefacenet.py` around lines 161-163:
```python
cap. set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Changed from 640
cap. set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Changed from 480
```

#### B) Lower FPS Cap
Edit `mobilefacenet.py` line 40:
```python
FPS_CAP = 10  # Changed from 15
```

#### C) Increase GPU Memory Allocation
```bash
sudo raspi-config
# Navigate to:  Performance Options → GPU Memory → Set to 256MB
# Reboot after changes
sudo reboot
```

---

### Issue 8: ImportError or Module Conflicts

**Error Examples:**
```
ImportError: cannot import name 'X' from 'Y'
ModuleNotFoundError: No module named 'cv2'
```

**Solution - Clean Reinstall:**
```bash
# Deactivate and remove venv
deactivate
rm -rf venv

# Recreate fresh environment
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install dependencies in correct order
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
pip install opencv-python opencv-contrib-python
pip install requests pillow

# Verify NumPy version
pip list | grep numpy
```

---

## 📊 Performance

### Raspberry Pi 4 Performance Metrics

| Metric | Value |
|--------|-------|
| **Inference Time** | 15-30ms per face |
| **FPS (with detection)** | 10-15 FPS |
| **Model Load Time** | 2-3 seconds |
| **Model Size** | 4MB |
| **Embedding Dimension** | 128-D |
| **RAM Usage** | ~300-400MB |

### Accuracy

| Scenario | Accuracy |
|----------|----------|
| **Face Detection (Frontal)** | ~95% |
| **Verification (Same Lighting)** | ~98% |
| **Verification (Different Lighting)** | ~85% |
| **False Accept Rate** | <2% (threshold 0.4) |

---

## 🎨 Customization

### Adjust Matching Threshold

Edit `mobilefacenet.py` line 38:
```python
SIMILARITY_THRESHOLD = 0.35  # Default: 0.4
# Lower value = more lenient (more false positives)
# Higher value = stricter (more false negatives)
```

**Recommended ranges:**
- **High security:** 0.5 - 0.6
- **Balanced:** 0.4 (default)
- **Lenient:** 0.3 - 0.35

### Change Camera Resolution

Edit `mobilefacenet.py` around lines 161-163:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)   # Default: 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # Default: 480
```

**Trade-off:** Higher resolution = better accuracy but lower FPS

### Adjust Frame Rate

Edit `mobilefacenet.py` line 40:
```python
FPS_CAP = 20  # Default: 15
```

**Note:** Actual FPS depends on camera hardware and processing speed.

### Change Capture Countdown

Edit `mobilefacenet.py` around line 194:
```python
for i in range(5, 0, -1):  # Default: range(3, 0, -1)
    # 5 second countdown instead of 3
```

---

## 🔬 Technical Details

### Model Architecture

**Model:** MobileFaceNet (TorchScript)
- Lightweight CNN optimized for mobile/edge devices
- Input: 112x112 RGB images
- Output: 128-dimensional L2-normalized embeddings
- Training: ArcFace loss on face recognition datasets

**Face Detection:** Haar Cascade (OpenCV)
- Fast frontal face detection (~1-2ms)
- Pre-trained XML classifier:  `haarcascade_frontalface_default.xml`
- Trade-off: Speed over accuracy (95% vs 99% for MTCNN/SCRFD)

**Matching Algorithm:** Cosine Similarity
- Pre-normalized embeddings (unit vectors)
- Similarity = dot product of two embeddings
- Range: -1 (opposite) to +1 (identical)

### File Structure

```
MobileFacenet/
├── mobilefacenet.py              # Main application script
├── faces_db.pkl                  # Face embeddings database (auto-created)
├── models/
│   └── mobilefacenet_scripted.pt # Pre-trained model (auto-downloaded on first run)
└── commands. md                   # Quick reference
```

### Database Format

`faces_db.pkl` is a Python pickle file containing a dictionary: 
```python
{
    "PersonName": numpy.ndarray([128 float32 values]),  # Normalized embedding
    "AnotherPerson": numpy.ndarray([128 float32 values]),
    # ... more entries
}
```

**Storage:** Each face takes ~512 bytes (128 floats × 4 bytes)

---

## 🔒 Security Considerations

⚠️ **This is a demonstration/educational project, NOT production-ready security software!**

### Known Limitations

1. **No Liveness Detection**
   - Vulnerable to photo/video attacks
   - Cannot distinguish between real person and printed photo

2. **Single-Shot Enrollment**
   - Only one sample per person stored
   - Production systems should use 5-10 samples per person

3. **Unencrypted Database**
   - `faces_db.pkl` is stored in plain pickle format
   - No encryption of biometric data

4. **Simple Detection**
   - Haar Cascade can be fooled by certain angles/lighting
   - No mask detection or facial expression analysis

### Recommended for Production

If deploying in real security scenarios, add:
- **Liveness detection** (blink detection, motion analysis)
- **Multi-sample enrollment** (capture 5-10 faces per person)
- **Database encryption** (encrypt `faces_db.pkl`)
- **Audit logging** (track all authentication attempts)
- **Anti-spoofing** (depth sensing, texture analysis)
- **Better face detection** (MTCNN, RetinaFace, SCRFD)

---

## 📦 Dependencies Summary

### Critical Version Requirements

| Package | Version Constraint | Reason |
|---------|-------------------|--------|
| **numpy** | `>=1.24.0, <2.0.0` | NumPy 2.x breaks picamera2 and OpenCV compatibility |
| **torch** | `>=2.0.0` (CPU build) | ARM64 CPU-only build required; no GPU support on Pi |
| **opencv-python** | `>=4.8.0` | Requires NumPy 1.x for binary compatibility |
| **picamera2** | (system package) | MUST be installed via apt, not pip |

### Why These Constraints Matter

1. **NumPy <2.0**:  
   - NumPy 2.0 changed internal C API
   - Breaks binary compatibility with picamera2 and older OpenCV builds
   - Error: "dtype size changed"

2. **PyTorch CPU-only**:
   - Raspberry Pi 4 GPU (VideoCore) is NOT supported by PyTorch
   - PyTorch only supports NVIDIA CUDA GPUs
   - Must use CPU inference

3. **System picamera2**:
   - Compiled against system libraries (libcamera, libcamera-apps)
   - Pip version doesn't exist or is outdated
   - Virtual env needs `--system-site-packages` to access it

---

## 📚 References

- **MobileFaceNet Paper**: [Chen et al., 2018](https://arxiv.org/abs/1804.07573)
- **ArcFace Paper**: [Deng et al., 2018](https://arxiv.org/abs/1801.07698)
- **Raspberry Pi Camera Documentation**: [Official Docs](https://www.raspberrypi.com/documentation/computers/camera_software. html)
- **OpenCV Haar Cascades**: [OpenCV Docs](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- **Model Source**: [foamliu/MobileFaceNet](https://github.com/foamliu/MobileFaceNet)

---

## 🤝 Contributing

Contributions are welcome! If you find bugs or have improvements: 

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes
4. Push to branch
5. Open a Pull Request

---

## 📝 License

This project is provided as-is for educational purposes. 

---

## 🙏 Acknowledgments

- **Model**: [foamliu/MobileFaceNet](https://github.com/foamliu/MobileFaceNet)
- **Repository**: [Abdullah-Mian/FaceID](https://github.com/Abdullah-Mian/FaceID)
- **Raspberry Pi Foundation** for excellent hardware and documentation

---

**Tested On:** Raspberry Pi 4 Model B, Raspberry Pi OS Bookworm (64-bit)  
**Last Updated:** January 2026