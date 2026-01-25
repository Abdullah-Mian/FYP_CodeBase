#!/bin/bash
# Automated installation script for MobileFaceNet on Raspberry Pi
# Usage: bash install.sh

set -e  # Exit on error

echo "=============================================="
echo "  MobileFaceNet Raspberry Pi Installer"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Installing system dependencies... ${NC}"
sudo apt update
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
    python3-dev \
    python3-picamera2 \
    python3-libcamera \
    python3-kms++

echo -e "${GREEN}✓ System dependencies installed${NC}"

echo -e "${YELLOW}Step 2: Testing camera... ${NC}"
if rpicam-hello --list-cameras 2>/dev/null; then
    echo -e "${GREEN}✓ Camera detected${NC}"
else
    echo -e "${RED}✗ Camera not detected! ${NC}"
    echo "Please enable camera:  sudo raspi-config -> Interface Options -> Camera"
    exit 1
fi

echo -e "${YELLOW}Step 3: Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment already exists.  Skipping..."
else
    python3 -m venv --system-site-packages venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

echo -e "${YELLOW}Step 4: Installing Python packages...${NC}"
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
pip install opencv-python opencv-contrib-python requests pillow

echo -e "${GREEN}✓ Python packages installed${NC}"

echo -e "${YELLOW}Step 5: Verifying installation...${NC}"
NUMPY_VERSION=$(pip list | grep numpy | awk '{print $2}')
echo "NumPy version: $NUMPY_VERSION"

if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo -e "${RED}✗ NumPy 2.x detected. This may cause issues. ${NC}"
    pip install "numpy>=1.24.0,<2.0.0"
else
    echo -e "${GREEN}✓ NumPy version compatible${NC}"
fi

echo ""
echo -e "${GREEN}=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "To run the application:"
echo "  cd MobileFacenet"
echo "  source ../venv/bin/activate"
echo "  python mobilefacenet.py"
echo ""