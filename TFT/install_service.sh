#!/bin/bash
# Installation script for Camera TFT System Service
# Run with: sudo bash install_service.sh

set -e  # Exit on error

echo "======================================"
echo "Camera TFT System Service Installer"
echo "======================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run as root (sudo bash install_service.sh)"
    exit 1
fi

# Get actual user (not root)
ACTUAL_USER=${SUDO_USER:-$USER}
SERVICE_DIR="/home/$ACTUAL_USER/Documents/TFT_LCD"

echo "Installing service for user: $ACTUAL_USER"
echo "Service directory: $SERVICE_DIR"
echo ""

# Check if service directory exists
if [ ! -d "$SERVICE_DIR" ]; then
    echo "ERROR: Service directory not found: $SERVICE_DIR"
    exit 1
fi

# Check if main script exists
if [ ! -f "$SERVICE_DIR/main_cam.py" ]; then
    echo "ERROR: main_cam.py not found in $SERVICE_DIR"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$SERVICE_DIR/env" ]; then
    echo "ERROR: Virtual environment not found at $SERVICE_DIR/env"
    echo "Please create virtual environment first:"
    echo "  cd $SERVICE_DIR"
    echo "  python3 -m venv --system-site-packages env"
    exit 1
fi

# Copy service file to systemd directory
echo "Installing systemd service file..."
cp camera-system.service /etc/systemd/system/

# Set correct permissions
chmod 644 /etc/systemd/system/camera-system.service

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable service (auto-start on boot)
echo "Enabling service for auto-start on boot..."
systemctl enable camera-system.service

# Check if service should be started now
echo ""
read -p "Do you want to start the service now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting service..."
    systemctl start camera-system.service
    
    # Wait a moment for service to start
    sleep 2
    
    # Show status
    echo ""
    echo "Service status:"
    systemctl status camera-system.service --no-pager
else
    echo "Service not started. You can start it later with:"
    echo "  sudo systemctl start camera-system.service"
fi

echo ""
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo ""
echo "Useful commands:"
echo "  View status:        sudo systemctl status camera-system.service"
echo "  View logs:          sudo journalctl -u camera-system.service -f"
echo "  Stop service:       sudo systemctl stop camera-system.service"
echo "  Start service:      sudo systemctl start camera-system.service"
echo "  Restart service:    sudo systemctl restart camera-system.service"
echo "  Disable auto-start: sudo systemctl disable camera-system.service"
echo ""
echo "The service will automatically start on boot!"
echo ""