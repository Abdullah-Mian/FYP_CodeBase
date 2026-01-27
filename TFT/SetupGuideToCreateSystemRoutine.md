# Complete Production System Setup Guide

## Overview

This guide will help you set up a production-ready camera system that:
- ✅ Starts automatically on boot (no login required)
- ✅ Runs as systemd service with auto-restart
- ✅ Keeps your password security intact
- ✅ Ready for ML model integration
- ✅ Professional logging and monitoring

---

## Quick Start (5 Minutes)

```bash
cd ~/Documents/TFT_LCD

# Make install script executable
chmod +x install_service.sh

# Install the service
sudo bash install_service.sh

# Reboot to test auto-start
sudo reboot
```

That's it! Your camera display will start automatically on boot.

---

## Detailed Setup Instructions

### Step 1: Prepare Your Files

Ensure you have these files in `/home/faizy/Documents/TFT_LCD/`:

```
TFT_LCD/
├── main_cam.py                  # Your camera script
├── camera-system.service        # Systemd service file
├── install_service.sh           # Installation script
├── requirements.txt             # Python dependencies
├── env/                         # Virtual environment
└── SETUP_INSTRUCTIONS.md        # Original setup guide
```

### Step 2: Install the Service

```bash
# Navigate to project directory
cd ~/Documents/TFT_LCD

# Make installation script executable
chmod +x install_service.sh

# Run installation (requires sudo)
sudo bash install_service.sh
```

The script will:
1. Check all prerequisites
2. Install service file to `/etc/systemd/system/`
3. Enable auto-start on boot
4. Optionally start the service immediately

### Step 3: Verify Installation

```bash
# Check if service is running
sudo systemctl status camera-system.service

# View live logs
sudo journalctl -u camera-system.service -f

# Check if enabled for boot
sudo systemctl is-enabled camera-system.service
```

Expected output:
```
● camera-system.service - Camera TFT Display System
     Loaded: loaded (/etc/systemd/system/camera-system.service; enabled)
     Active: active (running) since ...
```

### Step 4: Test Auto-Start

```bash
# Reboot your Pi
sudo reboot

# After reboot, check if service started automatically
sudo systemctl status camera-system.service
```

Your camera display should be running without any login!

---

## Service Management

### Essential Commands

```bash
# Start service
sudo systemctl start camera-system.service

# Stop service
sudo systemctl stop camera-system.service

# Restart service
sudo systemctl restart camera-system.service

# Enable auto-start on boot
sudo systemctl enable camera-system.service

# Disable auto-start on boot
sudo systemctl disable camera-system.service

# View status
sudo systemctl status camera-system.service

# View logs (real-time)
sudo journalctl -u camera-system.service -f

# View logs (last 50 lines)
sudo journalctl -u camera-system.service -n 50

# View logs since last boot
sudo journalctl -u camera-system.service -b

# View logs for specific date
sudo journalctl -u camera-system.service --since "2025-01-27"
```

### Troubleshooting Commands

```bash
# Check if service file is valid
sudo systemd-analyze verify camera-system.service

# Reload systemd after editing service file
sudo systemctl daemon-reload

# View service dependencies
systemctl list-dependencies camera-system.service

# Check why service failed
sudo systemctl status camera-system.service --no-pager --full
```

---

## Understanding the Service Configuration

### Service File Breakdown

```ini
[Unit]
Description=Camera TFT Display System
After=multi-user.target          # Start after basic system is ready
Wants=network-online.target      # Wait for network (optional)

[Service]
Type=simple                      # Process runs in foreground
User=faizy                       # Run as your user (not root!)
WorkingDirectory=/home/faizy/... # Set working directory

# Environment variables for Python
Environment="PYTHONUNBUFFERED=1" # Immediate log output
Environment="DISPLAY=:0"         # For GUI apps (if needed later)

# Command to run
ExecStart=/path/to/env/bin/python /path/to/main_cam.py

# Auto-restart policy
Restart=always                   # Always restart if crashes
RestartSec=10                    # Wait 10s before restart
StartLimitBurst=5                # Max 5 restarts in interval

# Logging
StandardOutput=journal           # Send stdout to systemd journal
StandardError=journal            # Send stderr to systemd journal

[Install]
WantedBy=multi-user.target       # Start in multi-user mode
```

### Why This is Secure

- ✅ Runs as your user (faizy), not root
- ✅ Password still required for manual login
- ✅ Only this specific service auto-starts
- ✅ Full audit trail in system logs
- ✅ Can be disabled anytime

---

## Modifying the Service

### Edit Service File

```bash
# Edit the service file
sudo nano /etc/systemd/system/camera-system.service

# After editing, reload systemd
sudo systemctl daemon-reload

# Restart the service
sudo systemctl restart camera-system.service
```

### Common Modifications

**Change Python script:**
```ini
ExecStart=/home/faizy/Documents/TFT_LCD/env/bin/python /home/faizy/Documents/TFT_LCD/new_script.py
```

**Add environment variables:**
```ini
Environment="MY_VAR=value"
Environment="DEBUG=1"
```

**Change restart policy:**
```ini
Restart=on-failure  # Only restart on crashes, not normal exits
RestartSec=5        # Restart faster
```

**Add delay before start:**
```ini
ExecStartPre=/bin/sleep 5  # Wait 5 seconds before starting
```

---

## Advanced: Multiple Services

If you want to run multiple components as separate services:

### Camera Service
```bash
sudo nano /etc/systemd/system/camera-capture.service
```

### Display Service
```bash
sudo nano /etc/systemd/system/camera-display.service
```

### ML Processing Service
```bash
sudo nano /etc/systemd/system/ml-processor.service
```

Each service can be independently started/stopped/monitored!

---

## Logging and Monitoring

### View Logs

```bash
# Real-time logs (like tail -f)
sudo journalctl -u camera-system.service -f

# Logs from last 1 hour
sudo journalctl -u camera-system.service --since "1 hour ago"

# Logs with priority ERROR and above
sudo journalctl -u camera-system.service -p err

# Export logs to file
sudo journalctl -u camera-system.service > camera-logs.txt
```

### Add Logging to Your Script

Modify `main_cam.py` to add logging:

```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Goes to systemd journal
    ]
)
logger = logging.getLogger(__name__)

# Use throughout your code
logger.info("Camera initialized")
logger.warning("Low frame rate detected")
logger.error("Camera connection failed")
```

These logs will appear in `journalctl`!

### Performance Monitoring

```bash
# CPU/Memory usage
systemctl status camera-system.service

# Detailed resource usage
systemd-cgtop

# Check how long service has been running
systemctl show camera-system.service | grep ActiveEnterTimestamp
```

---

## Integration with ML Models

### Modular Approach (Recommended)

Create separate modules:

```
TFT_LCD/
├── main_system.py           # Orchestrator
├── modules/
│   ├── camera_module.py     # Camera handling
│   ├── display_module.py    # Display handling
│   ├── ml_model_1.py        # First ML model
│   ├── ml_model_2.py        # Second ML model
│   └── ml_model_3.py        # Third ML model
└── config.yaml              # Configuration
```

### Update Service to Use Orchestrator

```ini
ExecStart=/home/faizy/Documents/TFT_LCD/env/bin/python /home/faizy/Documents/TFT_LCD/main_system.py
```

### Benefits

- ✅ Clean separation of concerns
- ✅ Easy to add/remove models
- ✅ Better debugging
- ✅ Can run models on demand vs continuously

---

## Python vs C++ Decision Guide

### Stick with Python If:
- ✅ Current performance is acceptable
- ✅ Need rapid development/iteration
- ✅ Team knows Python better
- ✅ Using TensorFlow/PyTorch models
- ✅ Want easy debugging and maintenance

### Add C++ Only If:
- ✅ Profiling shows Python as bottleneck
- ✅ Need sub-millisecond timing
- ✅ Memory critically constrained
- ✅ Specific algorithm benefits from C++
- ✅ Have C++ expertise on team

### Hybrid Approach (Best of Both):
```python
# Python orchestrator
from my_cpp_module import fast_inference

def process_frame(frame):
    # Do Python stuff
    result = fast_inference(frame)  # Call C++ for hot path
    # Continue with Python
```

**Start with pure Python. Add C++ only where proven necessary.**

---

## Uninstallation

If you need to remove the service:

```bash
# Stop the service
sudo systemctl stop camera-system.service

# Disable auto-start
sudo systemctl disable camera-system.service

# Remove service file
sudo rm /etc/systemd/system/camera-system.service

# Reload systemd
sudo systemctl daemon-reload

# Reset failed units (cleanup)
sudo systemctl reset-failed
```

---

## FAQ

### Q: Will this work after updating Raspberry Pi OS?
**A:** Yes! The service will persist across updates.

### Q: Can I still log in manually with my password?
**A:** Yes! Password security is unchanged. Only this service auto-starts.

### Q: How do I temporarily disable auto-start?
**A:** `sudo systemctl disable camera-system.service`

### Q: How do I update my Python script?
**A:** Just edit `main_cam.py` and run:
```bash
sudo systemctl restart camera-system.service
```

### Q: Can I run this on Raspberry Pi 3/5?
**A:** Yes! Works on all Raspberry Pi models with camera support.

### Q: What if the service crashes?
**A:** It automatically restarts (up to 5 times per interval).

### Q: How do I check if it's using too much CPU?
**A:** `systemctl status camera-system.service` shows resource usage.

---

## Next Steps

1. ✅ **Test auto-start:** Reboot and verify system starts
2. ✅ **Add logging:** Enhance your script with proper logging
3. ✅ **Integrate ML models:** Add your 2-3 ML models
4. ✅ **Monitor performance:** Use `journalctl` to watch system behavior
5. ✅ **Optimize if needed:** Profile first, then optimize hot spots

---

## Support and Resources

- **Systemd Documentation:** `man systemd.service`
- **Journal Documentation:** `man journalctl`
- **Raspberry Pi Forums:** https://forums.raspberrypi.com
- **Your project logs:** `sudo journalctl -u camera-system.service -f`

---

## Summary

You now have:
- ✅ Auto-starting camera system (no login needed)
- ✅ Robust service with auto-restart
- ✅ Professional logging and monitoring
- ✅ Security maintained (password still required)
- ✅ Foundation for ML model integration
- ✅ Production-ready architecture

**Your system is production-ready!** Start with Python, add ML models, profile performance, and only add C++ if profiling proves it necessary.