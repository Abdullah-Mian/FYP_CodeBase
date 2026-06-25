#!/usr/bin/env python3
"""
BlazeFace on TFT — max FPS, full FOV, correct RGB colors using MediaPipe BlazeFace
"""

import time
import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import ili9341
from picamera2 import Picamera2
import numpy as np
import mediapipe as mp

# ------------------ CONFIGURATION ------------------
# TFT Display pins
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D27)

spi = board.SPI()
disp = ili9341.ILI9341(
    spi, cs=cs_pin, dc=dc_pin, rst=reset_pin,
    baudrate=64000000, width=240, height=320, rotation=90
)
print("✅ TFT Display initialized: 320×240")

# Mediapipe BlazeFace detector
mp_face = mp.solutions.face_detection
blazeface = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
print("✅ Mediapipe BlazeFace detector initialized")

# ------------------ CAMERA SETUP ------------------
picam2 = Picamera2()

# Configuring the camera to provide full FOV via its `raw` stream
# Main output config: 320x240 (matches TFT), raw maintains full-resolution input
camera_config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},  # Match TFT display resolution
    raw={"size": picam2.sensor_resolution},         # Full FOV via raw sensor mode
    buffer_count=4,
)
picam2.configure(camera_config)
picam2.start()
time.sleep(1)

print("✅ Camera initialized: 320×240 RGB, full FOV via GPU ISP")
print("🎬 Starting... Press Ctrl+C to exit.")

# ------------------ MAIN LOOP ------------------
fps_smooth = 0.0
face_count = 0
start_time = time.time()

try:
    while True:
        # Read frame from camera (320x240 RGB)
        frame_rgb = picam2.capture_array("main")  # Native RGB, directly from picamera2

        # Detect faces
        results = blazeface.process(frame_rgb)
        frame_pil = Image.fromarray(frame_rgb)  # For TFT rendering
        draw = ImageDraw.Draw(frame_pil)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                # Map relative coordinates back to 320x240 frame
                x_min = int(bbox.xmin * 320)
                y_min = int(bbox.ymin * 240)
                width = int(bbox.width * 320)
                height = int(bbox.height * 240)
                x_max = x_min + width
                y_max = y_min + height

                # Draw rectangle and score
                draw.rectangle(
                    [(x_min, y_min), (x_max, y_max)],
                    outline=(0, 255, 0)
                )
                draw.text(
                    (x_min, y_min - 10),
                    f"{detection.score[0]:.2f}",
                    fill=(255, 255, 255)
                )

        # FPS calculation
        elapsed = time.time() - start_time
        fps_smooth = 0.9 * fps_smooth + 0.1 / max(elapsed, 0.001)
        start_time = time.time()

        # Draw FPS
        fps_display = f"FPS: {fps_smooth:.1f} Faces: {len(results.detections) if results.detections else 0}"
        draw.text((8, 10), fps_display, fill=(255, 255, 0))

        # Show frame on TFT display
        disp.image(frame_pil)

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting...")
    picam2.stop()
    disp.fill(0)
    print("TFT display cleared, and camera stopped. 👋 Bye!")
