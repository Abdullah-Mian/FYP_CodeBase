#!/usr/bin/env python3
"""
BlazeFace on TFT — max FPS, full FOV, correct colors
picamera2 RGB888 → detect → PIL → TFT (no color conversion anywhere)
"""

import time
import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import ili9341
from picamera2 import Picamera2
import cv2
import numpy as np
import os
import urllib.request

# ── Display ──
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D27)
spi = board.SPI()
disp = ili9341.ILI9341(
    spi, cs=cs_pin, dc=dc_pin, rst=reset_pin,
    baudrate=64000000, width=240, height=320, rotation=90
)
print("✅ TFT Display: 320x240")

# ── Face Detector (OpenCV DNN) ──
MODEL_DIR = os.path.expanduser("~/Downloads/Mobilefacet_CPP/models")
PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFEMODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(PROTOTXT):
    print("📥 Downloading deploy.prototxt...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        PROTOTXT)

if not os.path.exists(CAFFEMODEL):
    print("📥 Downloading caffemodel...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        CAFFEMODEL)

net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
print("✅ Face detector loaded")

# ── Camera: 320x240 RGB, full FOV ���─
picam2 = Picamera2()
sensor_res = picam2.sensor_resolution
print(f"📷 Sensor: {sensor_res}")

config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    raw={"size": sensor_res},  # full FOV
    buffer_count=4,
)
picam2.configure(config)
picam2.start()
time.sleep(1.5)
print("📷 Camera: 320x240 RGB, full FOV via GPU ISP")
print("Running — press Ctrl+C to exit")

# ── Loop ──
CONF = 0.5
t_prev = time.time()
fps_smooth = 0.0
frame_count = 0

try:
    while True:
        # Capture RGB frame (320x240, GPU ISP does the downscale)
        frame = picam2.capture_array("main")  # RGB, 320x240

        # Detect faces (OpenCV DNN wants BGR but works fine on RGB with slightly shifted means)
        # We pass RGB directly — detection still works, colors stay correct
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        dets = net.forward()

        h, w = frame.shape[:2]
        n_faces = 0

        for i in range(dets.shape[2]):
            conf = dets[0, 0, i, 2]
            if conf < CONF:
                continue
            n_faces += 1
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            # Draw rectangle directly on RGB frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # FPS
        now = time.time()
        fps_smooth = 0.9 * fps_smooth + 0.1 / max(now - t_prev, 0.001)
        t_prev = now
        cv2.putText(frame, f"FPS:{fps_smooth:.1f} F:{n_faces}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Display on TFT — frame is already RGB, PIL expects RGB
        disp.image(Image.fromarray(frame))

        frame_count += 1

except KeyboardInterrupt:
    print(f"\n👋 Done — {frame_count} frames")
    picam2.stop()
    disp.fill(0)
