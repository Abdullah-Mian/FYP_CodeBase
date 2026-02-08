#!/usr/bin/env python3
"""Quick test: what format does picamera2 actually give us?"""
from picamera2 import Picamera2
import numpy as np
import time

cam = Picamera2()

# Test BGR888
cam.configure(cam.create_preview_configuration(
    main={"size": (320, 240), "format": "BGR888"},
    raw={"size": (1640, 1232)},
))
cam.start()
time.sleep(1)
frame = cam.capture_array("main")
cam.stop()
print(f"BGR888 format: shape={frame.shape}, dtype={frame.dtype}")
print(f"  Pixel [100,100] = R:{frame[100,100,0]} G:{frame[100,100,1]} B:{frame[100,100,2]}")
print(f"  (If skin looks normal, R should be highest, ~150-200)")

time.sleep(0.5)

# Test RGB888
cam.configure(cam.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    raw={"size": (1640, 1232)},
))
cam.start()
time.sleep(1)
frame2 = cam.capture_array("main")
cam.stop()
print(f"\nRGB888 format: shape={frame2.shape}, dtype={frame2.dtype}")
print(f"  Pixel [100,100] = R:{frame2[100,100,0]} G:{frame2[100,100,1]} B:{frame2[100,100,2]}")
print(f"  (If skin looks normal, R should be highest, ~150-200)")

print(f"\nChannels same? {np.array_equal(frame, frame2)}")
print(f"Channels swapped? {np.array_equal(frame[:,:,0], frame2[:,:,2])}")
