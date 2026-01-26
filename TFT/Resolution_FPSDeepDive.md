# Resolution, FPS, and ISP Configuration Deep Dive

## Your Critical Questions Answered

### Q1: What resolution does `qcam` actually display?

**Answer:** qcam displays at 800×600 pixels by default, NOT 320×240!

When you run `qcam`, it:
1. Requests **800×600** output from libcamera
2. libcamera automatically selects an appropriate sensor mode
3. ISP scales sensor data → 800×600
4. Display shows 800×600 on your monitor

**This is why qcam looks bigger and shows more field of view than your initial 320×240 TFT display!**

### Q2: Who told the ISP to convert 3280×2464 to 320×240?

**Answer:** **YOU DID!** In this line:

```python
main={"size": (320, 240), "format": "RGB888"}
```

This is a **request to libcamera**, which then:
1. Configures the ISP hardware to output exactly 320×240
2. The ISP automatically scales from sensor resolution → requested resolution
3. You receive 320×240 RGB888 data ready to use

**The ISP doesn't have a "standard" - it outputs whatever you request!**

### Q3: Is 30 FPS a camera limit or ISP standard?

**Answer:** It's **neither fixed nor standard** - it depends on:
- Sensor mode selected
- Resolution requested
- ISP processing load
- Memory bandwidth available

**Let me show you the actual limits...**

---

## Camera Module V2 (IMX219) Sensor Modes & Framerates

Your camera supports **multiple modes** with different FPS limits:

```
┌──────┬───────────────┬──────────────┬──────────┬────────────┐
│ Mode │ Resolution    │ Binning/Crop │ Max FPS  │ FOV        │
├──────┼───────────────┼──────────────┼──────────┼────────────┤
│  0   │ 3280 × 2464   │ None         │ ~15 fps  │ Full (100%)│
│  1   │ 1920 × 1080   │ 2×2 binned   │ ~30 fps  │ Cropped    │
│  2   │ 1640 × 1232   │ 2×2 binned   │ ~40 fps  │ Full (100%)│
│  3   │ 640 × 480     │ 4×4 binned   │ ~90 fps  │ Full (100%)│
└──────┴───────────────┴──────────────┴──────────┴────────────┘
```

**Key insight:** Smaller sensor modes can run MUCH faster!

---

## How libcamera Selects Sensor Mode

### Without `raw` parameter (Automatic selection):

```python
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"}
)
```

**What libcamera does:**
1. User wants 320×240 output
2. Find sensor mode "close enough" to 320×240
3. Mode 3 (640×480) seems closest → select it
4. **BUT Mode 1 (1920×1080) might be chosen if libcamera thinks it's better**
5. If Mode 1 chosen → **CROPPED field of view!**

### With `raw` parameter (Forced sensor mode):

```python
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    raw={"size": (3280, 2464)}  # Force Mode 0!
)
```

**What libcamera does:**
1. User explicitly requests Mode 0 (3280×2464)
2. Select Mode 0 → **FULL field of view**
3. ISP scales 3280×2464 → 320×240
4. You get full FOV at 320×240!

**This is why adding `raw` parameter fixed your cropping issue!**

---

## ISP Configuration Process (Step-by-Step)

### Step 1: You call `create_preview_configuration()`
```python
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    raw={"size": (3280, 2464)}
)
```

### Step 2: Picamera2 translates this to libcamera API
```cpp
// Simplified C++ equivalent
StreamConfiguration main_stream;
main_stream.size = Size(320, 240);
main_stream.pixelFormat = PixelFormat::RGB888;

StreamConfiguration raw_stream;
raw_stream.size = Size(3280, 2464);  // Forces sensor mode selection
```

### Step 3: libcamera analyzes request and selects sensor mode
```
Requested output: 320×240
Requested raw: 3280×2464

Sensor mode selection algorithm:
├─ Check sensor modes that support 3280×2464 output
├─ Mode 0: 3280×2464 ✓ MATCH!
└─ Select Mode 0
```

### Step 4: libcamera configures ISP pipeline
```
ISP Pipeline Configuration:
├─ Input: Mode 0 sensor (3280×2464 Bayer)
├─ ISP Processing:
│   ├─ Black level correction
│   ├─ Demosaic (Bayer → RGB)
│   ├─ White balance
│   ├─ Color correction
│   ├─ ** SCALER: 3280×2464 → 320×240 **  ← ISP SCALING HERE!
│   └─ Format conversion → RGB888
└─ Output: 320×240 RGB888
```

### Step 5: ISP registers are programmed
```
Hardware register writes (simplified):
ISP_INPUT_WIDTH = 3280
ISP_INPUT_HEIGHT = 2464
ISP_OUTPUT_WIDTH = 320         ← This is what you specified!
ISP_OUTPUT_HEIGHT = 240        ← This is what you specified!
ISP_SCALER_ENABLE = 1
ISP_SCALER_RATIO_H = 10.25     ← Calculated: 3280/320
ISP_SCALER_RATIO_V = 10.27     ← Calculated: 2464/240
ISP_OUTPUT_FORMAT = RGB888
```

**The ISP hardware scaler reads these registers and automatically performs the scaling!**

---

## Framerate Determination

### Default Framerate (No specification)

When you don't specify framerate:
```python
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    raw={"size": (3280, 2464)}
)
```

**What happens:**
1. Sensor Mode 0 (3280×2464) maximum is ~15 fps
2. **BUT** because ISP only outputs 320×240, it can run faster!
3. libcamera uses **default framerate for that mode** ≈ 30 fps
4. ISP can easily keep up with 30 fps at 320×240 output

### Where does 30 FPS come from?

The default framerate for libcamera video applications is 30 fps. This is a **software default**, not a hardware limit!

**Actual limits depend on:**
```
Component          | Limit for 320×240 output
───────────────────┼─────────────────────────
Sensor Mode 0      | ~15 fps (full resolution)
ISP Processing     | >100 fps (hardware can handle this easily)
SPI Transfer       | ~34 fps (29ms transfer time)
Python overhead    | >60 fps (minimal CPU usage)
───────────────────┼─────────────────────────
ACTUAL BOTTLENECK  | Sensor Mode 0 at ~15 fps
```

**BUT** if you use a faster sensor mode:
```python
# Using Mode 3 (640×480) instead
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    raw={"size": (640, 480)}  # Mode 3: up to 90 fps!
)
```

Now you could theoretically get **90 fps** because:
- Sensor Mode 3 supports 90 fps
- ISP easily handles 640×480 → 320×240 scaling at 90 fps
- SPI becomes the new bottleneck at ~34 fps

---

## Changing Resolution and FPS Using Raw Parameter

### YES! You can control both resolution and FPS:

```python
from picamera2 import Picamera2
import time

picam2 = Picamera2()

# Example 1: Maximum quality, slower FPS (~15 fps)
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    raw={"size": (3280, 2464)}  # Mode 0: Full FOV, ~15 fps
)

# Example 2: Balanced quality and speed (~40 fps)
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    raw={"size": (1640, 1232)}  # Mode 2: Full FOV, ~40 fps
)

# Example 3: Maximum speed (~90 fps potential)
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    raw={"size": (640, 480)}   # Mode 3: Full FOV, ~90 fps
)

# Example 4: Explicitly set framerate
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "RGB888"},
    raw={"size": (1640, 1232)},
    controls={"FrameRate": 60}  # Request 60 fps (if sensor mode supports it)
)
```

### Understanding the Tradeoffs:

```
Raw Size        │ Sensor Mode │ Max FPS │ Quality    │ FOV
────────────────┼─────────────┼─────────┼────────────┼──────
3280×2464       │ Mode 0      │ ~15     │ Best       │ Full
1640×1232       │ Mode 2      │ ~40     │ Good       │ Full
640×480         │ Mode 3      │ ~90     │ Acceptable │ Full
1920×1080       │ Mode 1      │ ~30     │ Good       │ CROPPED!
```

---

## Complete Example: Configurable Resolution & FPS

```python
import time
import board
import digitalio
from PIL import Image
from adafruit_rgb_display import ili9341
from picamera2 import Picamera2

# Camera mode presets for easy switching
CAMERA_MODES = {
    "max_quality": (3280, 2464, 15),   # Full res, 15 fps
    "balanced": (1640, 1232, 40),       # Good res, 40 fps  
    "max_speed": (640, 480, 90),        # Lower res, 90 fps
    "custom": (1920, 1080, 30)          # Custom setting
}

# Select your preferred mode
MODE = "balanced"  # Change this!
raw_width, raw_height, target_fps = CAMERA_MODES[MODE]

# Display setup
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D27)

spi = board.SPI()
disp = ili9341.ILI9341(
    spi, cs=cs_pin, dc=dc_pin, rst=reset_pin,
    baudrate=64000000, width=240, height=320, rotation=90
)

DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240

# Camera setup with configurable mode
print(f"Initializing camera in '{MODE}' mode...")
print(f"Raw sensor: {raw_width}×{raw_height} @ {target_fps} fps target")

picam2 = Picamera2()

# Configure with selected mode
config = picam2.create_preview_configuration(
    main={"size": (DISPLAY_WIDTH, DISPLAY_HEIGHT), "format": "RGB888"},
    raw={"size": (raw_width, raw_height)},
    controls={"FrameRate": target_fps}
)

picam2.configure(config)
print(f"Configuration: {config}")

picam2.start()
print("Camera running. Press Ctrl+C to exit.")

# FPS counter
frame_count = 0
start_time = time.time()

try:
    while True:
        frame = picam2.capture_array("main")
        image = Image.fromarray(frame)
        disp.image(image)
        
        # Calculate actual FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed
            print(f"Actual FPS: {actual_fps:.1f}")

except KeyboardInterrupt:
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    print(f"\nExiting... Average FPS: {avg_fps:.1f}")
    picam2.stop()
    disp.fill(0)
```

---

## Key Insights Summary

### 1. **qcam vs Your Script**
- qcam: 800×600 display (larger window)
- Your script: 320×240 display (TFT size)
- **Different output resolutions, but both can use same sensor mode!**

### 2. **ISP Configuration**
- ISP has **NO fixed output resolution**
- It outputs **whatever you request** in `main={"size": (W, H)}`
- The `raw` parameter **forces sensor mode selection**
- ISP automatically calculates scaling ratio

### 3. **FPS Control**
- 30 fps is a **software default**, not hardware limit
- Actual FPS depends on **sensor mode** selected via `raw` parameter
- You can request specific FPS via `controls={"FrameRate": X}`
- Smaller sensor modes → higher FPS capability

### 4. **The Magic of `raw` Parameter**
```python
raw={"size": (3280, 2464)}  # This is NOT the output size!
                            # This FORCES sensor Mode 0 selection
                            # Guarantees FULL field of view
                            # ISP then scales to your main output size
```

### 5. **Why No Manual Resize Needed**
```
Without raw (WRONG):
Sensor Mode 1 (cropped) → ISP → 320×240 output
└─ Cropped FOV even though ISP scaled it!

With raw (CORRECT):  
Sensor Mode 0 (full) → ISP hardware scale → 320×240 output
└─ Full FOV because sensor captured full area!
```

---

## Advanced: Finding Available Sensor Modes

Add this to your script to see all available modes:

```python
from picamera2 import Picamera2

picam2 = Picamera2()
sensor_modes = picam2.sensor_modes

print("Available sensor modes:")
for i, mode in enumerate(sensor_modes):
    print(f"Mode {i}: {mode}")
```

**Example output for IMX219:**
```
Mode 0: {'bit_depth': 10, 'size': (3280, 2464), 'fps': 15.0, 'crop_limits': (0, 0, 3280, 2464), 'exposure_limits': (31, 667234)}
Mode 1: {'bit_depth': 10, 'size': (1920, 1080), 'fps': 30.0, 'crop_limits': (680, 692, 1920, 1080), 'exposure_limits': (31, 358696)}
Mode 2: {'bit_depth': 10, 'size': (1640, 1232), 'fps': 40.0, 'crop_limits': (0, 0, 3280, 2464), 'exposure_limits': (31, 269503)}
Mode 3: {'bit_depth': 10, 'size': (640, 480), 'fps': 90.0, 'crop_limits': (1000, 752, 1280, 960), 'exposure_limits': (31, 116526)}
```

Notice the `crop_limits` - Mode 1 is cropped, others use full sensor!

---

## For Your Project Defense

**Question:** "How does the system know to output 320×240?"

**Answer:** "I explicitly configured the ISP through the libcamera API by specifying `main={"size": (320, 240)}`. This writes the output dimensions to the ISP hardware registers, configuring its internal scaler to produce exactly 320×240 RGB888 frames. The `raw` parameter ensures we use the full sensor area (Mode 0: 3280×2464) as input, and the ISP automatically calculates the scaling ratio (10.25× horizontal, 10.27× vertical) to produce the requested output size while maintaining the full field of view."

**Question:** "Why is it 30 FPS?"

**Answer:** "The 30 FPS comes from the libcamera software default framerate setting, not a hardware limitation. Our sensor (IMX219) in Mode 0 can run at up to 15 fps at full 3280×2464 resolution. However, since the ISP only outputs 320×240, processing is much lighter, allowing the pipeline to run at the default 30 fps. We can increase this by selecting a faster sensor mode (like Mode 2 at 1640×1232 supporting 40 fps) or decrease it by explicitly requesting a lower framerate through the controls parameter. The actual limiting factor in our system is the SPI transfer to the display at ~34 fps maximum."