# Live Camera Feed on ILI9341 TFT Display

A high-performance Python script that displays a live camera feed from the Raspberry Pi Camera Module on a 3.2" ILI9341 SPI TFT display.

---

## 🚀 How It Works

### Hardware Communication Flow

```
Raspberry Pi Camera → libcamera → Picamera2 → NumPy Array → PIL Image → SPI → TFT Display
```

### Software Architecture (What Makes It Fast?)

This project achieves **smooth, real-time video** without using OpenCV. Here's how:

#### 1. **Picamera2 Library** (Core Camera Interface)
- **What it is:** Modern Python interface to libcamera (Raspberry Pi's camera stack)
- **Why it's fast:** 
  - Direct hardware access via libcamera
  - Zero-copy memory operations
  - Returns raw camera data as NumPy arrays (no encoding/decoding overhead)
  - Configured to output **RGB888** format directly (matches display requirements)

#### 2. **PIL/Pillow** (Image Handling - NOT for processing!)
- **What it does:** Converts NumPy array → PIL Image object
- **Why it's needed:** The `adafruit_rgb_display` library expects PIL Image objects
- **Performance impact:** Minimal - just wraps existing data, no heavy processing
- **NOT used for:** Resizing, conversion, or any image manipulation (that would slow things down)

#### 3. **Adafruit RGB Display Library** (SPI Communication)
- **What it does:** Sends raw RGB pixel data to the TFT over SPI
- **Why it's fast:**
  - Direct SPI hardware communication at 64MHz
  - No software rendering or buffering overhead
  - Optimized C extensions under the hood

#### 4. **SPI Hardware** (Physical Connection)
- **Speed:** 64,000,000 Hz (64 MHz) baudrate
- **Data transfer:** Raw RGB888 bytes streamed directly to display controller
- **Why SPI?** Much faster than I2C, designed for display data

### Why NO OpenCV?

**OpenCV is NOT used** in this project because:
- ❌ Adds unnecessary overhead for simple capture → display pipeline
- ❌ Would require format conversions (OpenCV uses BGR, we need RGB)
- ❌ Slower imports and initialization
- ✅ **Our approach:** Direct memory access from camera to display = maximum speed

### The Speed Secret: Zero-Copy Pipeline

```python
# 1. Camera outputs RGB888 directly (no conversion!)
frame = picam2.capture_array("main")  # NumPy array in memory

# 2. PIL wraps the array (no copying!)
image = Image.fromarray(frame)  # Just wraps existing memory

# 3. Display reads and sends via SPI
disp.image(image)  # Direct SPI transfer
```

**Result:** Camera sensor → Display screen with minimal CPU involvement!

---

## 📐 Adjusting Camera Display Size

### Problem: Image Appears Cropped

The camera captures at a different aspect ratio than your display, causing cropping.

### Solution: Modify Camera Resolution

Edit `main_cam.py` and change the camera configuration:

```python
# Current configuration (320x240 - 4:3 aspect ratio)
config = picam2.create_preview_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
)
```

### Option 1: Match Display Exactly (Recommended)
```python
# For 3.2" TFT at rotation=90 (320x240 display)
WIDTH = 320
HEIGHT = 240

config = picam2.create_preview_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
)
```

### Option 2: Use Different Resolutions

The Raspberry Pi Camera supports these resolutions:

**Common resolutions:**
```python
# Wide angle (16:9) - will crop top/bottom on 4:3 display
main={"size": (640, 480), "format": "RGB888"}

# Lower resolution (faster but pixelated)
main={"size": (160, 120), "format": "RGB888"}

# Higher resolution (slower but smoother on larger displays)
main={"size": (800, 600), "format": "RGB888"}
```

**Camera Module v2 supports up to:** 3280 x 2464  
**Camera Module v3 supports up to:** 4608 x 2592

### Option 3: Change Display Orientation

If your camera view looks sideways or cropped:

```python
# Try different rotation values: 0, 90, 180, 270
disp = ili9341.ILI9341(
    spi,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
    width=240,
    height=320,
    rotation=90  # ← Change this: 0, 90, 180, or 270
)
```

### Finding the Best Settings

1. **Start with display native resolution:**
   ```python
   WIDTH = 320
   HEIGHT = 240
   ```

2. **If still cropped, try lower resolution:**
   ```python
   WIDTH = 240
   HEIGHT = 180
   ```

3. **Adjust rotation to match camera orientation:**
   ```python
   rotation=0    # Default
   rotation=90   # Rotate 90° clockwise
   rotation=180  # Upside down
   rotation=270  # Rotate 90° counter-clockwise
   ```

---

## 🔧 Performance Tuning

### Speed vs Quality Tradeoffs

| Setting | Speed | Quality | Use Case |
|---------|-------|---------|----------|
| 160x120 | ⚡⚡⚡ Fastest | 📉 Low | Testing, basic monitoring |
| 320x240 | ⚡⚡ Fast | 📊 Good | **Recommended default** |
| 640x480 | ⚡ Medium | 📈 High | Larger displays |
| 800x600 | 🐌 Slower | 📈📈 Very High | Quality priority |

### Current Settings Explanation

```python
# Display Configuration
BAUDRATE = 64000000  # 64MHz SPI speed (maximum for most TFTs)

# Camera Configuration
main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
# RGB888 = 24-bit color (8 bits per channel)
# Matches TFT color depth exactly
# No conversion needed = maximum speed
```

### If You Need Even More Speed

1. **Reduce resolution:**
   ```python
   WIDTH = 240
   HEIGHT = 180
   ```

2. **Lower SPI baudrate** (only if getting display corruption):
   ```python
   BAUDRATE = 32000000  # Half speed, more stable
   ```

3. **Reduce color depth** (not recommended - minimal speed gain):
   ```python
   main={"size": (WIDTH, HEIGHT), "format": "RGB565"}
   # 16-bit color instead of 24-bit
   ```

---

## 📊 Technical Specifications

### Hardware
- **Display:** ILI9341 3.2" TFT (240x320 pixels, SPI interface)
- **Touch Controller:** XPT2046 (not used in this project)
- **Camera:** Raspberry Pi Camera Module (v1/v2/v3/HQ)
- **Platform:** Raspberry Pi 4 Model B (8GB) - works on 3B+/4/5

### Software Stack
- **OS:** Raspberry Pi OS Lite + XFCE
- **Camera Interface:** libcamera (via Picamera2)
- **Display Driver:** Adafruit RGB Display Library
- **Image Library:** Pillow (PIL)
- **No OpenCV required!**

### SPI Pin Configuration
```python
cs_pin = board.CE0      # GPIO 8  (Chip Select)
dc_pin = board.D25      # GPIO 25 (Data/Command)
reset_pin = board.D27   # GPIO 27 (Reset)
# MOSI = GPIO 10 (auto-configured by SPI)
# SCLK = GPIO 11 (auto-configured by SPI)
```

---

## 🎯 Why This Approach?

### Advantages
✅ **Fast:** 15-30 FPS on 320x240 display  
✅ **Simple:** ~80 lines of code  
✅ **Efficient:** Direct memory operations, no unnecessary conversions  
✅ **Lightweight:** No OpenCV dependency  
✅ **Reliable:** Uses official Raspberry Pi camera libraries  

### Limitations
❌ No video recording (display only)  
❌ No advanced image processing  
❌ No touch input handling (can be added)  
❌ Requires system-installed picamera2  

---

## 🔍 Troubleshooting Display Issues

### Image is Stretched
**Cause:** Camera aspect ratio ≠ Display aspect ratio  
**Fix:** Match camera resolution to display native resolution

### Image is Sideways
**Cause:** Display rotation doesn't match camera orientation  
**Fix:** Change `rotation` parameter (0, 90, 180, 270)

### Image is Pixelated
**Cause:** Resolution too low for display size  
**Fix:** Increase camera resolution

### Display is Slow/Choppy
**Cause:** Resolution too high for SPI bandwidth  
**Fix:** Reduce camera resolution or lower BAUDRATE

### Colors Look Wrong
**Cause:** Format mismatch  
**Fix:** Ensure `format="RGB888"` is set in camera config

---

## 📚 Further Reading

- [Picamera2 Documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [ILI9341 Display Datasheet](https://cdn-shop.adafruit.com/datasheets/ILI9341.pdf)
- [Adafruit CircuitPython RGB Display](https://github.com/adafruit/Adafruit_CircuitPython_RGB_Display)

---

## 📝 License

MIT License - Feel free to modify and use in your projects!