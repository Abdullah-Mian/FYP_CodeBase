import time
import board
import digitalio
from PIL import Image
from adafruit_rgb_display import ili9341
from picamera2 import Picamera2

# ---------------- CONFIGURATION ----------------
# Display Pins
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D27)
BAUDRATE = 64000000

# Camera Configuration for Pi Camera V2
# Option 1: 1080p (Full HD, best quality for video) - RECOMMENDED
CAMERA_WIDTH = 1920   # 1080p resolution
CAMERA_HEIGHT = 1080  # 16:9 aspect ratio

# Option 2: 720p (Good balance of speed and quality)
# CAMERA_WIDTH = 1280
# CAMERA_HEIGHT = 720

# Option 3: Maximum sensor resolution (3280x2464) - Very slow!
# CAMERA_WIDTH = 3280
# CAMERA_HEIGHT = 2464

# Display is fixed at 320x240 (physical limitation)
DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240

# ---------------- SETUP DISPLAY ----------------
spi = board.SPI()
disp = ili9341.ILI9341(
    spi,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
    width=240,
    height=320,
    rotation=90
)

print(f"Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
print(f"Camera capture: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

# ---------------- SETUP CAMERA ----------------
print("Initializing Camera...")
picam2 = Picamera2()

# Capture at HIGHER resolution to get full field of view
config = picam2.create_preview_configuration(
    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("Camera Running. Press Ctrl+C to exit.")
print("Full camera view will be scaled down to fit display")

# ---------------- MAIN LOOP ----------------
try:
    while True:
        # Capture full resolution frame
        frame = picam2.capture_array("main")
        
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Resize to fit display (this preserves full field of view!)
        # Using LANCZOS for best quality (can use BILINEAR for more speed)
        image_resized = image.resize(
            (DISPLAY_WIDTH, DISPLAY_HEIGHT), 
            Image.Resampling.LANCZOS
        )
        
        # Display on TFT
        disp.image(image_resized)

except KeyboardInterrupt:
    print("\nExiting...")
    picam2.stop()
    disp.fill(0)
except Exception as e:
    print(f"Error: {e}")
    picam2.stop()