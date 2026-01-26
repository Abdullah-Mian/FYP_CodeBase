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

# Display resolution (fixed by hardware)
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

# ---------------- SETUP CAMERA ----------------
print("Initializing Camera...")
picam2 = Picamera2()

# Get full sensor resolution for Camera Module V2
sensor_resolution = picam2.sensor_resolution
print(f"Full sensor resolution: {sensor_resolution}")

# CRITICAL FIX: Force full field of view by specifying raw stream
# This prevents libcamera from choosing a cropped sensor mode
# For Camera V2: sensor_resolution should be (3280, 2464)
config = picam2.create_preview_configuration(
    main={"size": (DISPLAY_WIDTH, DISPLAY_HEIGHT), "format": "RGB888"},
    raw={"size": sensor_resolution}  # This forces full FOV!
)

picam2.configure(config)

# Verify the sensor mode being used
print(f"Camera configuration:")
print(f"  Main output: {config['main']}")
print(f"  Raw sensor: {config['raw']}")

picam2.start()

print("Camera Running with FULL field of view. Press Ctrl+C to exit.")

# ---------------- MAIN LOOP ----------------
try:
    while True:
        # Capture frame
        frame = picam2.capture_array("main")
        
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Display on TFT
        disp.image(image)

except KeyboardInterrupt:
    print("\nExiting...")
    picam2.stop()
    disp.fill(0)
except Exception as e:
    print(f"Error: {e}")
    picam2.stop()