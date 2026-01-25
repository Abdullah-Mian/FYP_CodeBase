import time
import board
import digitalio
from PIL import Image
from adafruit_rgb_display import ili9341
from picamera2 import Picamera2  # FIX: Was "from picam2 import Picam2"

# ---------------- CONFIGURATION ----------------
# Display Pins
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D27)
BAUDRATE = 64000000

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

# Determine correct width/height based on rotation
if disp.rotation % 180 == 90:
    WIDTH = disp.height
    HEIGHT = disp.width
else:
    WIDTH = disp.width
    HEIGHT = disp.height

print(f"Display set to {WIDTH}x{HEIGHT}")

# ---------------- SETUP CAMERA ----------------
print("Initializing Camera...")
picam2 = Picamera2()

# Configure for speed - RGB888 format, exact screen size
config = picam2.create_preview_configuration(
    main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("Camera Running. Press Ctrl+C to exit.")

# ---------------- MAIN LOOP ----------------
try:
    while True:
        # Capture frame directly as numpy array
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