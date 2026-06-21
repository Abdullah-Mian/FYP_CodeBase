import time
import spidev
import RPi.GPIO as GPIO
from PIL import Image, ImageDraw, ImageFont
import digitalio
import board
from adafruit_rgb_display import ili9341

# ---------------- CONFIGURATION ----------------
# Display Pins
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D27)
BAUDRATE = 24000000  # 24 MHz for Display

# Touch Pins & Calibration
TOUCH_CS_PIN = 7  # CE1 (GPIO 7)
TOUCH_IRQ_PIN = 24 # GPIO 24
# Calibration values (You might need to tweak these for your specific unit)
X_MIN, X_MAX = 300, 3800
Y_MIN, Y_MAX = 200, 3700

# ---------------- SETUP DISPLAY ----------------
spi = board.SPI()
disp = ili9341.ILI9341(
    spi,
    cs=cs_pin,
    dc=dc_pin,
    rst=reset_pin,
    baudrate=BAUDRATE,
    width=240,
    height=320, # This forces the display library to treat it as 240x320
    rotation=90 # Rotate 90 degrees for Landscape mode
)

# Create a blank image for drawing
if disp.rotation % 180 == 90:
    height = disp.width  # Swap height/width because we rotated
    width = disp.height
else:
    width = disp.width
    height = disp.height

image = Image.new("RGB", (width, height))
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

# ---------------- SETUP TOUCH (XPT2046) ----------------
# We use raw spidev for touch because it needs a different SPI speed/mode than the screen
touch_spi = spidev.SpiDev()
touch_spi.open(0, 1) # Bus 0, Device 1 (CE1)
touch_spi.max_speed_hz = 1000000 # 1MHz is plenty for touch

GPIO.setup(TOUCH_IRQ_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def read_touch():
    """Reads the XPT2046 Touch Controller"""
    # If IRQ is High, no touch is detected (it's active low)
    if GPIO.input(TOUCH_IRQ_PIN):
        return None
    
    # Send command to read X and Y
    # Command for X: 0xD0, Command for Y: 0x90
    try:
        # Read X
        resp_x = touch_spi.xfer2([0xD0, 0, 0])
        x_raw = ((resp_x[1] << 8) | resp_x[2]) >> 3
        
        # Read Y
        resp_y = touch_spi.xfer2([0x90, 0, 0])
        y_raw = ((resp_y[1] << 8) | resp_y[2]) >> 3

        # Simple mapping for Landscape mode (Rotation=90)
        # We need to map the raw 0-4096 values to screen pixels
        # Note: X/Y axes are often swapped or inverted on these panels vs the screen rotation
        
        # Mapping logic for standard Landscape:
        sx = int(((x_raw - X_MIN) / (X_MAX - X_MIN)) * width)
        sy = int(((y_raw - Y_MIN) / (Y_MAX - Y_MIN)) * height)
        
        # Clamp to screen bounds
        sx = max(0, min(width, sx))
        sy = max(0, min(height, sy))
        
        return (sx, sy)
    except Exception as e:
        print(e)
        return None

# ---------------- UI ELEMENTS ----------------
btn_x, btn_y = 60, 100
btn_w, btn_h = 200, 60
btn_color = (0, 255, 0) # Green
text_color = (255, 255, 255)

def draw_ui(message="Ready"):
    # 1. Clear screen (Fill black)
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    
    # 2. Draw Button
    draw.rectangle((btn_x, btn_y, btn_x + btn_w, btn_y + btn_h), fill=btn_color)
    draw.text((btn_x + 20, btn_y + 15), "CLICK ME", font=font, fill=(0, 0, 0))
    
    # 3. Draw Log Text
    draw.text((10, 200), f"Status: {message}", font=font, fill=text_color)
    
    # 4. Push to display
    disp.image(image)

# ---------------- MAIN LOOP ----------------
print("Starting Display...")
draw_ui()

last_touch_time = 0

try:
    while True:
        touch = read_touch()
        
        if touch:
            tx, ty = touch
            
            # Since touch panels vary wildly, I'll print the raw coords first 
            # so you can debug the calibration
            # print(f"Touched at: {tx}, {ty}") 

            # Check if touch is inside button box
            # Note: Touch coordinates might be inverted. If clicking the button doesn't work,
            # watch the printed coordinates and adjust the math in read_touch()
            if (btn_x <= tx <= btn_x + btn_w) and (btn_y <= ty <= btn_y + btn_h):
                current_time = time.time()
                if current_time - last_touch_time > 0.5: # Debounce (prevent double click)
                    print("[SERIAL LOG] Button was pressed!")
                    draw_ui(message="Button Pressed!")
                    last_touch_time = current_time
            else:
                # Touched outside button
                pass
                
        time.sleep(0.05)

except KeyboardInterrupt:
    # Clear screen on exit
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    disp.image(image)
    print("Exiting...")