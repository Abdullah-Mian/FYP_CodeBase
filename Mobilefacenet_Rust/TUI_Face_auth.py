#!/usr/bin/env python3
"""
Raspberry Pi 4 Face Recognition System
Picamera2 + BlazeFace + MobileFaceNet (Rust) + ILI9341 TFT

Terminal UI powered by Rich
"""

import time
import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import ili9341
import cv2
import numpy as np
import os
import pickle
import sys
import select

# Rich UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich import box

# Picamera2
try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ Error: Picamera2 not found!")
    sys.exit(1)

# MediaPipe (BlazeFace)
try:
    import mediapipe as mp
except ImportError:
    print("❌ Error: MediaPipe not found!")
    sys.exit(1)

# Rust inference engine
try:
    from rust_wrapper import get_face_embedding, compute_similarity
except ImportError:
    print("❌ Error: rust_wrapper.py not found!")
    sys.exit(1)

# ========================================
# Configuration
# ========================================
SIMILARITY_THRESHOLD = 0.4
DATABASE_FILE = 'faces_db.pkl'

# Display
DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240
BAUDRATE = 64000000

# Camera (IMX219)
PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 240
CAPTURE_WIDTH = 3280
CAPTURE_HEIGHT = 2464

# Detection
MIN_DETECTION_CONFIDENCE = 0.5

# ========================================
# Rich Console
# ========================================
console = Console()

def print_banner():
    """Print startup banner"""
    banner_text = Text()
    banner_text.append("╔══════════════════════════════════════════════════╗\n", style="bold cyan")
    banner_text.append("║", style="bold cyan")
    banner_text.append("        FACE RECOGNITION SYSTEM v2.0          ", style="bold white")
    banner_text.append("║\n", style="bold cyan")
    banner_text.append("║", style="bold cyan")
    banner_text.append("   Raspberry Pi 4  •  MobileFaceNet  •  Rust  ", style="dim white")
    banner_text.append("║\n", style="bold cyan")
    banner_text.append("╚══════════════════════════════════════════════════╝", style="bold cyan")
    console.print(banner_text)

def print_system_info(cam_res, disp_res, db_count):
    """Print system status table"""
    table = Table(
        title="[bold cyan]SYSTEM STATUS[/]",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
        title_style="bold cyan",
        show_header=False,
        padding=(0, 1)
    )
    table.add_column("Component", style="dim white", width=20)
    table.add_column("Status", style="bold green", width=30)

    table.add_row("🎥 Camera", f"[green]IMX219 @ {cam_res}[/]")
    table.add_row("🖥️  Display", f"[green]ILI9341 @ {disp_res}[/]")
    table.add_row("🦀 Engine", "[green]Rust + tract-onnx[/]")
    table.add_row("🧠 Detector", "[green]BlazeFace (MediaPipe)[/]")
    table.add_row("💾 Database", f"[green]{db_count} face(s) enrolled[/]")

    console.print(table)

def print_controls():
    """Print control panel"""
    ctrl = Table(
        title="[bold yellow]CONTROLS[/]",
        box=box.ROUNDED,
        border_style="yellow",
        show_header=False,
        padding=(0, 1)
    )
    ctrl.add_column("Key", style="bold cyan", width=8, justify="center")
    ctrl.add_column("Action", style="white", width=30)

    ctrl.add_row("[bold cyan]a[/]", "Add new face to database")
    ctrl.add_row("[bold cyan]v[/]", "Verify face against database")
    ctrl.add_row("[bold cyan]l[/]", "List all registered faces")
    ctrl.add_row("[bold cyan]d[/]", "Delete a face from database")
    ctrl.add_row("[bold red]q[/]", "Quit application")

    console.print(ctrl)

def print_access_granted(name, score):
    """Print access granted panel"""
    content = Text()
    content.append("██████████████████████████████████████\n", style="bold green")
    content.append("█                                    █\n", style="bold green")
    content.append("█", style="bold green")
    content.append("        ACCESS  GRANTED             ", style="bold white on green")
    content.append("█\n", style="bold green")
    content.append("█                                    █\n", style="bold green")
    content.append("██████████████████████████████████████\n", style="bold green")
    content.append(f"\n  👤 Identity:   {name}\n", style="bold white")
    content.append(f"  📊 Confidence: {score:.3f}\n", style="bold green")
    content.append(f"  🔑 Threshold:  {SIMILARITY_THRESHOLD}\n", style="dim white")

    console.print(Panel(
        content,
        border_style="bold green",
        box=box.DOUBLE,
        title="[bold green]✅ VERIFIED[/]",
        subtitle=f"[dim]{time.strftime('%H:%M:%S')}[/]"
    ))

def print_access_denied(name, score):
    """Print access denied panel"""
    content = Text()
    content.append("██████████████████████████████████████\n", style="bold red")
    content.append("█                                    █\n", style="bold red")
    content.append("█", style="bold red")
    content.append("        ACCESS  DENIED              ", style="bold white on red")
    content.append("█\n", style="bold red")
    content.append("█                                    █\n", style="bold red")
    content.append("██████████████████████████████████████\n", style="bold red")
    content.append(f"\n  👤 Closest:    {name}\n", style="bold white")
    content.append(f"  📊 Score:      {score:.3f}\n", style="bold red")
    content.append(f"  🔑 Threshold:  {SIMILARITY_THRESHOLD}\n", style="dim white")

    console.print(Panel(
        content,
        border_style="bold red",
        box=box.DOUBLE,
        title="[bold red]❌ REJECTED[/]",
        subtitle=f"[dim]{time.strftime('%H:%M:%S')}[/]"
    ))

def print_comparison_table(scores, threshold):
    """Print face comparison results table"""
    table = Table(
        title="[bold magenta]🔍 DATABASE COMPARISON[/]",
        box=box.SIMPLE_HEAVY,
        border_style="magenta",
        padding=(0, 1)
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Name", style="bold white", width=20)
    table.add_column("Score", width=10, justify="center")
    table.add_column("Status", width=12, justify="center")

    for i, (name, score) in enumerate(scores, 1):
        if score > threshold:
            score_style = "bold green"
            status = "[bold green]✓ MATCH[/]"
        elif score > threshold * 0.8:
            score_style = "yellow"
            status = "[yellow]~ CLOSE[/]"
        else:
            score_style = "dim red"
            status = "[dim red]✗ NO[/]"

        table.add_row(str(i), name, f"[{score_style}]{score:.3f}[/]", status)

    console.print(table)

def print_face_list(db):
    """Print registered faces table"""
    table = Table(
        title="[bold cyan]📋 REGISTERED FACES[/]",
        box=box.ROUNDED,
        border_style="cyan",
        padding=(0, 1)
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Name", style="bold white", width=25)
    table.add_column("Enrolled", style="dim", width=20)

    if not db:
        console.print(Panel(
            "[dim]No faces registered yet[/]",
            border_style="dim",
            title="[dim]Empty Database[/]"
        ))
        return

    for i, name in enumerate(db.keys(), 1):
        table.add_row(str(i), name, "[dim]Active[/]")

    table.add_section()
    table.add_row("", f"[bold cyan]Total: {len(db)}[/]", "")

    console.print(table)

def print_processing(message, style="magenta"):
    """Print processing status"""
    console.print(f"  [bold {style}]⟫[/] {message}")

def print_section(title):
    """Print section header"""
    console.print()
    console.rule(f"[bold cyan]{title}[/]", style="cyan")

# ========================================
# TFT Display
# ========================================
class TFTDisplay:
    """ILI9341 TFT Display - Full Screen"""

    def __init__(self):
        cs_pin = digitalio.DigitalInOut(board.CE0)
        dc_pin = digitalio.DigitalInOut(board.D25)
        reset_pin = digitalio.DigitalInOut(board.D27)

        spi = board.SPI()
        self.disp = ili9341.ILI9341(
            spi, cs=cs_pin, dc=dc_pin, rst=reset_pin,
            baudrate=BAUDRATE, width=240, height=320, rotation=90
        )

        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10
            )
        except Exception:
            self.font = ImageFont.load_default()

        print_processing("TFT Display initialized: 320×240", "green")

    def show_frame(self, frame, faces=None, status=None, fps=0):
        if faces is None:
            faces = []

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if status or fps > 0:
            text = f"FPS:{fps:.0f}"
            if status:
                text += f" | {status}"
            cv2.putText(
                frame_bgr, text, (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

        # ILI9341 BGR quirk
        pil_img = Image.fromarray(frame_bgr, mode='RGB')
        self.disp.image(pil_img)

    def clear(self):
        self.disp.image(
            Image.new("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0))
        )

# ========================================
# Camera
# ========================================
class PiCamera:
    """Dual-resolution camera"""

    def __init__(self):
        self.picam2 = None
        self.preview_config = None
        self.capture_config = None

    def start(self):
        try:
            self.picam2 = Picamera2()

            self.preview_config = self.picam2.create_preview_configuration(
                main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"},
                raw={"size": (1640, 1232)}
            )

            self.capture_config = self.picam2.create_still_configuration(
                main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
            )

            self.picam2.configure(self.preview_config)
            self.picam2.start()
            time.sleep(2)

            print_processing(
                f"Camera online: Preview {PREVIEW_WIDTH}×{PREVIEW_HEIGHT}, "
                f"Capture {CAPTURE_WIDTH}×{CAPTURE_HEIGHT}",
                "green"
            )
            return True

        except Exception as e:
            console.print(f"  [bold red]✗[/] Camera failed: {e}")
            return False

    def read_preview(self):
        try:
            return True, self.picam2.capture_array()
        except Exception:
            return False, None

    def capture_high_res(self):
        try:
            print_processing(f"Switching to {CAPTURE_WIDTH}×{CAPTURE_HEIGHT}...")

            self.picam2.stop()
            self.picam2.configure(self.capture_config)
            self.picam2.start()
            time.sleep(0.8)

            frame = self.picam2.capture_array()

            self.picam2.stop()
            self.picam2.configure(self.preview_config)
            self.picam2.start()
            time.sleep(0.5)

            print_processing(
                f"Captured {frame.shape[1]}×{frame.shape[0]} image",
                "green"
            )
            return frame

        except Exception as e:
            console.print(f"  [bold red]✗[/] Capture failed: {e}")
            return None

    def stop(self):
        if self.picam2:
            self.picam2.stop()

# ========================================
# BlazeFace Detector
# ========================================
class FaceDetector:
    """MediaPipe BlazeFace"""

    def __init__(self):
        self.mp_fd = mp.solutions.face_detection
        self.detector = self.mp_fd.FaceDetection(
            model_selection=0,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE
        )

    def detect(self, frame):
        results = self.detector.process(frame)
        if not results.detections:
            return []

        h, w, _ = frame.shape
        faces = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            bx = max(0, int(bb.xmin * w))
            by = max(0, int(bb.ymin * h))
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            faces.append((bx, by, bw, bh))
        return faces

    def close(self):
        self.detector.close()

# ========================================
# Database
# ========================================
def load_database():
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            console.print("  [yellow]⚠ Database corrupted, starting fresh[/]")
    return {}


def save_database(db):
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(db, f)

# ========================================
# Face Capture Pipeline
# ========================================
def capture_face_embedding(camera, detector, display):
    """Detect → Capture high-res → Extract embedding"""
    console.print("\n  [bold magenta]📸 Position your face... Auto-capture in 3s[/]")

    countdown = 3
    start_time = time.time()

    while countdown > 0:
        ret, frame = camera.read_preview()
        if not ret:
            continue

        faces = detector.detect(frame)

        if len(faces) > 0:
            elapsed = time.time() - start_time
            countdown = 3 - int(elapsed)
            if countdown < 0:
                countdown = 0
            status = f"CAPTURE IN {countdown}..." if countdown > 0 else "PROCESSING..."
        else:
            status = "NO FACE"
            start_time = time.time()
            countdown = 3

        display.show_frame(frame, faces=faces, status=status)

        if len(faces) > 0 and countdown <= 0:
            hires_frame = camera.capture_high_res()
            if hires_frame is None:
                return None

            hires_faces = detector.detect(hires_frame)
            if len(hires_faces) == 0:
                console.print("  [yellow]⚠ Face lost in high-res capture[/]")
                return None

            (x, y, w, h) = max(hires_faces, key=lambda f: f[2] * f[3])
            margin = int(max(w, h) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(CAPTURE_WIDTH, x + w + margin)
            y2 = min(CAPTURE_HEIGHT, y + h + margin)

            face_crop = hires_frame[y1:y2, x1:x2]
            print_processing(
                f"Face crop: {face_crop.shape[1]}×{face_crop.shape[0]}"
            )

            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            temp_file = "temp_capture.jpg"
            cv2.imwrite(temp_file, face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            print_processing("Rust inference engine processing...")
            embedding = get_face_embedding(temp_file)

            if os.path.exists(temp_file):
                os.remove(temp_file)

            if embedding is not None:
                print_processing("Embedding extracted!", "green")
                return embedding
            else:
                console.print("  [bold red]✗[/] Rust engine failed")
                return None

    return None

# ========================================
# CLI Input
# ========================================
def non_blocking_input():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

# ========================================
# Main
# ========================================
def main():
    console.clear()
    print_banner()

    # Initialize
    print_section("INITIALIZING")

    display = TFTDisplay()
    camera = PiCamera()

    if not camera.start():
        sys.exit(1)

    detector = FaceDetector()
    print_processing("BlazeFace detector loaded", "green")

    db = load_database()
    print_processing(f"Database loaded: {len(db)} face(s)", "green")

    # System info
    print_system_info(
        f"{CAPTURE_WIDTH}×{CAPTURE_HEIGHT}",
        f"{DISPLAY_WIDTH}×{DISPLAY_HEIGHT}",
        len(db)
    )

    print_controls()
    console.print("\n[dim]Waiting for input...[/]\n")

    frame_count = 0
    fps_start = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = camera.read_preview()
            if not ret:
                continue

            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30.0 / (time.time() - fps_start)
                fps_start = time.time()

            faces = detector.detect(frame)
            display.show_frame(frame, faces=faces, status="Ready", fps=fps)

            cmd = non_blocking_input()

            # ── ADD FACE ──────────────────────────────────
            if cmd == 'a':
                print_section("ADD FACE")

                embedding = capture_face_embedding(camera, detector, display)
                if embedding is not None:
                    name = console.input("\n  [bold cyan]👤 Enter name:[/] ").strip()
                    if name:
                        db[name] = embedding
                        save_database(db)
                        console.print(
                            Panel(
                                f"[bold green]✨ Registered: {name}[/]\n"
                                f"[dim]Database now has {len(db)} face(s)[/]",
                                border_style="green",
                                box=box.ROUNDED
                            )
                        )
                    else:
                        console.print("  [yellow]⚠ No name entered, discarded[/]")
                console.print()

            # ── VERIFY FACE ───────────────────────────────
            elif cmd == 'v':
                print_section("VERIFY FACE")

                if not db:
                    console.print(
                        Panel(
                            "[yellow]Database is empty — add faces first[/]",
                            border_style="yellow",
                            box=box.ROUNDED
                        )
                    )
                    continue

                embedding = capture_face_embedding(camera, detector, display)
                if embedding is not None:
                    scores = []
                    best_score, best_name = -1.0, "Unknown"

                    for name, stored in db.items():
                        score = compute_similarity(embedding, stored)
                        scores.append((name, score))
                        if score > best_score:
                            best_score, best_name = score, name

                    print_comparison_table(scores, SIMILARITY_THRESHOLD)

                    if best_score > SIMILARITY_THRESHOLD:
                        print_access_granted(best_name, best_score)
                    else:
                        print_access_denied(best_name, best_score)
                console.print()

            # ── LIST FACES ────────────────────────────────
            elif cmd == 'l':
                print_section("DATABASE")
                print_face_list(db)
                console.print()

            # ── DELETE FACE ───────────────────────────────
            elif cmd == 'd':
                print_section("DELETE FACE")
                if not db:
                    console.print("  [yellow]Database is empty[/]")
                else:
                    print_face_list(db)
                    name = console.input("\n  [bold red]Enter name to delete:[/] ").strip()
                    if name in db:
                        del db[name]
                        save_database(db)
                        console.print(f"  [green]✓ Deleted '{name}'[/]")
                    elif name:
                        console.print(f"  [red]✗ '{name}' not found[/]")
                console.print()

            # ── QUIT ──────────────────────────────────────
            elif cmd == 'q':
                print_section("SHUTDOWN")
                save_database(db)
                console.print("  [dim]Database saved[/]")
                console.print("  [bold cyan]👋 Goodbye![/]\n")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        console.print("\n\n  [yellow]⚠ Interrupted by user[/]")
    finally:
        camera.stop()
        detector.close()
        save_database(db)
        display.clear()
        console.print("  [green]✓ Cleanup complete[/]\n")


if __name__ == "__main__":
    main()