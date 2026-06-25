#!/usr/bin/env python3
"""
Biometric Authentication Client  v5.0  ·  Orchestra.py front-end
─────────────────────────────────────────────────────────────────
Professional Textual TUI — runs on Windows 11 laptop,
talks to Raspberry Pi 4 (Orchestra.py) over WebSocket / mDNS.

Changes from v4 → v5
  ✓  Added:   Delete mode selector — delete face only, voice only, or both
  ✓  Added:   Threshold tuning panel in Settings — face similarity, min gap,
               enroll duplicate, min gallery, voice verify, voice duplicate
  ✓  Added:   Thresholds auto-fetched from server on connect
  ✓  Added:   Thresholds pushed to server immediately on save
  ✓  Server:  New commands — delete_face, delete_voice, get_thresholds,
               set_thresholds (face + voice workers respond to threshold ops)
  ✓  Server:  Workers use module-level globals so threshold changes take
               effect in match_probe / check_enrollment_duplicate immediately

Install (once on Windows):
    pip install "textual>=0.52" websockets zeroconf

Run:
    python client.py
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button, DataTable, Footer, Header, Input,
    Label, RichLog, Select, Static, Switch,
    TabbedContent, TabPane,
)

try:
    from zeroconf import Zeroconf, ServiceBrowser
    _ZEROCONF_OK = True
except ImportError:
    _ZEROCONF_OK = False

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

APP_VERSION      = "5.0"
SERVICE_TYPE     = "_biometric-auth._tcp.local."
DEFAULT_POLL_S   = 3.0
DEFAULT_LOG_DIR  = str(Path.home())

# Exponential back-off ladder (seconds) for auto-reconnect
_BACKOFF = [2, 5, 10, 20, 30, 60]


# ─────────────────────────────────────────────────────────────────────────────
#  Persistent settings  (~/.biometric_client_settings.json)
# ─────────────────────────────────────────────────────────────────────────────

_SETTINGS_PATH = Path.home() / ".biometric_client_settings.json"

_DEFAULTS: dict = {
    "theme":          "dark",
    "manual_url":     "",            # blank → use mDNS
    "poll_interval":  DEFAULT_POLL_S,
    "log_dir":        DEFAULT_LOG_DIR,
    "auto_reconnect": True,
    # ── Thresholds (fetched from server on connect, editable in Settings) ──
    "face_similarity":       0.75,
    "face_min_score_gap":    0.10,
    "face_enroll_duplicate": 0.80,
    "face_min_gallery":      2,
    "voice_verify":          0.70,
    "voice_duplicate":       0.72,
}


def _load_settings() -> dict:
    try:
        if _SETTINGS_PATH.exists():
            saved = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            merged = dict(_DEFAULTS)
            merged.update({k: v for k, v in saved.items() if k in _DEFAULTS})
            return merged
    except Exception:
        pass
    return dict(_DEFAULTS)


def _save_settings(s: dict) -> None:
    try:
        _SETTINGS_PATH.write_text(json.dumps(s, indent=2), encoding="utf-8")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  mDNS listener
# ─────────────────────────────────────────────────────────────────────────────

class _MDNSListener:
    def __init__(self, callback):
        self._cb   = callback
        self._seen: set[str] = set()

    def _url_from_info(self, zc, type_, name) -> str | None:
        try:
            info = zc.get_service_info(type_, name)
            if info and info.addresses:
                ip = ".".join(str(b) for b in info.addresses[0])
                return f"ws://{ip}:{info.port}"
        except Exception:
            pass
        return None

    def add_service(self, zc, type_, name) -> None:
        url = self._url_from_info(zc, type_, name)
        if url and url not in self._seen:
            self._seen.add(url)
            self._cb(url)

    def remove_service(self, *_) -> None:
        pass

    def update_service(self, zc, type_, name) -> None:
        # Called on server restart — remove from seen so we re-announce
        url = self._url_from_info(zc, type_, name)
        if url:
            self._seen.discard(url)
            self.add_service(zc, type_, name)


# ─────────────────────────────────────────────────────────────────────────────
#  Settings Screen
# ─────────────────────────────────────────────────────────────────────────────

_THEME_OPTIONS = [
    ("Dark  (default)",   "dark"),
    ("Light",             "light"),
    ("Monokai",           "monokai"),
    ("Nord",              "nord"),
    ("Gruvbox",           "gruvbox"),
    ("Tokyo Night",       "tokyo-night"),
    ("Dracula",           "dracula"),
    ("GitHub Dark",       "github-dark"),
    ("Catppuccin Mocha",  "catppuccin-mocha"),
    ("Solarized Light",   "solarized-light"),
]


class SettingsScreen(Screen):
    """Full-overlay settings panel — push via app.push_screen()."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save",   "Save & Close"),
    ]

    CSS = """
    SettingsScreen {
        align: center middle;
        background: $background 60%;
    }

    #stg-panel {
        width: 72;
        height: auto;
        max-height: 92vh;
        background: $surface;
        border: double $primary;
        padding: 1 2 2 2;
        overflow-y: auto;
    }

    #stg-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        width: 100%;
        padding: 0 0 1 0;
        border-bottom: solid $primary-darken-2;
        margin-bottom: 1;
    }

    .stg-section {
        text-style: bold;
        color: $primary;
        margin: 1 0 1 0;
        padding-bottom: 0;
    }

    .stg-row {
        height: auto;
        margin-bottom: 1;
        align: left middle;
    }

    .stg-lbl {
        width: 26;
        color: $text-muted;
    }

    .stg-val { width: 1fr; }
    Select   { width: 1fr; }
    Input    { width: 1fr; }
    Switch   { margin-top: 0; }

    #stg-footer {
        margin-top: 2;
        align: right middle;
        height: auto;
    }

    #stg-footer Button { margin-left: 1; width: 20; }

    #stg-hint {
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
    }

    #stg-hint2 {
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
    }
    """

    def __init__(self, settings: dict, on_save):
        super().__init__()
        self._s       = dict(settings)
        self._on_save = on_save

    def compose(self) -> ComposeResult:
        s = self._s
        with Vertical(id="stg-panel"):
            yield Static("⚙   S E T T I N G S", id="stg-title")

            # ── Connection ────────────────────────────────────────────────
            yield Static("◈  CONNECTION", classes="stg-section")

            with Horizontal(classes="stg-row"):
                yield Label("Manual Server URL", classes="stg-lbl")
                yield Input(
                    value=s.get("manual_url", ""),
                    placeholder="ws://192.168.x.x:8765  (blank = mDNS auto-discover)",
                    id="s-url", classes="stg-val",
                )

            with Horizontal(classes="stg-row"):
                yield Label("Auto-Reconnect", classes="stg-lbl")
                yield Switch(value=bool(s.get("auto_reconnect", True)), id="s-ar")

            yield Static(
                "  Leave URL blank to use mDNS discovery. "
                "Fill it in if mDNS is blocked on your network.",
                id="stg-hint",
            )

            # ── Logs ──────────────────────────────────────────────────────
            yield Static("◈  LOGS", classes="stg-section")

            with Horizontal(classes="stg-row"):
                yield Label("Poll Interval  (seconds)", classes="stg-lbl")
                yield Input(
                    value=str(s.get("poll_interval", DEFAULT_POLL_S)),
                    placeholder="3.0   (minimum 0.5)",
                    id="s-poll", classes="stg-val",
                )

            with Horizontal(classes="stg-row"):
                yield Label("Download Save Folder", classes="stg-lbl")
                yield Input(
                    value=str(s.get("log_dir", DEFAULT_LOG_DIR)),
                    placeholder=DEFAULT_LOG_DIR,
                    id="s-logdir", classes="stg-val",
                )

            # ── Appearance ────────────────────────────────────────────────
            yield Static("◈  APPEARANCE", classes="stg-section")

            with Horizontal(classes="stg-row"):
                yield Label("Theme", classes="stg-lbl")
                yield Select(
                    options=_THEME_OPTIONS,
                    value=s.get("theme", "dark"),
                    id="s-theme",
                )

            # ── Thresholds ────────────────────────────────────────────────
            yield Static("◈  THRESHOLDS  (Face)", classes="stg-section")

            with Horizontal(classes="stg-row"):
                yield Label("Similarity  (verify)", classes="stg-lbl")
                yield Input(
                    value=str(s.get("face_similarity", 0.75)),
                    placeholder="0.75",
                    id="s-face-sim", classes="stg-val",
                )
            with Horizontal(classes="stg-row"):
                yield Label("Min Score Gap", classes="stg-lbl")
                yield Input(
                    value=str(s.get("face_min_score_gap", 0.10)),
                    placeholder="0.10",
                    id="s-face-gap", classes="stg-val",
                )
            with Horizontal(classes="stg-row"):
                yield Label("Enroll Dup Threshold", classes="stg-lbl")
                yield Input(
                    value=str(s.get("face_enroll_duplicate", 0.80)),
                    placeholder="0.80",
                    id="s-face-dup", classes="stg-val",
                )
            with Horizontal(classes="stg-row"):
                yield Label("Min Gallery Size", classes="stg-lbl")
                yield Input(
                    value=str(s.get("face_min_gallery", 2)),
                    placeholder="2",
                    id="s-face-gal", classes="stg-val",
                )

            yield Static("◈  THRESHOLDS  (Voice)", classes="stg-section")

            with Horizontal(classes="stg-row"):
                yield Label("Verify Threshold", classes="stg-lbl")
                yield Input(
                    value=str(s.get("voice_verify", 0.70)),
                    placeholder="0.70",
                    id="s-voice-ver", classes="stg-val",
                )
            with Horizontal(classes="stg-row"):
                yield Label("Enroll Dup Threshold", classes="stg-lbl")
                yield Input(
                    value=str(s.get("voice_duplicate", 0.72)),
                    placeholder="0.72",
                    id="s-voice-dup", classes="stg-val",
                )

            yield Static(
                "  Thresholds are pushed to the server immediately on save.",
                id="stg-hint2",
            )

            # ── Buttons ───────────────────────────────────────────────────
            with Horizontal(id="stg-footer"):
                yield Button("✕  Cancel",      id="s-cancel", variant="default")
                yield Button("✓  Save & Close", id="s-save",   variant="success")

    # ── handlers ──────────────────────────────────────────────────────────────
    @on(Button.Pressed, "#s-cancel")
    def action_cancel(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#s-save")
    def action_save(self) -> None:
        try:
            url     = self.query_one("#s-url",    Input).value.strip()
            ar      = self.query_one("#s-ar",     Switch).value
            raw_pi  = self.query_one("#s-poll",   Input).value.strip()
            log_dir = self.query_one("#s-logdir", Input).value.strip()

            try:
                poll = max(0.5, float(raw_pi))
            except ValueError:
                poll = DEFAULT_POLL_S

            theme = "dark"
            try:
                v = self.query_one("#s-theme", Select).value
                if v is not None and v is not Select.BLANK:
                    theme = str(v)
            except Exception:
                pass

            # ── Thresholds ────────────────────────────────────────────
            def _float_or(id_, fallback):
                try:
                    return float(self.query_one(id_, Input).value.strip())
                except (ValueError, Exception):
                    return fallback

            def _int_or(id_, fallback):
                try:
                    return int(self.query_one(id_, Input).value.strip())
                except (ValueError, Exception):
                    return fallback

            self._s.update({
                "manual_url":          url,
                "auto_reconnect":      ar,
                "poll_interval":       poll,
                "log_dir":             log_dir or DEFAULT_LOG_DIR,
                "theme":               theme,
                "face_similarity":       _float_or("#s-face-sim", 0.75),
                "face_min_score_gap":    _float_or("#s-face-gap", 0.10),
                "face_enroll_duplicate": _float_or("#s-face-dup", 0.80),
                "face_min_gallery":      _int_or("#s-face-gal", 2),
                "voice_verify":          _float_or("#s-voice-ver", 0.70),
                "voice_duplicate":       _float_or("#s-voice-dup", 0.72),
            })
            self._on_save(self._s)

        except Exception as exc:
            self.app.notify(f"Settings error: {exc}", severity="error")
            return

        self.app.pop_screen()


# ─────────────────────────────────────────────────────────────────────────────
#  Confirm Dialog
# ─────────────────────────────────────────────────────────────────────────────

class ConfirmScreen(Screen):
    """Modal yes/no dialog. Calls on_confirm() if the user clicks Yes."""

    CSS = """
    ConfirmScreen {
        align: center middle;
        background: $background 60%;
    }

    #cfm-box {
        width: 56;
        height: auto;
        background: $surface;
        border: double $warning;
        padding: 1 2 2 2;
        align: center top;
    }

    #cfm-title {
        text-align: center;
        text-style: bold;
        color: $warning;
        width: 100%;
        margin-bottom: 1;
    }

    #cfm-msg {
        text-align: center;
        width: 100%;
        margin-bottom: 2;
    }

    #cfm-btns            { align: center middle; height: auto; }
    #cfm-btns Button     { margin: 0 1; width: 16; }
    """

    def __init__(self, title: str, message: str, on_confirm):
        super().__init__()
        self._title = title
        self._msg   = message
        self._cb    = on_confirm

    def compose(self) -> ComposeResult:
        with Vertical(id="cfm-box"):
            yield Static(self._title, id="cfm-title")
            yield Static(self._msg,   id="cfm-msg")
            with Horizontal(id="cfm-btns"):
                yield Button("✕  No",  id="c-no",  variant="default")
                yield Button("✓  Yes", id="c-yes", variant="error")

    @on(Button.Pressed, "#c-no")
    def _no(self)  -> None: self.app.pop_screen()

    @on(Button.Pressed, "#c-yes")
    def _yes(self) -> None:
        self.app.pop_screen()
        self._cb()


# ─────────────────────────────────────────────────────────────────────────────
#  Main TUI Application
# ─────────────────────────────────────────────────────────────────────────────

class BiometricClient(App):

    TITLE = f"Biometric Auth Client  v{APP_VERSION}"

    BINDINGS = [
        Binding("q",      "quit",      "Quit"),
        Binding("r",      "reconnect", "Reconnect"),
        Binding("ctrl+s", "settings",  "Settings"),
        Binding("f5",     "refresh",   "Refresh Users"),
    ]

    CSS = """
    /* ── Layout ──────────────────────────────────────────────────────────── */
    Screen { layout: horizontal; }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    #sidebar {
        width: 34;
        background: $panel;
        border-right: tall $primary;
        padding: 1 1 2 1;
        overflow-y: auto;
    }

    .section-label {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }

    .divider {
        border-bottom: dashed $primary-darken-2;
        margin: 1 0;
    }

    Button { width: 100%; margin-top: 1; }
    Input  { margin-top: 1; }
    #del-mode { width: 100%; margin-top: 1; }

    /* ── Main area ───────────────────────────────────────────────────────── */
    #main-area { width: 1fr; padding: 1 2; }

    #face-log, #voice-log, #system-log {
        height: 1fr;
        min-height: 8;
        border: round $primary;
        padding: 0 1;
    }

    #user-table {
        height: 30%;
        min-height: 6;
        border: round $accent;
        margin-top: 1;
    }

    /* ── Status bar ──────────────────────────────────────────────────────── */
    #status-bar {
        background: $boost;
        dock: bottom;
        height: 1;
        padding: 0 1;
    }

    /* ── State colours ───────────────────────────────────────────────────── */
    .online      { color: $success; text-style: bold; }
    .offline     { color: $error;   text-style: bold; }
    .connecting  { color: $accent;  text-style: bold; }
    .searching   { color: $warning; }

    /* ── Cancel-enroll button visibility ─────────────────────────────────── */
    #btn-cancel-enroll { display: none; }
    .cancel-visible #btn-cancel-enroll { display: block; }
    """

    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        # Load settings BEFORE super().__init__() so TITLE is already set
        self._settings: dict          = _load_settings()
        super().__init__()

        self.ws                       = None
        self.server_url: str | None   = None

        # Serialises all WebSocket command traffic.
        # _poll_logs skips its cycle when this is locked, eliminating races.
        self._cmd_lock                = asyncio.Lock()

        self._log_timer               = None
        self._last_log_count: int     = 0
        self._enrolling: bool         = False
        self._reconnect_step: int     = 0

        self._zc                      = None
        self._listener: _MDNSListener | None = None
        self._browser                 = None

    # ── Compose ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():

            # ── Sidebar ───────────────────────────────────────────────────
            with VerticalScroll(id="sidebar"):

                yield Static("◈ REGISTER", classes="section-label")
                yield Input(placeholder="Full name…", id="reg-name")
                yield Button("Enroll Face",      id="btn-register-face",  variant="primary")
                yield Button("Cancel Enrolment", id="btn-cancel-enroll",  variant="error")
                yield Button("Enroll Voice",     id="btn-register-voice", variant="default")

                yield Static("", classes="divider")
                yield Static("◈ AUTHENTICATE", classes="section-label")
                yield Button("Verify Face",  id="btn-auth-face",  variant="success")
                yield Button("Verify Voice", id="btn-auth-voice", variant="success")

                yield Static("", classes="divider")
                yield Static("◈ MANAGE USERS", classes="section-label")
                yield Button("↺  Refresh Users", id="btn-list",   variant="default")
                yield Input(placeholder="Name to delete…", id="del-name")
                yield Select(
                    options=[
                        ("Face + Voice",  "both"),
                        ("Face Only",     "face"),
                        ("Voice Only",    "voice"),
                    ],
                    value="both",
                    id="del-mode",
                )
                yield Button("Delete User",      id="btn-delete", variant="error")

                yield Static("", classes="divider")
                yield Static("◈ SYSTEM", classes="section-label")
                yield Button("Fetch All Logs",   id="btn-logs",      variant="warning")
                yield Button("Download Log CSV", id="btn-download",  variant="warning")
                yield Button("Clear Display",    id="btn-clear",     variant="default")
                yield Button("⟳  Reconnect",     id="btn-reconnect", variant="primary")
                yield Button("⚙  Settings",      id="btn-settings",  variant="default")

            # ── Main panel ────────────────────────────────────────────────
            with Vertical(id="main-area"):
                yield Static("Enrolled Users", classes="section-label")
                yield DataTable(id="user-table")

                with TabbedContent():
                    with TabPane("Face Log",  id="tab-face"):
                        yield RichLog(id="face-log",
                                      highlight=True, markup=True, wrap=True)
                    with TabPane("Voice Log", id="tab-voice"):
                        yield RichLog(id="voice-log",
                                      highlight=True, markup=True, wrap=True)
                    with TabPane("System",    id="tab-system"):
                        yield RichLog(id="system-log",
                                      highlight=True, markup=True, wrap=True)

        with Horizontal(id="status-bar"):
            yield Label("Server: ")
            yield Label("SEARCHING…", id="server-status", classes="searching")
            yield Label("   │   ")
            yield Label("", id="last-action")

        yield Footer()

    # ── Mount ─────────────────────────────────────────────────────────────────
    def on_mount(self) -> None:
        tbl = self.query_one("#user-table", DataTable)
        tbl.add_columns("Name", "Face ✓", "Voice ✓")
        tbl.cursor_type = "row"
        self._apply_theme(self._settings.get("theme", "dark"))
        self._start_discovery()

    # ─────────────────────────────────────────────────────────────────────────
    #  mDNS discovery
    # ─────────────────────────────────────────────────────────────────────────

    def _start_discovery(self) -> None:
        manual = self._settings.get("manual_url", "").strip()
        if manual:
            self.server_url = manual
            self._sys_log(
                f"[cyan]Using manual server URL: [bold]{manual}[/bold][/cyan]")
            self._kick_connect()
            return

        if not _ZEROCONF_OK:
            self._sys_log(
                "[red]zeroconf not installed — open Settings (⚙) "
                "and enter the server URL manually.[/red]")
            return

        self._sys_log("[yellow]Searching for biometric server via mDNS…[/yellow]")
        try:
            self._zc       = Zeroconf()
            self._listener = _MDNSListener(self._on_server_found)
            self._browser  = ServiceBrowser(self._zc, SERVICE_TYPE, self._listener)
        except Exception as exc:
            self._sys_log(f"[red]mDNS init failed: {exc}[/red]")

    def _stop_discovery(self) -> None:
        for obj, method in [(self._browser, "cancel"), (self._zc, "close")]:
            if obj:
                try:
                    getattr(obj, method)()
                except Exception:
                    pass
        self._zc = self._browser = self._listener = None

    def _on_server_found(self, url: str) -> None:
        self.server_url = url
        self.call_from_thread(
            self._sys_log,
            f"[cyan]mDNS: server found → [bold]{url}[/bold][/cyan]")
        self.call_from_thread(self._kick_connect)

    def _kick_connect(self) -> None:
        self.run_worker(self._connect(), exclusive=True, name="ws-connect")

    # ─────────────────────────────────────────────────────────────────────────
    #  WebSocket connect / reconnect
    # ─────────────────────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        if not self.server_url:
            return
        self._set_status("CONNECTING…", "connecting")
        try:
            self.ws = await websockets.connect(
                self.server_url,
                ping_interval=30,
                ping_timeout=120,
                close_timeout=10,
                open_timeout=15,
            )
            self._reconnect_step = 0
            self._set_status("ONLINE", "online")
            self._sys_log("[green]✓ Connected to server.[/green]")
            self.notify("Server online", severity="information")
            # Refresh user list + fetch thresholds immediately after connecting
            async with self._cmd_lock:
                await self._refresh_users_locked()
                await self._fetch_thresholds_locked()
            self._start_log_polling()

        except Exception as exc:
            self.ws = None
            self._set_status("OFFLINE", "offline")
            self._sys_log(f"[red]✗ Connection failed: {exc}[/red]")
            self._schedule_auto_reconnect()

    def _schedule_auto_reconnect(self) -> None:
        if not self._settings.get("auto_reconnect", True):
            self._sys_log("[dim]Auto-reconnect is disabled — use ⟳ Reconnect manually.[/dim]")
            return
        if not self.server_url:
            return
        delay = _BACKOFF[min(self._reconnect_step, len(_BACKOFF) - 1)]
        self._reconnect_step += 1
        self._sys_log(
            f"[yellow]Auto-reconnect in {delay}s  "
            f"(attempt #{self._reconnect_step})…[/yellow]")
        self.set_timer(float(delay), self._auto_reconnect_fire)

    def _auto_reconnect_fire(self) -> None:
        """Called by set_timer — runs in event loop."""
        if self.ws is None and self.server_url:
            # Allow mDNS to re-deliver this URL if it changed
            if self._listener:
                self._listener._seen.discard(self.server_url)
            self._kick_connect()

    async def _do_reconnect(self) -> None:
        """Full manual reconnect — resets back-off, restarts discovery."""
        self._sys_log("[yellow]⟳  Manual reconnect…[/yellow]")
        self._reconnect_step = 0

        # Stop log polling
        if self._log_timer:
            self._log_timer.stop()
            self._log_timer = None

        # Close existing connection
        old_ws = self.ws
        self.ws = None
        if old_ws:
            try:
                await old_ws.close()
            except Exception:
                pass

        self._last_log_count = 0
        self._set_status("RECONNECTING…", "connecting")

        manual = self._settings.get("manual_url", "").strip()
        if manual:
            # Manual URL — connect directly
            self.server_url = manual
            await self._connect()
        elif self.server_url:
            # Known mDNS URL — retry directly, allow re-announce
            if self._listener:
                self._listener._seen.discard(self.server_url)
            await self._connect()
        else:
            # No known URL — restart full mDNS discovery
            self._stop_discovery()
            self._start_discovery()

    def action_reconnect(self) -> None:
        self.run_worker(self._do_reconnect(), name="manual-reconnect")

    # ─────────────────────────────────────────────────────────────────────────
    #  Log polling  (incremental — only delivers new entries)
    # ─────────────────────────────────────────────────────────────────────────

    def _start_log_polling(self) -> None:
        if self._log_timer:
            self._log_timer.stop()
        poll_s = max(0.5, float(self._settings.get("poll_interval", DEFAULT_POLL_S)))
        self._log_timer = self.set_interval(
            poll_s, self._poll_logs, name="log-poll")

    async def _poll_logs(self) -> None:
        if not self.ws:
            return
        # KEY FIX: skip this cycle if a command is in flight to avoid
        # WS frame interleaving (was the root cause of "delete is shaky")
        if self._cmd_lock.locked():
            return
        async with self._cmd_lock:
            if not self.ws:
                return
            try:
                await self.ws.send(json.dumps({"command": "get_logs"}))
                raw = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                if msg.get("status") != "success":
                    return
                entries  = msg.get("data", [])
                new_cnt  = len(entries)
                if new_cnt > self._last_log_count:
                    for entry in entries[self._last_log_count:]:
                        self._render_log_entry(entry)
                    self._last_log_count = new_cnt
            except asyncio.TimeoutError:
                pass
            except (ConnectionClosed, WebSocketException):
                self._ws_lost("log poll")
            except Exception:
                pass

    def _ws_lost(self, ctx: str = "") -> None:
        """Handle unexpected connection drop."""
        if self._log_timer:
            self._log_timer.stop()
            self._log_timer = None
        old = self.ws
        self.ws = None
        self._set_status("OFFLINE", "offline")
        self._sys_log(
            f"[red]✗ Connection lost"
            f"{f'  ({ctx})' if ctx else ''}.[/red]")
        self._schedule_auto_reconnect()

    # ─────────────────────────────────────────────────────────────────────────
    #  Core send / receive  (must be called while holding self._cmd_lock)
    # ─────────────────────────────────────────────────────────────────────────

    async def _send(self, payload: dict, timeout: float = 180.0) -> dict | None:
        if not self.ws:
            self.notify("Not connected.", severity="error")
            return None
        try:
            await self.ws.send(json.dumps(payload))
            while True:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
                msg = json.loads(raw)
                if msg.get("status") == "progress":
                    step = msg.get("step", "").upper()
                    txt  = msg.get("message", "")
                    self._face_log(f"[yellow]  ⟳  {step}: {txt}[/yellow]")
                else:
                    return msg
        except asyncio.TimeoutError:
            self._sys_log(f"[red]Request timed out ({timeout:.0f} s).[/red]")
            return None
        except (ConnectionClosed, WebSocketException):
            self._ws_lost()
            return None
        except Exception as exc:
            self._sys_log(f"[red]Unexpected send error: {exc}[/red]")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Button event handler
    # ─────────────────────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id

        # ── Settings ──────────────────────────────────────────────────────
        if btn == "btn-settings":
            self.action_settings()
            return

        # ── Cancel enrolment (bypass cmd_lock — must interrupt enroll) ────
        if btn == "btn-cancel-enroll":
            if self._enrolling:
                self._face_log("[yellow]Sending cancel signal…[/yellow]")
                self.run_worker(self._fire_cancel(), name="fire-cancel")
            return

        # ── Clear display (pure UI, no WS) ────────────────────────────────
        if btn == "btn-clear":
            self.query_one("#face-log",   RichLog).clear()
            self.query_one("#voice-log",  RichLog).clear()
            self.query_one("#system-log", RichLog).clear()
            self._last_log_count = 0
            return

        # ── Reconnect ─────────────────────────────────────────────────────
        if btn == "btn-reconnect":
            self.run_worker(self._do_reconnect(), name="manual-reconnect")
            return

        # ── Delete — show confirmation before doing anything ──────────────
        if btn == "btn-delete":
            name = self.query_one("#del-name", Input).value.strip()
            if not name:
                self.notify("Enter a name to delete.", severity="warning")
                return
            # Read delete mode
            try:
                mode_val = self.query_one("#del-mode", Select).value
                mode = str(mode_val) if mode_val is not None and mode_val is not Select.BLANK else "both"
            except Exception:
                mode = "both"
            mode_label = {"face": "face only", "voice": "voice only"}.get(mode, "face + voice")
            # Capture in closure
            captured_name = name
            captured_mode = mode

            def _confirmed():
                self.run_worker(
                    self._delete_worker(captured_name, captured_mode),
                    name="delete-user",
                )

            self.push_screen(ConfirmScreen(
                "⚠  Confirm Delete",
                f'Delete  "{captured_name}"  ({mode_label})?',
                _confirmed,
            ))
            return

        # ── All other WS commands ─────────────────────────────────────────
        self.run_worker(self._command_worker(btn), name=f"cmd-{btn}")

    # ── Cancel-enroll: fire-and-forget, no lock ───────────────────────────────
    async def _fire_cancel(self) -> None:
        """Send cancel without acquiring cmd_lock (must not block enroll receive)."""
        if self.ws:
            try:
                await self.ws.send(
                    json.dumps({"command": "cancel_enroll_face"}))
            except Exception:
                pass

    # ── General command worker ────────────────────────────────────────────────
    async def _command_worker(self, btn: str) -> None:
        async with self._cmd_lock:
            await self._handle(btn)

    # ─────────────────────────────────────────────────────────────────────────
    #  Command handlers  (all run inside self._cmd_lock)
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle(self, btn: str) -> None:

        # ── ENROLL FACE ───────────────────────────────────────────────────
        if btn == "btn-register-face":
            name = self._reg_name()
            if not name:
                return
            self._enrolling = True
            self.query_one("#sidebar").add_class("cancel-visible")
            self._action(f"Enrolling face: {name}…")
            self._face_log(
                f"\n[cyan]► Enrol face for [bold]{name}[/bold]"
                f" (retries until success or cancel)…[/cyan]")
            res = await self._send({"command": "register_face", "name": name})
            self._enrolling = False
            self.query_one("#sidebar").remove_class("cancel-visible")
            self._show_enroll_result(res, "Face", name)
            if res and res.get("status") == "success":
                self.query_one("#reg-name", Input).value = ""
                await self._refresh_users_locked()
            self._action("Ready")

        # ── ENROLL VOICE ──────────────────────────────────────────────────
        elif btn == "btn-register-voice":
            name = self._reg_name()
            if not name:
                return
            self._action(f"Enrolling voice: {name}…")
            self._voice_log(
                f"\n[cyan]► Enrol voice for [bold]{name}[/bold]"
                f" — trigger ESP32 and speak for 5–10 s…[/cyan]")
            res = await self._send({"command": "register_voice", "name": name})
            self._show_enroll_result(res, "Voice", name)
            if res and res.get("status") == "success":
                self.query_one("#reg-name", Input).value = ""
                await self._refresh_users_locked()
            self._action("Ready")

        # ── AUTH FACE ─────────────────────────────────────────────────────
        elif btn == "btn-auth-face":
            self._action("Face auth — look at camera…")
            self._face_log("\n[cyan]► Face verification — look at camera…[/cyan]")
            res = await self._send({"command": "auth_face"})
            self._show_auth_result(res, "FACE")
            self._action("Ready")

        # ── AUTH VOICE ────────────────────────────────────────────────────
        elif btn == "btn-auth-voice":
            self._action("Voice auth — trigger ESP32…")
            self._voice_log(
                "\n[cyan]► Voice verification — trigger ESP32 and speak…[/cyan]")
            res = await self._send({"command": "auth_voice"})
            self._show_auth_result(res, "VOICE")
            self._action("Ready")

        # ── LIST USERS ────────────────────────────────────────────────────
        elif btn == "btn-list":
            self._action("Fetching user list…")
            await self._refresh_users_locked()
            self._action("Ready")

        # ── FETCH LOGS ────────────────────────────────────────────────────
        elif btn == "btn-logs":
            self._action("Fetching full logs…")
            res = await self._send({"command": "get_logs"})
            if res is None:
                self._action("Ready")
                return
            entries = res.get("data", [])
            if not entries:
                self._sys_log("[yellow]No log entries yet.[/yellow]")
            else:
                self._sys_log(
                    f"\n[bold cyan]─── FULL LOG  "
                    f"({len(entries)} entries) ───[/bold cyan]")
                self._last_log_count = 0
                for entry in entries:
                    self._render_log_entry(entry)
                self._last_log_count = len(entries)
                self._sys_log("[bold cyan]─── END ───[/bold cyan]")
            self._action("Ready")

        # ── DOWNLOAD CSV ──────────────────────────────────────────────────
        elif btn == "btn-download":
            await self._download_log()

    # ─────────────────────────────────────────────────────────────────────────
    #  Delete with confirmation + retry  (runs inside cmd_lock via _delete_worker)
    # ─────────────────────────────────────────────────────────────────────────

    async def _delete_worker(self, name: str, mode: str = "both") -> None:
        async with self._cmd_lock:
            await self._delete_locked(name, mode)

    async def _delete_locked(self, name: str, mode: str = "both",
                             max_retries: int = 3) -> None:
        mode_label = {"face": "face only", "voice": "voice only"}.get(mode, "face + voice")
        self._action(f"Deleting {name} ({mode_label})…")
        self._sys_log(f"\n[yellow]► Delete [bold]{name}[/bold] ({mode_label})…[/yellow]")

        # Choose the correct server command
        cmd_name = {"face": "delete_face", "voice": "delete_voice"}.get(mode, "delete")

        for attempt in range(1, max_retries + 1):
            if not self.ws:
                self._sys_log(
                    f"[yellow]  Attempt {attempt}: not connected — aborting.[/yellow]")
                self._action("Ready")
                return

            res = await self._send(
                {"command": cmd_name, "name": name}, timeout=30.0)

            if res is None:
                if attempt < max_retries:
                    self._sys_log(
                        f"[yellow]  Attempt {attempt}: no response — retrying…[/yellow]")
                    await asyncio.sleep(1.0)
                    continue
                self._sys_log(
                    "[red]  ✘ Delete failed: no response after "
                    f"{max_retries} attempts.[/red]")
                self.notify("Delete failed: server did not respond.",
                            severity="error")
                self._action("Ready")
                return

            f_r  = res.get("face",  {}) or {}
            v_r  = res.get("voice", {}) or {}
            f_ok = f_r.get("status") in ("success", "skipped")
            v_ok = v_r.get("status") in ("success", "skipped")
            f_del = f_r.get("status") == "success"
            v_del = v_r.get("status") == "success"

            if f_del or v_del:
                face_tag  = "[green]✓[/green]" if f_del else ("[dim]skip[/dim]" if f_r.get("status") == "skipped" else "[dim]—[/dim]")
                voice_tag = "[green]✓[/green]" if v_del else ("[dim]skip[/dim]" if v_r.get("status") == "skipped" else "[dim]—[/dim]")
                self._sys_log(
                    f"[green]  ✓ Deleted [bold]{name}[/bold]  "
                    f"│  Face: {face_tag}  │  Voice: {voice_tag}[/green]")
                self.query_one("#del-name", Input).value = ""
                await self._refresh_users_locked()
                self.notify(f'Deleted "{name}"', severity="information")
                self._action("Ready")
                return

            # Workers reported not-found → no point retrying
            reason = (f_r.get("message", "")
                      or v_r.get("message", "")
                      or "User not found in the selected database(s).")
            self._sys_log(f"[red]  ✘ {reason}[/red]")
            self.notify(reason, title="Delete failed", severity="error")
            self._action("Ready")
            return

        self._sys_log("[red]  ✘ Delete failed after all retries.[/red]")
        self._action("Ready")

    # ─────────────────────────────────────────────────────────────────────────
    #  Log download — always delivers latest, retries, timestamped filename
    # ─────────────────────────────────────────────────────────────────────────

    async def _download_log(self, max_retries: int = 3) -> None:
        self._action("Downloading log CSV…")

        for attempt in range(1, max_retries + 1):
            if not self.ws:
                if attempt < max_retries:
                    self._sys_log(
                        f"[yellow]  Download attempt {attempt}: "
                        "not connected — waiting…[/yellow]")
                    await asyncio.sleep(2.0)
                    continue
                self._sys_log("[red]  ✘ Download failed: not connected.[/red]")
                self.notify("Download failed: not connected.", severity="error")
                self._action("Ready")
                return

            res = await self._send({"command": "get_log_file"}, timeout=30.0)

            if res is None:
                if attempt < max_retries:
                    self._sys_log(
                        f"[yellow]  No response on attempt {attempt} — retrying…[/yellow]")
                    await asyncio.sleep(1.5)
                    continue
                self._sys_log("[red]  ✘ Download failed: no server response.[/red]")
                self.notify("Download failed: no response.", severity="error")
                self._action("Ready")
                return

            if res.get("status") != "success":
                err = res.get("message", "unknown error")
                if attempt < max_retries:
                    self._sys_log(
                        f"[yellow]  Server error on attempt {attempt}: "
                        f"{err} — retrying…[/yellow]")
                    await asyncio.sleep(1.5)
                    continue
                self._sys_log(f"[red]  ✘ Download failed: {err}[/red]")
                self.notify(f"Download failed: {err}", severity="error")
                self._action("Ready")
                return

            csv_data = res.get("csv", "")
            fname    = res.get("filename", "biometric_log.csv")

            if not csv_data.strip():
                self._sys_log("[yellow]  Log file is empty on the server.[/yellow]")
                self.notify("Log file is empty.", severity="warning")
                self._action("Ready")
                return

            # Build timestamped filename so each download is kept
            stem, ext = os.path.splitext(fname)
            ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"{stem}_{ts_str}{ext}"

            # Try configured directory first, home directory as fallback
            log_dir   = self._settings.get("log_dir", DEFAULT_LOG_DIR)
            saved     = False
            for directory in [log_dir, DEFAULT_LOG_DIR]:
                try:
                    os.makedirs(directory, exist_ok=True)
                    path = os.path.join(directory, save_name)
                    with open(path, "w", encoding="utf-8", newline="") as fh:
                        fh.write(csv_data)
                    self._sys_log(
                        f"[green]  ✓ Log saved: [bold]{path}[/bold]"
                        f"  ({len(csv_data):,} bytes)[/green]")
                    self.notify(f"Saved: {path}", severity="information")
                    saved = True
                    break
                except Exception as exc:
                    self._sys_log(
                        f"[yellow]  Cannot write to {directory!r}: {exc}[/yellow]")

            if not saved:
                self._sys_log(
                    "[red]  ✘ Could not save log to any directory.[/red]")
                self.notify("Save failed — check log directory in Settings.",
                            severity="error")

            self._action("Ready")
            return   # success (or permanent save failure) — do not retry

    # ─────────────────────────────────────────────────────────────────────────
    #  Settings action
    # ─────────────────────────────────────────────────────────────────────────

    def action_settings(self) -> None:
        # Snapshot threshold keys so we can detect changes
        _THRESH_KEYS = (
            "face_similarity", "face_min_score_gap",
            "face_enroll_duplicate", "face_min_gallery",
            "voice_verify", "voice_duplicate",
        )
        old_thresholds = {k: self._settings.get(k) for k in _THRESH_KEYS}

        def _on_save(new_s: dict) -> None:
            old_theme = self._settings.get("theme")
            old_poll  = self._settings.get("poll_interval")
            self._settings = new_s
            _save_settings(new_s)

            if new_s.get("theme") != old_theme:
                self._apply_theme(new_s["theme"])

            if new_s.get("poll_interval") != old_poll and self.ws:
                self._start_log_polling()

            # Push thresholds to server if any changed
            thresh_changed = any(
                new_s.get(k) != old_thresholds.get(k) for k in _THRESH_KEYS)
            if thresh_changed and self.ws:
                self.run_worker(self._push_thresholds(),
                                name="push-thresholds")

            self._sys_log("[green]✓ Settings saved.[/green]")

        self.push_screen(SettingsScreen(self._settings, _on_save))

    def action_refresh(self) -> None:
        self.run_worker(self._command_worker("btn-list"), name="refresh-f5")

    def _apply_theme(self, theme: str) -> None:
        try:
            self.theme = theme
        except Exception:
            try:
                self.theme = "dark"
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    #  User table helpers  (called inside cmd_lock)
    # ─────────────────────────────────────────────────────────────────────────

    async def _refresh_users_locked(self) -> None:
        res = await self._send({"command": "get_identities"})
        if res:
            await self._populate_user_table(res)

    async def _populate_user_table(self, res: dict) -> None:
        tbl = self.query_one("#user-table", DataTable)
        tbl.clear()
        if res.get("status") != "success":
            self._sys_log(
                f"[red]User list error: {res.get('message', '')}[/red]")
            return
        all_names  = res.get("data", [])
        face_only  = set(res.get("face_only",  []))
        voice_only = set(res.get("voice_only", []))
        if not all_names:
            self._sys_log("[yellow]No users enrolled yet.[/yellow]")
            return
        for name in sorted(all_names):
            tbl.add_row(
                name,
                "✓" if name not in voice_only else "—",
                "✓" if name not in face_only  else "—",
            )
        n_both = len(all_names) - len(face_only) - len(voice_only)
        self._sys_log(
            f"[cyan]{len(all_names)} user(s)  │  "
            f"both={n_both}  "
            f"face-only={len(face_only)}  "
            f"voice-only={len(voice_only)}[/cyan]")

    # ─────────────────────────────────────────────────────────────────────────
    #  Threshold helpers  (called inside cmd_lock)
    # ─────────────────────────────────────────────────────────────────────────

    async def _fetch_thresholds_locked(self) -> None:
        """Fetch current thresholds from server and update local settings."""
        res = await self._send({"command": "get_thresholds"}, timeout=10.0)
        if res is None or res.get("status") != "success":
            self._sys_log("[yellow]Could not fetch thresholds from server.[/yellow]")
            return
        _THRESH_KEYS = (
            "face_similarity", "face_min_score_gap",
            "face_enroll_duplicate", "face_min_gallery",
            "voice_verify", "voice_duplicate",
        )
        updated = []
        for k in _THRESH_KEYS:
            if k in res:
                self._settings[k] = res[k]
                updated.append(f"{k}={res[k]}")
        _save_settings(self._settings)
        if updated:
            self._sys_log(
                f"[cyan]Thresholds synced: {', '.join(updated)}[/cyan]")

    async def _push_thresholds(self) -> None:
        """Push local threshold settings to the server (runs in cmd_lock)."""
        async with self._cmd_lock:
            payload = {
                "command":               "set_thresholds",
                "face_similarity":       self._settings.get("face_similarity", 0.75),
                "face_min_score_gap":    self._settings.get("face_min_score_gap", 0.10),
                "face_enroll_duplicate": self._settings.get("face_enroll_duplicate", 0.80),
                "face_min_gallery":      self._settings.get("face_min_gallery", 2),
                "voice_verify":          self._settings.get("voice_verify", 0.70),
                "voice_duplicate":       self._settings.get("voice_duplicate", 0.72),
            }
            res = await self._send(payload, timeout=10.0)
            if res and res.get("status") == "success":
                changed = res.get("changed", [])
                if changed:
                    self._sys_log(
                        f"[green]✓ Thresholds pushed: "
                        f"{', '.join(changed)}[/green]")
                else:
                    self._sys_log("[dim]Thresholds unchanged.[/dim]")
            else:
                self._sys_log(
                    "[red]✘ Failed to push thresholds to server.[/red]")

    # ─────────────────────────────────────────────────────────────────────────
    #  Result display helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _show_enroll_result(
            self, res: dict | None, modality: str, name: str) -> None:
        if res is None:
            return
        fn = self._face_log if modality == "Face" else self._voice_log
        if res.get("status") == "success":
            fn(f"[green]  ✓ {modality} enrolled: [bold]{name}[/bold][/green]")
            self.notify(f"{modality} enrolled: {name}", severity="information")
        else:
            msg = res.get("message", "Unknown error")
            fn(f"[red]  ✘ {modality} failed: {msg}[/red]")
            self.notify(msg, title=f"{modality} failed", severity="error")

    def _show_auth_result(self, res: dict | None, method: str) -> None:
        if res is None:
            return
        fn = self._face_log if method == "FACE" else self._voice_log
        if res.get("status") != "success":
            fn(f"[red]  ✘ {method}: {res.get('message', 'Error')}[/red]")
            return
        granted = res.get("granted", False)
        user    = res.get("user",    "—")
        score   = res.get("score",   0.0)
        ts      = datetime.now().strftime("%H:%M:%S")
        if granted:
            fn(f"[green]  ✓  [{ts}]  {method}  │  GRANTED  │  "
               f"[bold]{user}[/bold]  │  {score:.4f}[/green]")
            self.notify(
                f"Welcome, {user}!  ({score:.2f})",
                title=f"{method} — GRANTED", severity="information")
        else:
            fn(f"[red]  ✗  [{ts}]  {method}  │  DENIED   │  "
               f"{user}  │  {score:.4f}[/red]")
            self.notify("Access denied.",
                        title=f"{method} — DENIED", severity="warning")

    # ─────────────────────────────────────────────────────────────────────────
    #  Log-entry renderer
    # ─────────────────────────────────────────────────────────────────────────

    def _render_log_entry(self, entry: dict) -> None:
        method    = (entry.get("method") or "system").lower()
        event     = entry.get("event", "")
        user      = entry.get("user",  "—") or "—"
        conf      = entry.get("confidence",      0.0) or 0.0
        ts        = entry.get("timestamp",        "?")
        reason    = entry.get("reason",           "")
        s2_user   = entry.get("second_best_user", "")
        s2_score  = entry.get("second_best_score", 0.0) or 0.0
        audio_dur = entry.get("audio_duration_s",  0.0) or 0.0
        inf_ms    = entry.get("inference_time_ms", 0.0) or 0.0
        detail    = entry.get("detail",            "")

        eu = event.upper()
        if   "GRANTED"   in eu or "ENROLLED" in eu:  color = "green"
        elif "DENIED"    in eu or "FAILED"   in eu:  color = "red"
        elif "REJECTED"  in eu:                       color = "red"
        elif "ERROR"     in eu or "CRASH"    in eu:  color = "red bold"
        elif "CANCELLED" in eu or "DELETED"  in eu:  color = "magenta"
        elif "STANDBY"   in eu or "LISTENING" in eu: color = "yellow"
        elif "PROCESSING" in eu:                      color = "yellow"
        else:                                          color = "dim"

        parts = [f"{ts}  {event:<28} user={user:<12} conf={conf:.4f}"]
        if s2_user:   parts.append(f"2nd={s2_user}({s2_score:.4f})")
        if audio_dur: parts.append(f"audio={audio_dur:.1f}s")
        if inf_ms:    parts.append(f"inf={inf_ms:.0f}ms")
        if reason:    parts.append(f"reason={reason}")
        if detail:    parts.append(detail)

        line = f"[{color}]  {'  '.join(parts)}[/{color}]"

        if   method == "face":  self._face_log(line)
        elif method == "voice": self._voice_log(line)
        else:                   self._sys_log(line)

    # ─────────────────────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _reg_name(self) -> str:
        name = self.query_one("#reg-name", Input).value.strip()
        if not name:
            self.notify("Enter a name first.", severity="warning")
        return name

    def _face_log(self, m: str) -> None:
        try:
            self.query_one("#face-log", RichLog).write(m)
        except Exception:
            pass

    def _voice_log(self, m: str) -> None:
        try:
            self.query_one("#voice-log", RichLog).write(m)
        except Exception:
            pass

    def _sys_log(self, m: str) -> None:
        try:
            self.query_one("#system-log", RichLog).write(m)
        except Exception:
            pass

    def _set_status(self, text: str, css_class: str) -> None:
        try:
            lbl = self.query_one("#server-status", Label)
            lbl.update(text)
            lbl.set_classes(css_class)
        except Exception:
            pass

    def _action(self, text: str) -> None:
        try:
            self.query_one("#last-action", Label).update(text)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    #  Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    async def on_unmount(self) -> None:
        if self._log_timer:
            self._log_timer.stop()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        self._stop_discovery()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BiometricClient().run()
