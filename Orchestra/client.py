#!/usr/bin/env python3
"""
Biometric Authentication Client  v3.0  ·  Orchestra.py front-end
Textual TUI with mDNS auto-discovery, live log streaming, CSV download,
face enrollment with persistent retry, and professional layout.

Logs are displayed in separate Face / Voice / System tabs.

Install deps (once):
    pip install textual>=0.52 websockets zeroconf
"""
import asyncio
import json
import os
from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Button, Static, Input,
    Label, DataTable, RichLog, TabbedContent, TabPane,
)
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.timer import Timer
from zeroconf import Zeroconf, ServiceBrowser
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


SERVICE_TYPE = "_biometric-auth._tcp.local."
LOG_POLL_INTERVAL = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# mDNS discovery
# ─────────────────────────────────────────────────────────────────────────────
class _DiscoveryListener:
    def __init__(self, callback):
        self._cb   = callback
        self._seen = set()

    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if not info or not info.addresses:
            return
        ip  = ".".join(str(b) for b in info.addresses[0])
        url = f"ws://{ip}:{info.port}"
        if url not in self._seen:
            self._seen.add(url)
            self._cb(url)

    def remove_service(self, zc, type_, name):
        pass

    def update_service(self, zc, type_, name):
        self.add_service(zc, type_, name)


# ─────────────────────────────────────────────────────────────────────────────
# Main TUI
# ─────────────────────────────────────────────────────────────────────────────
class BiometricClient(App):

    TITLE = "Biometric Auth Client v3"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reconnect", "Reconnect"),
    ]

    CSS = """
    Screen { layout: horizontal; }

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
        padding-bottom: 0;
    }

    .divider {
        border-bottom: dashed $primary;
        margin-bottom: 1;
    }

    Button { width: 100%; margin-top: 1; }
    Input  { margin-top: 1; }

    #main-area { width: 1fr; padding: 1 2; }

    #face-log, #voice-log, #system-log {
        height: 1fr;
        min-height: 8;
        border: round $primary;
        padding: 0 1;
    }

    #user-table {
        height: 35%;
        min-height: 6;
        border: round $accent;
        margin-top: 1;
    }

    #status-bar {
        background: $boost;
        dock: bottom;
        height: 1;
        padding: 0 1;
    }

    .online    { color: $success; text-style: bold; }
    .offline   { color: $error;   text-style: bold; }
    .busy      { color: $warning; text-style: bold; }
    .searching { color: $warning; }

    #btn-cancel-enroll { display: none; }
    .cancel-visible #btn-cancel-enroll { display: block; }
    """

    def __init__(self):
        super().__init__()
        self.ws           = None
        self.server_url   = None
        self._cmd_lock    = asyncio.Lock()
        self._log_timer   = None
        self._last_log_count = 0
        self._enrolling   = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            # ── Sidebar ───────────────────────────────────────────────────
            with VerticalScroll(id="sidebar"):
                yield Static("◈ REGISTER", classes="section-label")
                yield Input(placeholder="Full name…",      id="reg-name")
                yield Button("Enroll Face",                id="btn-register-face",  variant="primary")
                yield Button("Cancel Enrolment",           id="btn-cancel-enroll",  variant="error")
                yield Button("Enroll Voice",               id="btn-register-voice", variant="default")
                yield Button("Register (Face + Voice)",    id="btn-register",       variant="default")

                yield Static("", classes="divider")
                yield Static("◈ AUTHENTICATE", classes="section-label")
                yield Button("Verify Face",  id="btn-auth-face",  variant="success")
                yield Button("Verify Voice", id="btn-auth-voice", variant="success")

                yield Static("", classes="divider")
                yield Static("◈ MANAGE USERS", classes="section-label")
                yield Button("↺ Refresh User List", id="btn-list",   variant="default")
                yield Input(placeholder="Name to delete…", id="del-name")
                yield Button("Delete User",              id="btn-delete", variant="error")

                yield Static("", classes="divider")
                yield Static("◈ SYSTEM", classes="section-label")
                yield Button("Fetch All Logs",   id="btn-logs",      variant="warning")
                yield Button("Download Log CSV", id="btn-download",  variant="warning")
                yield Button("Clear Display",    id="btn-clear",     variant="default")
                yield Button("⟳ Reconnect",      id="btn-reconnect", variant="primary")
                yield Button("⏻ Shutdown Server", id="btn-shutdown",  variant="error")

            # ── Main panel ────────────────────────────────────────────────
            with Vertical(id="main-area"):
                yield Static("Enrolled Users", classes="section-label")
                yield DataTable(id="user-table")

                with TabbedContent():
                    with TabPane("Face Log", id="tab-face"):
                        yield RichLog(id="face-log", highlight=True,
                                      markup=True, wrap=True)
                    with TabPane("Voice Log", id="tab-voice"):
                        yield RichLog(id="voice-log", highlight=True,
                                      markup=True, wrap=True)
                    with TabPane("System", id="tab-system"):
                        yield RichLog(id="system-log", highlight=True,
                                      markup=True, wrap=True)

        with Horizontal(id="status-bar"):
            yield Label("Server: ")
            yield Label("SEARCHING…", id="server-status", classes="searching")
            yield Label("   │   ")
            yield Label("", id="last-action")

        yield Footer()

    def on_mount(self):
        tbl = self.query_one("#user-table", DataTable)
        tbl.add_columns("Name", "Face ✓", "Voice ✓")
        tbl.cursor_type = "row"
        self._sys_log("[yellow]Searching for biometric server via mDNS…[/yellow]")
        self._zc       = Zeroconf()
        self._listener = _DiscoveryListener(self._on_server_found)
        self._browser  = ServiceBrowser(self._zc, SERVICE_TYPE, self._listener)

    # ── mDNS ──────────────────────────────────────────────────────────────────
    def _on_server_found(self, url):
        self.server_url = url
        self.call_from_thread(self._sys_log,
            f"[cyan]Server found → [bold]{url}[/bold][/cyan]")
        self.call_from_thread(self._kick_connect)

    def _kick_connect(self):
        self.run_worker(self._connect(), exclusive=True, name="ws-connect")

    async def _connect(self):
        try:
            self.ws = await websockets.connect(
                self.server_url,
                ping_interval=30, ping_timeout=120, close_timeout=10)
            self._set_status("ONLINE", "online")
            self._sys_log("[green]Connected.[/green]")
            self.notify("Server online", severity="information")
            await self._refresh_users()
            self._start_log_polling()
        except Exception as exc:
            self._set_status("OFFLINE", "offline")
            self._sys_log(f"[red]Connection failed: {exc}[/red]")
            self.set_timer(10.0, self._auto_reconnect)

    def _auto_reconnect(self):
        if self.ws is None and self.server_url:
            self._sys_log("[yellow]Auto-reconnecting…[/yellow]")
            self._kick_connect()

    # ── Live log polling ──────────────────────────────────────────────────────
    def _start_log_polling(self):
        if self._log_timer is not None:
            self._log_timer.stop()
        self._log_timer = self.set_interval(
            LOG_POLL_INTERVAL, self._poll_logs, name="log-poll")

    async def _poll_logs(self):
        if not self.ws:
            return
        try:
            await self.ws.send(json.dumps({"command": "get_logs"}))
            raw = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            msg = json.loads(raw)
            if msg.get("status") != "success":
                return
            entries = msg.get("data", [])
            new_count = len(entries)
            if new_count > self._last_log_count:
                for entry in entries[self._last_log_count:]:
                    self._render_log_entry(entry)
                self._last_log_count = new_count
        except asyncio.TimeoutError:
            pass
        except (ConnectionClosed, WebSocketException):
            self._set_status("OFFLINE", "offline")
            self.ws = None
            self._sys_log("[red]Connection lost during log poll[/red]")
            self.set_timer(10.0, self._auto_reconnect)
        except Exception:
            pass

    # ── send/receive ──────────────────────────────────────────────────────────
    async def _send(self, payload: dict) -> dict | None:
        if not self.ws:
            self.notify("Not connected.", severity="error")
            return None
        try:
            await self.ws.send(json.dumps(payload))
            while True:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=180.0)
                msg = json.loads(raw)
                if msg.get("status") == "progress":
                    step = msg.get("step", "").upper()
                    txt  = msg.get("message", "")
                    self._face_log(f"[yellow]  ⟳  {step}: {txt}[/yellow]")
                else:
                    return msg
        except asyncio.TimeoutError:
            self._sys_log("[red]Request timed out (180s).[/red]")
            return None
        except (ConnectionClosed, WebSocketException) as exc:
            self._set_status("OFFLINE", "offline")
            self.ws = None
            if self.server_url:
                self._listener._seen.discard(self.server_url)
            self._sys_log(f"[red]Connection lost: {exc}[/red]")
            self.set_timer(10.0, self._auto_reconnect)
            return None
        except Exception as exc:
            self._sys_log(f"[red]Unexpected error: {exc}[/red]")
            return None

    # ── Button routing ────────────────────────────────────────────────────────
    async def on_button_pressed(self, event):
        btn = event.button.id
        if btn == "btn-cancel-enroll":
            if self._enrolling:
                self._face_log("[yellow]Sending cancel…[/yellow]")
                await self._send({"command": "cancel_enroll_face"})
            return
        async with self._cmd_lock:
            await self._handle(btn)

    async def _handle(self, btn):

        # ── ENROLL FACE (persistent retry until success or cancel) ────────
        if btn == "btn-register-face":
            name = self._reg_name()
            if not name: return
            self._enrolling = True
            self.query_one("#sidebar").add_class("cancel-visible")
            self._action(f"Enrolling face: {name}…")
            self._face_log(
                f"\n[cyan]► Enrol face for [bold]{name}[/bold] "
                f"(retries until success or cancel)…[/cyan]")
            res = await self._send({"command": "register_face", "name": name})
            self._enrolling = False
            self.query_one("#sidebar").remove_class("cancel-visible")
            self._show_enroll_result(res, "Face", name)
            if res and res.get("status") == "success":
                self.query_one("#reg-name", Input).value = ""
                await self._refresh_users()
            self._action("Ready")

        # ── ENROLL VOICE ──────────────────────────────────────────────────
        elif btn == "btn-register-voice":
            name = self._reg_name()
            if not name: return
            self._action(f"Enrolling voice: {name}…")
            self._voice_log(
                f"\n[cyan]► Enrol voice for [bold]{name}[/bold] — "
                f"trigger ESP32 and speak for 5-10s…[/cyan]")
            res = await self._send({"command": "register_voice", "name": name})
            self._show_enroll_result(res, "Voice", name)
            if res and res.get("status") == "success":
                self.query_one("#reg-name", Input).value = ""
                await self._refresh_users()
            self._action("Ready")

        # ── REGISTER BOTH ─────────────────────────────────────────────────
        elif btn == "btn-register":
            name = self._reg_name()
            if not name: return
            self._action(f"Registering {name}…")
            self._sys_log(
                f"\n[cyan]► Register [bold]{name}[/bold] (face then voice)…[/cyan]")
            res = await self._send({"command": "register", "name": name})
            if res is None: return self._action("Ready")
            f_ok = res.get("face",{}).get("status") == "success"
            v_ok = res.get("voice",{}).get("status") == "success"
            f_err = res.get("face",{}).get("message","")
            v_err = res.get("voice",{}).get("message","")
            self._face_log(
                f"[{'green' if f_ok else 'red'}]  Face: "
                f"{'✓ enrolled' if f_ok else f'✗ {f_err}'}[/]")
            self._voice_log(
                f"[{'green' if v_ok else 'red'}]  Voice: "
                f"{'✓ enrolled' if v_ok else f'✗ {v_err}'}[/]")
            if f_ok or v_ok:
                self.query_one("#reg-name", Input).value = ""
                await self._refresh_users()
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
            res = await self._send({"command": "get_identities"})
            if res: await self._populate_user_table(res)
            self._action("Ready")

        # ── DELETE USER ───────────────────────────────────────────────────
        elif btn == "btn-delete":
            name = self.query_one("#del-name", Input).value.strip()
            if not name:
                self.notify("Enter a name to delete.", severity="warning")
                return
            self._action(f"Deleting {name}…")
            self._sys_log(f"\n[yellow]► Delete [bold]{name}[/bold]…[/yellow]")
            res = await self._send({"command": "delete", "name": name})
            if res is None: return self._action("Ready")
            f_r = res.get("face",{})
            v_r = res.get("voice",{})
            f_ok = f_r.get("status") == "success"
            v_ok = v_r.get("status") == "success"
            if f_ok or v_ok:
                self._sys_log(
                    f"[green]  Deleted [bold]{name}[/bold] — "
                    f"Face: {'✓' if f_ok else '—'}  "
                    f"Voice: {'✓' if v_ok else '—'}[/green]")
                self.query_one("#del-name", Input).value = ""
                await self._refresh_users()
            else:
                reason = (f_r.get("message","") or v_r.get("message","")
                          or "User not found.")
                self._sys_log(f"[red]  ✘ {reason}[/red]")
            self._action("Ready")

        # ── FETCH LOGS ────────────────────────────────────────────────────
        elif btn == "btn-logs":
            self._action("Fetching logs…")
            res = await self._send({"command": "get_logs"})
            if res is None: return self._action("Ready")
            entries = res.get("data", [])
            if not entries:
                self._sys_log("[yellow]No log entries yet.[/yellow]")
            else:
                self._sys_log(
                    f"\n[bold cyan]─── FULL LOG ({len(entries)} entries) ───[/bold cyan]")
                self._last_log_count = 0
                for entry in entries:
                    self._render_log_entry(entry)
                self._last_log_count = len(entries)
                self._sys_log("[bold cyan]─── END ───[/bold cyan]")
            self._action("Ready")

        # ── DOWNLOAD CSV ──────────────────────────────────────────────────
        elif btn == "btn-download":
            self._action("Downloading log CSV…")
            res = await self._send({"command": "get_log_file"})
            if res is None: return self._action("Ready")
            if res.get("status") == "success":
                csv_data = res.get("csv", "")
                fname    = res.get("filename", "biometric_log.csv")
                if not csv_data.strip():
                    self._sys_log("[yellow]Log file is empty.[/yellow]")
                else:
                    path = os.path.expanduser(f"~/{fname}")
                    try:
                        with open(path, "w") as f:
                            f.write(csv_data)
                        self._sys_log(
                            f"[green]Log saved: [bold]{path}[/bold] "
                            f"({len(csv_data)} bytes)[/green]")
                        self.notify(f"Saved: {path}", severity="information")
                    except Exception as e:
                        self._sys_log(f"[red]Save failed: {e}[/red]")
            else:
                self._sys_log(
                    f"[red]Download failed: {res.get('message','')}[/red]")
            self._action("Ready")

        # ── CLEAR DISPLAY ─────────────────────────────────────────────────
        elif btn == "btn-clear":
            self.query_one("#face-log", RichLog).clear()
            self.query_one("#voice-log", RichLog).clear()
            self.query_one("#system-log", RichLog).clear()
            self._last_log_count = 0

        # ── RECONNECT ─────────────────────────────────────────────────────
        elif btn == "btn-reconnect":
            await self._do_reconnect()

        # ── SHUTDOWN SERVER ───────────────────────────────────────────────
        elif btn == "btn-shutdown":
            self._sys_log("[red bold]Sending shutdown command…[/red bold]")
            res = await self._send({"command": "shutdown"})
            if res and res.get("status") == "success":
                self._sys_log("[red]Server shutting down.[/red]")
            self._action("Ready")

    # ── reconnect ─────────────────────────────────────────────────────────────
    async def _do_reconnect(self):
        self._sys_log("[yellow]Reconnecting…[/yellow]")
        if self.ws:
            try: await self.ws.close()
            except Exception: pass
            self.ws = None
        self._listener._seen.clear()
        self._last_log_count = 0
        self._set_status("SEARCHING…", "searching")
        if self.server_url:
            self._kick_connect()
        else:
            self._sys_log("[yellow]Waiting for mDNS…[/yellow]")

    def action_reconnect(self):
        self.run_worker(self._do_reconnect())

    # ── result helpers ────────────────────────────────────────────────────────
    def _show_enroll_result(self, res, modality, name):
        if res is None: return
        log_fn = self._face_log if modality == "Face" else self._voice_log
        if res.get("status") == "success":
            log_fn(f"[green]  ✓ {modality} enrolled: [bold]{name}[/bold][/green]")
            self.notify(f"{modality} enrolled: {name}", severity="information")
        else:
            msg = res.get("message", "Unknown error")
            log_fn(f"[red]  ✘ {modality} failed: {msg}[/red]")
            self.notify(msg, title=f"{modality} failed", severity="error")

    def _show_auth_result(self, res, method):
        if res is None: return
        log_fn = self._face_log if method == "FACE" else self._voice_log
        if res.get("status") != "success":
            msg = res.get("message", "Error")
            log_fn(f"[red]  ✘ {method}: {msg}[/red]")
            return
        granted = res.get("granted", False)
        user    = res.get("user", "—")
        score   = res.get("score", 0.0)
        ts      = datetime.now().strftime("%H:%M:%S")
        if granted:
            log_fn(
                f"[green]  ✓  [{ts}]  {method}  │  GRANTED  │  "
                f"[bold]{user}[/bold]  │  {score:.4f}[/green]")
            self.notify(f"Welcome, {user}!  ({score:.2f})",
                        title=f"{method} — GRANTED", severity="information")
        else:
            log_fn(
                f"[red]  ✗  [{ts}]  {method}  │  DENIED   │  "
                f"{user}  │  {score:.4f}[/red]")
            self.notify("Access denied.",
                        title=f"{method} — DENIED", severity="warning")

    # ── log entry renderer ────────────────────────────────────────────────────
    def _render_log_entry(self, entry):
        method     = (entry.get("method") or "system").lower()
        event      = entry.get("event", "")
        user       = entry.get("user", "—") or "—"
        conf       = entry.get("confidence", 0.0) or 0.0
        ts         = entry.get("timestamp", "?")
        reason     = entry.get("reason", "")
        s2_user    = entry.get("second_best_user", "")
        s2_score   = entry.get("second_best_score", 0.0) or 0.0
        audio_dur  = entry.get("audio_duration_s", 0.0) or 0.0
        inf_ms     = entry.get("inference_time_ms", 0.0) or 0.0
        detail     = entry.get("detail", "")

        eu = event.upper()
        if "GRANTED" in eu or "ENROLLED" in eu:
            color = "green"
        elif "DENIED" in eu or "FAILED" in eu or "REJECTED" in eu:
            color = "red"
        elif "ERROR" in eu or "CRASH" in eu:
            color = "red bold"
        elif "STANDBY" in eu or "LISTENING" in eu or "PROCESSING" in eu:
            color = "yellow"
        else:
            color = "dim"

        parts = [f"{ts}  {event:<28} user={user:<12} conf={conf:.4f}"]
        if s2_user:
            parts.append(f"2nd={s2_user}({s2_score:.4f})")
        if audio_dur > 0:
            parts.append(f"audio={audio_dur:.1f}s")
        if inf_ms > 0:
            parts.append(f"inf={inf_ms:.0f}ms")
        if reason:
            parts.append(f"reason={reason}")
        if detail:
            parts.append(detail)

        line = f"[{color}]  {'  '.join(parts)}[/{color}]"

        if method == "face":
            self._face_log(line)
        elif method == "voice":
            self._voice_log(line)
        else:
            self._sys_log(line)

    # ── user table ────────────────────────────────────────────────────────────
    async def _refresh_users(self):
        res = await self._send({"command": "get_identities"})
        if res: await self._populate_user_table(res)

    async def _populate_user_table(self, res):
        tbl = self.query_one("#user-table", DataTable)
        tbl.clear()
        if res.get("status") != "success":
            self._sys_log(f"[red]Error: {res.get('message','')}[/red]")
            return
        all_names  = res.get("data", [])
        face_only  = set(res.get("face_only", []))
        voice_only = set(res.get("voice_only", []))
        if not all_names:
            self._sys_log("[yellow]No users enrolled yet.[/yellow]")
            return
        for name in sorted(all_names):
            has_face  = name not in voice_only
            has_voice = name not in face_only
            tbl.add_row(name,
                        "✓" if has_face  else "—",
                        "✓" if has_voice else "—")
        n_both = len(all_names) - len(face_only) - len(voice_only)
        self._sys_log(
            f"[cyan]{len(all_names)} user(s)  |  "
            f"both: {n_both}  face-only: {len(face_only)}  "
            f"voice-only: {len(voice_only)}[/cyan]")

    # ── utilities ─────────────────────────────────────────────────────────────
    def _reg_name(self) -> str:
        name = self.query_one("#reg-name", Input).value.strip()
        if not name:
            self.notify("Enter a name first.", severity="warning")
        return name

    def _face_log(self, m):
        try: self.query_one("#face-log", RichLog).write(m)
        except Exception: pass

    def _voice_log(self, m):
        try: self.query_one("#voice-log", RichLog).write(m)
        except Exception: pass

    def _sys_log(self, m):
        try: self.query_one("#system-log", RichLog).write(m)
        except Exception: pass

    def _set_status(self, text, css_class):
        lbl = self.query_one("#server-status", Label)
        lbl.update(text)
        lbl.set_classes(css_class)

    def _action(self, text):
        self.query_one("#last-action", Label).update(text)

    # ── cleanup ───────────────────────────────────────────────────────────────
    async def on_unmount(self):
        if self._log_timer:
            self._log_timer.stop()
        if self.ws:
            try: await self.ws.close()
            except Exception: pass
        self._zc.close()


if __name__ == "__main__":
    BiometricClient().run()
