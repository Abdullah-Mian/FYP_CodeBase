#!/usr/bin/env python3
"""
Biometric Authentication Client  ·  Orchestra.py front-end
Textual TUI with mDNS auto-discovery, full error handling, and live logs.

Install deps (once):
    pip install textual>=0.52 websockets zeroconf
"""
import asyncio
import json
from datetime import datetime

from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Button, Static, Input,
    Label, DataTable, RichLog,
)
from textual.containers import Vertical, Horizontal, VerticalScroll
from zeroconf import Zeroconf, ServiceBrowser
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


SERVICE_TYPE = "_biometric-auth._tcp.local."


# ─────────────────────────────────────────────────────────────────────────────
# mDNS discovery listener
# ─────────────────────────────────────────────────────────────────────────────
class _DiscoveryListener:
    def __init__(self, callback):
        self._cb   = callback
        self._seen = set()

    def add_service(self, zc: Zeroconf, type_: str, name: str):
        info = zc.get_service_info(type_, name)
        if not info or not info.addresses:
            return
        ip  = ".".join(str(b) for b in info.addresses[0])
        url = f"ws://{ip}:{info.port}"
        if url not in self._seen:
            self._seen.add(url)
            self._cb(url)

    def remove_service(self, zc, type_, name):
        pass   # handled by connection error in the app

    def update_service(self, zc, type_, name):
        self.add_service(zc, type_, name)


# ─────────────────────────────────────────────────────────────────────────────
# Main TUI
# ─────────────────────────────────────────────────────────────────────────────
class BiometricClient(App):

    # ── Styling ───────────────────────────────────────────────────────────────
    CSS = """
    /* ── Root layout ── */
    Screen { layout: horizontal; }

    /* ── Sidebar ── */
    #sidebar {
        width: 30;
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

    /* ── Main area ── */
    #main-area { width: 1fr; padding: 1 2; }

    #log-area {
        height: 60%;
        min-height: 10;
        border: round $primary;
        margin-top: 1;
        padding: 0 1;
    }

    #user-table {
        height: 40%;
        min-height: 6;
        border: round $accent;
        margin-top: 1;
    }

    /* ── Status bar ── */
    #status-bar {
        background: $boost;
        dock: bottom;
        height: 1;
        padding: 0 1;
    }

    .online   { color: $success; text-style: bold; }
    .offline  { color: $error;   text-style: bold; }
    .busy     { color: $warning; text-style: bold; }
    .searching { color: $warning; }
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.ws:         websockets.WebSocketClientProtocol | None = None
        self.server_url: str | None = None
        self._cmd_lock  = asyncio.Lock()   # one in-flight command at a time

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            # ── Sidebar ───────────────────────────────────────────────────────
            with VerticalScroll(id="sidebar"):
                yield Static("◈ REGISTER", classes="section-label")
                yield Input(placeholder="Full name…",     id="reg-name")
                yield Button("Register  (Face + Voice)",  id="btn-register",      variant="primary")
                yield Button("Enroll Face only",          id="btn-register-face", variant="default")
                yield Button("Enroll Voice only",         id="btn-register-voice",variant="default")

                yield Static("◈ AUTHENTICATE", classes="section-label")
                yield Button("Verify Face",  id="btn-auth-face",  variant="success")
                yield Button("Verify Voice", id="btn-auth-voice", variant="success")

                yield Static("◈ MANAGE USERS", classes="section-label")
                yield Button("↺ Refresh User List", id="btn-list",   variant="default")
                yield Input(placeholder="Name to delete…", id="del-name")
                yield Button("Delete User",             id="btn-delete", variant="error")

                yield Static("◈ SYSTEM", classes="section-label")
                yield Button("Fetch System Logs", id="btn-logs",      variant="warning")
                yield Button("Clear Event Log",   id="btn-clear",     variant="default")
                yield Button("⟳ Reconnect",       id="btn-reconnect", variant="primary")

            # ── Main panel ────────────────────────────────────────────────────
            with Vertical(id="main-area"):
                yield Static("Event Log", classes="section-label")
                yield RichLog(id="log-area", highlight=True, markup=True, wrap=True)

                yield Static("Enrolled Users", classes="section-label")
                yield DataTable(id="user-table")

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

        self._log("[yellow]Searching for biometric server via mDNS…[/yellow]")
        self._zc       = Zeroconf()
        self._listener = _DiscoveryListener(self._on_server_found)
        self._browser  = ServiceBrowser(self._zc, SERVICE_TYPE, self._listener)

    # ── mDNS callback — runs in zeroconf thread ───────────────────────────────
    def _on_server_found(self, url: str):
        self.server_url = url
        self.call_from_thread(self._log, f"[cyan]Server found → [bold]{url}[/bold][/cyan]")
        self.call_from_thread(self._kick_connect)

    def _kick_connect(self):
        self.run_worker(self._connect(), exclusive=True, name="ws-connect")

    # ── WebSocket connect ─────────────────────────────────────────────────────
    async def _connect(self):
        try:
            self.ws = await websockets.connect(
                self.server_url, ping_interval=20, ping_timeout=10)
            self._set_status("ONLINE", "online")
            self._log("[green]Connected.[/green]")
            self.notify("Server online", severity="information")
            await self._refresh_users()
        except Exception as exc:
            self._set_status("OFFLINE", "offline")
            self._log(f"[red]Connection failed: {exc}[/red]")

    # ── Low-level send/receive ────────────────────────────────────────────────
    async def _send(self, payload: dict) -> dict | None:
        """
        Send one command; consume any 'progress' messages transparently;
        return the final response dict.  Returns None on any error.
        """
        if not self.ws:
            self.notify("Not connected.", severity="error")
            return None
        try:
            await self.ws.send(json.dumps(payload))
            while True:
                # 120s ceiling: a face+voice "register" runs two capture windows
                # back-to-back. Progress heartbeats from the server reset this
                # timer on every message, so this only fires on a genuine stall.
                raw = await asyncio.wait_for(self.ws.recv(), timeout=120.0)
                msg = json.loads(raw)
                if msg.get("status") == "progress":
                    step = msg.get("step", "").upper()
                    txt  = msg.get("message", "")
                    self._log(f"[yellow]  ⟳  {step}: {txt}[/yellow]")
                else:
                    return msg
        except asyncio.TimeoutError:
            self._log("[red]Request timed out (120 s).[/red]")
            return None
        except (ConnectionClosed, WebSocketException) as exc:
            self._set_status("OFFLINE", "offline")
            self.ws = None
            # Forget this server in the discovery listener so mDNS can
            # re-announce and we can auto-reconnect (RPi Wi-Fi drops are common).
            if self.server_url:
                self._listener._seen.discard(self.server_url)
            self._log(f"[red]Connection lost: {exc}[/red]")
            return None
        except Exception as exc:
            self._log(f"[red]Unexpected error: {exc}[/red]")
            return None

    # ── Button routing ────────────────────────────────────────────────────────
    async def on_button_pressed(self, event: Button.Pressed):
        async with self._cmd_lock:
            await self._handle(event.button.id)

    async def _handle(self, btn: str):

        # ──────────────────────────────────────────────────────────────────────
        # REGISTER  (face + voice)
        # ──────────────────────────────────────────────────────────────────────
        if btn == "btn-register":
            name = self._reg_name()
            if not name:
                return
            self._action(f"Registering {name}…")
            self._log(f"\n[cyan]► Register [bold]{name}[/bold] (face then voice)…[/cyan]")
            res = await self._send({"command": "register", "name": name})
            if res is None:
                return self._action("Ready")
            face_ok  = res.get("face",  {}).get("status") == "success"
            voice_ok = res.get("voice", {}).get("status") == "success"
            if res.get("status") in ("success", "partial"):
                face_err  = res.get("face",  {}).get("message", "")
                voice_err = res.get("voice", {}).get("message", "")
                self._log(
                    f"[{'green' if face_ok and voice_ok else 'yellow'}]"
                    f"  Face:  {'✓ enrolled' if face_ok  else f'✗ {face_err}'}\n"
                    f"  Voice: {'✓ enrolled' if voice_ok else f'✗ {voice_err}'}[/]"
                )
                self.notify(
                    f"{'Both modalities' if face_ok and voice_ok else 'Partial'} enrolled.",
                    title=name, severity="information"
                )
                self.query_one("#reg-name", Input).value = ""
                await self._refresh_users()
            else:
                self._log(f"[red]  ✘ {res.get('message','Unknown error')}[/red]")
                self.notify(res.get("message",""), title="Register failed", severity="error")
            self._action("Ready")

        # ──────────────────────────────────────────────────────────────────────
        # REGISTER  (face only)
        # ──────────────────────────────────────────────────────────────────────
        elif btn == "btn-register-face":
            name = self._reg_name()
            if not name:
                return
            self._action(f"Enrolling face: {name}…")
            self._log(f"\n[cyan]► Enrol face for [bold]{name}[/bold]…[/cyan]")
            res = await self._send({"command": "register_face", "name": name})
            self._show_enroll_result(res, "Face", name)
            if res and res.get("status") == "success":
                await self._refresh_users()
            self._action("Ready")

        # ──────────────────────────────────────────────────────────────────────
        # REGISTER  (voice only)
        # ──────────────────────────────────────────────────────────────────────
        elif btn == "btn-register-voice":
            name = self._reg_name()
            if not name:
                return
            self._action(f"Enrolling voice: {name}…")
            self._log(f"\n[cyan]► Enrol voice for [bold]{name}[/bold] — speak now…[/cyan]")
            res = await self._send({"command": "register_voice", "name": name})
            self._show_enroll_result(res, "Voice", name)
            if res and res.get("status") == "success":
                await self._refresh_users()
            self._action("Ready")

        # ──────────────────────────────────────────────────────────────────────
        # AUTH  (face)
        # ──────────────────────────────────────────────────────────────────────
        elif btn == "btn-auth-face":
            self._action("Face auth — look at camera…")
            self._log("\n[cyan]► Face authentication — look at camera…[/cyan]")
            res = await self._send({"command": "auth_face"})
            self._show_auth_result(res, method="FACE")
            self._action("Ready")

        # ──────────────────────────────────────────────────────────────────────
        # AUTH  (voice)
        # ──────────────────────────────────────────────────────────────────────
        elif btn == "btn-auth-voice":
            self._action("Voice auth — speak now…")
            self._log("\n[cyan]► Voice authentication — speak now…[/cyan]")
            res = await self._send({"command": "auth_voice"})
            self._show_auth_result(res, method="VOICE")
            self._action("Ready")

        # ──────────────────────────────────────────────────────────────────────
        # LIST USERS
        # ──────────────────────────────────────────────────────────────────────
        elif btn == "btn-list":
            self._action("Fetching user list…")
            res = await self._send({"command": "get_identities"})
            if res:
                await self._populate_user_table(res)
            self._action("Ready")

        # ──────────────────────────────────────────────────────────────────────
        # DELETE USER
        # ──────────────────────────────────────────────────────────────────────
        elif btn == "btn-delete":
            name = self.query_one("#del-name", Input).value.strip()
            if not name:
                self.notify("Enter a name to delete.", severity="warning")
                return
            self._action(f"Deleting {name}…")
            self._log(f"\n[yellow]► Delete [bold]{name}[/bold]…[/yellow]")
            res = await self._send({"command": "delete", "name": name})
            if res is None:
                return self._action("Ready")
            face_res  = res.get("face",  {})
            voice_res = res.get("voice", {})
            face_ok   = face_res.get("status")  == "success"
            voice_ok  = voice_res.get("status") == "success"
            if face_ok or voice_ok:
                self._log(
                    f"[green]  Deleted [bold]{name}[/bold] — "
                    f"Face: {'✓' if face_ok else '—'}  "
                    f"Voice: {'✓' if voice_ok else '—'}[/green]"
                )
                self.query_one("#del-name", Input).value = ""
                self.notify(f"Deleted {name}.", severity="information")
                await self._refresh_users()
            else:
                face_msg  = face_res.get("message",  "")
                voice_msg = voice_res.get("message", "")
                reason    = face_msg or voice_msg or "User not found in any database."
                self._log(f"[red]  ✘ {reason}[/red]")
                self.notify(reason, title="Delete failed", severity="error")
            self._action("Ready")

        # ──────────────────────────────────────────────────────────────────────
        # FETCH LOGS
        # ──────────────────────────────────────────────────────────────────────
        elif btn == "btn-logs":
            self._action("Fetching logs…")
            res = await self._send({"command": "get_logs"})
            if res is None:
                return self._action("Ready")
            entries = res.get("data", [])
            if not entries:
                self._log("[yellow]No log entries yet.[/yellow]")
            else:
                self._log(f"\n[bold cyan]─── SYSTEM LOG  ({len(entries)} entries) ───[/bold cyan]")
                # Header row
                self._log(
                    "[bold dim]  TIME      WORKER  STATUS      USER                 CONFIDENCE[/bold dim]"
                )
                for entry in entries:
                    self._render_log_entry(entry)
                self._log("[bold cyan]─── END OF LOG ───[/bold cyan]")
            self._action("Ready")

        # ──────────────────────────────────────────────────────────────────────
        # CLEAR LOG
        # ──────────────────────────────────────────────────────────────────────
        elif btn == "btn-clear":
            self.query_one("#log-area", RichLog).clear()

        # ──────────────────────────────────────────────────────────────────────
        # RECONNECT  (manual recovery after a Wi-Fi/network drop)
        # ──────────────────────────────────────────────────────────────────────
        elif btn == "btn-reconnect":
            self._log("[yellow]Reconnecting…[/yellow]")
            # Drop any half-open socket.
            if self.ws is not None:
                try:
                    await self.ws.close()
                except Exception:
                    pass
                self.ws = None
            # Forget discovered servers so mDNS can re-announce them.
            self._listener._seen.clear()
            self._set_status("SEARCHING…", "searching")
            if self.server_url:
                # We already know an address — try it straight away.
                self._kick_connect()
            else:
                self._log("[yellow]Waiting for server via mDNS…[/yellow]")

    # ── Enroll result helper ──────────────────────────────────────────────────
    def _show_enroll_result(self, res: dict | None, modality: str, name: str):
        if res is None:
            return
        if res.get("status") == "success":
            self._log(f"[green]  ✓ {modality} enrolled for [bold]{name}[/bold][/green]")
            self.notify(f"{modality} enrolled: {name}", severity="information")
        else:
            msg = res.get("message", "Unknown error")
            self._log(f"[red]  ✘ {modality} enrol failed: {msg}[/red]")
            self.notify(msg, title=f"{modality} enrol failed", severity="error")

    # ── Auth result helper ────────────────────────────────────────────────────
    def _show_auth_result(self, res: dict | None, method: str):
        if res is None:
            return
        if res.get("status") != "success":
            msg = res.get("message", "Error")
            self._log(f"[red]  ✘ {method} error: {msg}[/red]")
            self.notify(msg, title=f"{method} error", severity="error")
            return
        granted = res.get("granted", False)
        user    = res.get("user",    "—")
        score   = res.get("score",   0.0)
        ts      = datetime.now().strftime("%H:%M:%S")
        if granted:
            self._log(
                f"[green]  ✓  [{ts}]  {method}  │  GRANTED  │  "
                f"[bold]{user}[/bold]  │  confidence: {score:.4f}[/green]"
            )
            self.notify(
                f"Welcome, {user}!  ({score:.2f})",
                title=f"{method} — ACCESS GRANTED",
                severity="information",
            )
        else:
            self._log(
                f"[red]  ✗  [{ts}]  {method}  │  DENIED   │  "
                f"{user}  │  confidence: {score:.4f}[/red]"
            )
            self.notify("Access denied.", title=f"{method} — DENIED", severity="warning")

    # ── Log entry renderer ────────────────────────────────────────────────────
    def _render_log_entry(self, entry: dict):
        """
        Parses an Orchestra log entry:
          {"time": "19:23:45", "worker": "face"|"voice",
           "event": "GRANTED Alice score=0.8920"}

        Displays: timestamp · worker · status · user · confidence score
        """
        ts      = entry.get("time",   "?")
        worker  = (entry.get("worker") or "?").upper()
        event   = entry.get("event",  "").strip()

        # ── Parse event string ─────────────────────────────────────────────
        # Formats seen from Orchestra:
        #   "GRANTED Alice score=0.8920"
        #   "DENIED  Alice score=0.4500"
        #   "AMBIGUOUS Alice score=0.7800 ..."
        #   "Enrolled 'Alice'"
        #   "Deleted  'Alice'"
        status     = "—"
        user       = "—"
        score_str  = "—"
        color      = "dim"

        if event.upper().startswith("GRANTED"):
            status = "GRANTED"
            color  = "green"
        elif event.upper().startswith("DENIED") or event.upper().startswith("AMBIGUOUS"):
            status = event.split()[0].upper()
            color  = "red"
        elif event.lower().startswith("enrolled"):
            status = "ENROLLED"
            color  = "cyan"
        elif event.lower().startswith("deleted"):
            status = "DELETED"
            color  = "yellow"
        else:
            status = event[:12]
            color  = "dim"

        # Extract user name (2nd token, strip quotes)
        parts = event.replace("'", "").split()
        if len(parts) >= 2:
            user = parts[1]

        # Extract score  (looks like "score=0.8920" or a bare float at end)
        for token in parts:
            if token.startswith("score="):
                try:
                    score_str = f"{float(token.split('=')[1]):.4f}"
                except ValueError:
                    pass

        self._log(
            f"[{color}]  {ts}  {worker:<5}  "
            f"{status:<10}  {user:<20}  {score_str}[/{color}]"
        )

    # ── User table ────────────────────────────────────────────────────────────
    async def _refresh_users(self):
        res = await self._send({"command": "get_identities"})
        if res:
            await self._populate_user_table(res)

    async def _populate_user_table(self, res: dict):
        tbl = self.query_one("#user-table", DataTable)
        tbl.clear()

        if res.get("status") != "success":
            self._log(f"[red]get_identities error: {res.get('message','')}[/red]")
            return

        all_names  = res.get("data",       [])
        face_only  = set(res.get("face_only",  []))
        voice_only = set(res.get("voice_only", []))

        if not all_names:
            self._log("[yellow]No users enrolled yet.[/yellow]")
            return

        for name in sorted(all_names):
            has_face  = name not in voice_only
            has_voice = name not in face_only
            tbl.add_row(name,
                        "✓" if has_face  else "—",
                        "✓" if has_voice else "—")

        n_both  = len(all_names) - len(face_only) - len(voice_only)
        self._log(
            f"[cyan]{len(all_names)} user(s)  —  "
            f"both: {n_both}  face-only: {len(face_only)}  "
            f"voice-only: {len(voice_only)}[/cyan]"
        )

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _reg_name(self) -> str:
        name = self.query_one("#reg-name", Input).value.strip()
        if not name:
            self.notify("Enter a name first.", severity="warning")
        return name

    def _log(self, markup: str):
        self.query_one("#log-area", RichLog).write(markup)

    def _set_status(self, text: str, css_class: str):
        lbl = self.query_one("#server-status", Label)
        lbl.update(text)
        lbl.set_classes(css_class)

    def _action(self, text: str):
        self.query_one("#last-action", Label).update(text)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    async def on_unmount(self):
        if self.ws:
            await self.ws.close()
        self._zc.close()


if __name__ == "__main__":
    BiometricClient().run()
