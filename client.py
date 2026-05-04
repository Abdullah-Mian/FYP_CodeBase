import asyncio
import json
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Static, Input, Log, Label
from textual.containers import Vertical, Horizontal
from zeroconf import Zeroconf, ServiceBrowser
import websockets

class DiscoveryListener:
    def __init__(self, callback):
        self.callback = callback

    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if info:
            addresses = [f"{addr[0]}.{addr[1]}.{addr[2]}.{addr[3]}" for addr in info.addresses]
            url = f"ws://{addresses[0]}:{info.port}"
            self.callback(url)

class BiometricClient(App):
    CSS = """
    Screen { layout: grid; grid-size: 2; grid-columns: 1fr 3fr; }
    #sidebar { background: $panel; border-right: tall $primary; padding: 1; }
    #display-area { padding: 1; }
    Button { width: 100%; margin-top: 1; }
    Input { margin-bottom: 1; border: tall $accent; }
    Log { background: $surface; border: solid $primary; height: 1fr; margin-top: 1; }
    .label { text-style: bold; color: $secondary; margin-top: 1; }
    
    #status-bar {
        background: $boost;
        dock: bottom;
        height: 1;
        padding: 0 1;
    }
    .online { color: green; text-style: bold; }
    .offline { color: red; text-style: bold; }
    """

    def __init__(self):
        super().__init__()
        self.ws = None
        self.server_url = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="sidebar"):
            yield Static("NEW REGISTRATION", classes="label")
            yield Input(placeholder="User Name...", id="input-name")
            yield Button("Enroll Voice", variant="primary", id="reg_voice")
            yield Button("Enroll Face", variant="primary", id="reg_face")
            yield Static("AUTHENTICATION", classes="label")
            yield Button("Verify Voice", variant="success", id="auth_voice")
            yield Button("Verify Face", variant="success", id="auth_face")
            yield Static("SYSTEM TOOLS", classes="label")
            yield Button("Get Logs", variant="warning", id="get_logs")
            yield Button("List Users", variant="default", id="get_ids")
        
        with Vertical(id="display-area"):
            yield Static("System Output", classes="label")
            yield Log(id="main-log")
        
        with Horizontal(id="status-bar"):
            yield Label("System Status: ")
            yield Label("OFFLINE", id="server-status", classes="offline")
            
        yield Footer()

    def on_mount(self):
        self.write_log("Searching for server on LAN...")
        self.zeroconf = Zeroconf()
        self.listener = DiscoveryListener(self.on_server_found)
        self.browser = ServiceBrowser(self.zeroconf, "_biometric-auth._tcp.local.", self.listener)

    def on_server_found(self, url):
        self.server_url = url
        self.write_log(f"Server found at {url}")
        self.run_worker(self.connect_to_server())

    async def connect_to_server(self):
        try:
            self.ws = await websockets.connect(self.server_url)
            self.write_log("Connected to WebSocket Server.")
            status_label = self.query_one("#server-status")
            status_label.update("ONLINE")
            status_label.set_classes("online")
            self.notify("Server Connection Established", severity="information")
        except Exception as e:
            self.write_log(f"Connection error: {e}")

    async def talk_to_server(self, command, extra_data=None):
        if not self.ws:
            self.notify("Not connected to server!", severity="error")
            return
        
        payload = {"command": command}
        if extra_data: payload.update(extra_data)
            
        await self.ws.send(json.dumps(payload))
        response = await self.ws.recv()
        return json.loads(response)

    def write_log(self, message):
        log_widget = self.query_one("#main-log")
        log_widget.write_line(f"> {message}")
        log_widget.scroll_end()

    async def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id
        
        if btn_id.startswith("reg"):
            name_input = self.query_one("#input-name")
            name = name_input.value
            if not name.strip():
                self.notify("Please enter a name", severity="error")
                return
                
            res = await self.talk_to_server("register", {"name": name})
            self.write_log(f"Registered: {res['user']} (success)")
            name_input.value = "" # Clear the field
            self.notify(f"Enrolled {res['user']}", title="Registration")
        
        elif btn_id in ["auth_voice", "auth_face"]:
            res = await self.talk_to_server(btn_id)
            self.write_log(f"Auth Success: Verified as {res['user']}")
            self.notify(f"Welcome back, {res['user']}!", title="Authentication")

        elif btn_id == "get_logs":
            res = await self.talk_to_server("get_logs")
            self.write_log("--- SYSTEM LOGS ---")
            for entry in res['data']:
                self.write_log(f"[{entry['time']}] {entry['user']}: {entry['status']}")

        elif btn_id == "get_ids":
            res = await self.talk_to_server("get_identities")
            self.write_log(f"Registered Users: {', '.join(res['data'])}")

if __name__ == "__main__":
    BiometricClient().run()