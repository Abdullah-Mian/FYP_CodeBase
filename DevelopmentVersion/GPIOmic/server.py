import asyncio
import json
import socket
from websockets.server import serve
from zeroconf import IPVersion, ServiceInfo, Zeroconf

# --- Mock Database ---
db = {
    "users": ["Faizan", "Admin"],
    "logs": [
        {"user": "Admin", "time": "2026-05-04 05:00", "status": "System Startup"}
    ]
}

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

async def handle_client(websocket):
    client_addr = websocket.remote_address
    print(f"[+] Client connected: {client_addr}")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            command = data.get("command")
            
            # Logic Router
            if command == "register":
                name = data.get("name", "Unknown")
                if name not in db["users"]:
                    db["users"].append(name)
                print(f"[DB] Registered new identity: {name}")
                response = {"status": "success", "user": name}
            
            elif command in ["auth_face", "auth_voice"]:
                user = db["users"][0] # Mocking a match
                db["logs"].append({"user": user, "time": "06:30", "status": f"Verified via {command}"})
                print(f"[AUTH] {command} success for {user}")
                response = {"status": "success", "user": user}
                
            elif command == "get_logs":
                response = {"status": "success", "data": db["logs"]}
                
            elif command == "get_identities":
                response = {"status": "success", "data": db["users"]}
                
            else:
                response = {"status": "error", "message": "Invalid Command"}

            await websocket.send(json.dumps(response))
            
    except Exception as e:
        print(f"[-] Connection error: {e}")
    finally:
        print(f"[-] Client {client_addr} disconnected")

async def main():
    ip = get_local_ip()
    port = 8765
    zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
    info = ServiceInfo(
        "_biometric-auth._tcp.local.",
        "BiometricServer._biometric-auth._tcp.local.",
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties={'version': '1.0.0'}
    )
    
    await zeroconf.async_register_service(info)
    print(f"[*] Server Live at ws://{ip}:{port}")
    
    async with serve(handle_client, "0.0.0.0", port):
        try:
            await asyncio.Future()
        finally:
            await zeroconf.async_unregister_all_services()
            zeroconf.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Shutting down...")