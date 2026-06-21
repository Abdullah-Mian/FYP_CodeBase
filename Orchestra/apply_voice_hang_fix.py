import sys
PATH = "rpi4_ecapa_voice_biometric_v2.py"
with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

REPLACEMENTS = [
    (
        "1) add on_start param to receive_session()",
        '    def receive_session(self, min_seconds, max_seconds, label="audio"):\n'
        '        """\n'
        '        Block until a complete audio session arrives.\n'
        '        Enforces min_seconds (reject if shorter) and max_seconds (truncate).\n'
        '        Returns raw PCM bytes, or None on failure / audio too short.\n'
        '        """\n',
        '    def receive_session(self, min_seconds, max_seconds, label="audio",\n'
        '                        on_start=None):\n'
        '        """\n'
        '        Block until a complete audio session arrives.\n'
        '        Enforces min_seconds (reject if shorter) and max_seconds (truncate).\n'
        '        Returns raw PCM bytes, or None on failure / audio too short.\n'
        '\n'
        '        on_start: optional zero-arg callback fired exactly once, the\n'
        '        instant the first real audio frame arrives (i.e. the clap that\n'
        '        triggered the ESP32 to start streaming). Lets the caller flip a\n'
        '        TFT display from a standby/"clap to start" message straight to\n'
        '        "Listening...", in lockstep with when audio actually begins.\n'
        '        """\n',
    ),
    (
        "2) fix the hang: return None instead of looping forever pre-clap",
        '            if frame is None:\n'
        '                if started and accum:\n'
        '                    break     # timeout after data -> end session\n'
        '                continue      # timeout before first frame -> keep waiting\n',
        '            if frame is None:\n'
        '                if started and accum:\n'
        '                    break     # timeout after data -> end session\n'
        '                # Nobody has started talking yet within this read\'s\n'
        '                # timeout window. Give up THIS attempt instead of\n'
        '                # looping forever -- the caller (voice_worker_process\'s\n'
        '                # autonomous loop) needs control back so it can check\n'
        '                # cmd_q for a queued command (delete/enroll/verify/...).\n'
        '                return None   # caller treats this exactly like "no audio"\n',
    ),
    (
        "3) fire on_start() the moment real audio begins",
        '            if not started:\n'
        '                started = True\n'
        '                print(f"[UART] Session started  {datetime.now().strftime(\'%H:%M:%S\')}")\n'
        '\n'
        '            accum.extend(frame)\n',
        '            if not started:\n'
        '                started = True\n'
        '                print(f"[UART] Session started  {datetime.now().strftime(\'%H:%M:%S\')}")\n'
        '                if on_start is not None:\n'
        '                    try:\n'
        '                        on_start()\n'
        '                    except Exception as _e:\n'
        '                        print(f"[UART] on_start callback error (ignored): {_e}")\n'
        '\n'
        '            accum.extend(frame)\n',
    ),
]

missing = [label for label, old, _ in REPLACEMENTS if old not in src]
if missing:
    print("[!] Aborting -- could not find expected text for:")
    for m in missing:
        print(f"      - {m}")
    sys.exit(1)

dup = [label for label, old, _ in REPLACEMENTS if src.count(old) > 1]
if dup:
    print("[!] Aborting -- multiple matches for:")
    for m in dup:
        print(f"      - {m}")
    sys.exit(1)

for label, old, new in REPLACEMENTS:
    src = src.replace(old, new, 1)
    print(f"[OK] Applied: {label}")

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)
print("done")