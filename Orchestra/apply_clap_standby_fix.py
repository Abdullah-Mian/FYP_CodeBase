#!/usr/bin/env python3
"""
Adds a real "standby, waiting for a clap" state to the TFT voice display,
instead of showing "Listening…" before any audio has actually started.

Requires apply_voice_hang_fix.py (the combined hang-fix + on_start callback
patch for rpi4_ecapa_voice_biometric_v2.py) to be applied FIRST — this
script wires Orchestra.py up to the new `on_start` parameter that adds.

WHAT THIS CHANGES
─────────────────
  • New VOICE_STANDBY_TEXT constant: "Clap to start voice"
  • Autonomous loop: shows VOICE_STANDBY_TEXT instead of "Listening…" while
    waiting for a clap; flips to "Listening…" the instant real audio starts
    arriving (via the on_start callback) — not before.
  • Same treatment for manual Enroll-voice and Verify-voice commands, for
    consistency, so every voice flow reads "clap to start" -> "speak now /
    listening" -> "Processing…" -> result, instead of jumping straight to
    "speak now" before you've actually clapped.

No countdown timer is added (per your call — the firmware would need
changes to support a real pre-roll, so this just reflects the clap moment
honestly instead of pretending to listen before it happens).

Run from inside your Orchestra/ directory, with the orchestrator service
STOPPED:
    python3 apply_clap_standby_fix.py
"""
import sys

PATH = "Orchestra.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

REPLACEMENTS = [
    (
        "1) add VOICE_STANDBY_TEXT constant",
        'VOICE_RESULT_HOLD_S = 2.5\n',
        'VOICE_RESULT_HOLD_S = 2.5\n'
        '\n'
        '# Standby label shown on the TFT while the voice worker is waiting for a\n'
        '# clap to trigger the ESP32 into streaming audio (i.e. before any session\n'
        '# has started). The instant real audio starts arriving, the display flips\n'
        '# to "Listening..." via UARTReceiver.receive_session()\'s on_start callback\n'
        '# -- so the screen never claims to be listening before a clap happened.\n'
        'VOICE_STANDBY_TEXT = "Clap to start voice"\n',
    ),
    (
        "2) enroll: standby -> speak now, only once audio actually starts",
        '            _set_voice(f"Enrolling \'{name}\' — speak now…")\n'
        '            raw = uart.receive_session(ENROLL_MIN_S, ENROLL_MAX_S,\n'
        '                                       label=f"enrollment \'{name}\'")\n',
        '            _set_voice(f"Enrolling \'{name}\' — {VOICE_STANDBY_TEXT}")\n'
        '            raw = uart.receive_session(\n'
        '                ENROLL_MIN_S, ENROLL_MAX_S,\n'
        '                label=f"enrollment \'{name}\'",\n'
        '                on_start=lambda: _set_voice(f"Enrolling \'{name}\' — speak now…"))\n',
    ),
    (
        "3) verify: standby -> listening, only once audio actually starts",
        '            _set_voice("Listening for verification…")\n'
        '            raw = uart.receive_session(VERIFY_MIN_S, VERIFY_MAX_S,\n'
        '                                       label="verification")\n',
        '            _set_voice(f"Verify — {VOICE_STANDBY_TEXT}")\n'
        '            raw = uart.receive_session(\n'
        '                VERIFY_MIN_S, VERIFY_MAX_S,\n'
        '                label="verification",\n'
        '                on_start=lambda: _set_voice("Listening for verification…"))\n',
    ),
    (
        "4) autonomous loop: standby -> listening, only once audio actually starts",
        '            log.debug("[VOICE][AUTO] Listening on UART…")\n'
        '            _set_voice("Listening…")\n'
        '            raw = uart.receive_session(\n'
        '                min_seconds=VERIFY_MIN_S,\n'
        '                max_seconds=VERIFY_MAX_S,\n'
        '                label="auto-identify")\n'
        '\n'
        '            # If a command arrived while we were blocking, skip inference\n'
        '            if not cmd_q.empty():\n'
        '                continue\n'
        '\n'
        '            if raw is None:\n'
        '                _set_voice("Idle")\n'
        '                continue   # timeout / short session — loop again\n',
        '            log.debug("[VOICE][AUTO] Waiting for clap…")\n'
        '            _set_voice(VOICE_STANDBY_TEXT)\n'
        '            raw = uart.receive_session(\n'
        '                min_seconds=VERIFY_MIN_S,\n'
        '                max_seconds=VERIFY_MAX_S,\n'
        '                label="auto-identify",\n'
        '                on_start=lambda: _set_voice("Listening…"))\n'
        '\n'
        '            # If a command arrived while we were blocking, skip inference\n'
        '            if not cmd_q.empty():\n'
        '                continue\n'
        '\n'
        '            if raw is None:\n'
        '                _set_voice(VOICE_STANDBY_TEXT)\n'
        '                continue   # timeout / short session — loop again\n',
    ),
]

missing = [label for label, old, _ in REPLACEMENTS if old not in src]
if missing:
    print("[!] Aborting -- could not find expected text for:")
    for m in missing:
        print(f"      - {m}")
    print("    No changes were written. The file may already be patched,")
    print("    or has been edited since this script was generated.")
    sys.exit(1)

dup = [label for label, old, _ in REPLACEMENTS if src.count(old) > 1]
if dup:
    print("[!] Aborting -- multiple matches found for:")
    for m in dup:
        print(f"      - {m}")
    print("    Refusing to guess which one to patch.")
    sys.exit(1)

for label, old, new in REPLACEMENTS:
    src = src.replace(old, new, 1)
    print(f"[OK] Applied: {label}")

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("\nDone. Review with: diff -u Orchestra.py.bak Orchestra.py")
print("Restart the orchestrator service for this to take effect.")