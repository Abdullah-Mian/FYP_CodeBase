#!/usr/bin/env python3
"""
Patches Orchestra.py so the voice worker's autonomous loop HOLDS the
GRANTED / DENIED / error text on the TFT for a couple seconds, instead of
overwriting it with "Listening…" on the very next loop iteration.

WHY: the autonomous loop sets _set_voice(f"{tag}: {name} ({score})") and
then immediately loops back to the top, where _set_voice("Listening…") is
called again before the next uart.receive_session() call. There's no pause
in between, so the result text is on screen for milliseconds — that's why
the TFT only ever appears to show "Listening…" / "Processing…".

Run this from inside your Orchestra/ directory:
    python3 apply_tft_fix.py

It edits Orchestra.py in place. If any of the target blocks of text aren't
found EXACTLY as expected (e.g. you've already modified the file), it
aborts and writes nothing — so it's safe to run, it won't half-apply or
corrupt the file. Diff it / read the printed summary before restarting
the service.
"""
import sys

PATH = "Orchestra.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

REPLACEMENTS = [
    (
        "1) add VOICE_RESULT_HOLD_S constant",
        'AUTO_FACE_RETRY_S   = 2.0     # retry period while face is UNKNOWN/DENIED\n'
        'GRANTED_IOU_THRESH  = 0.3     # IOU above this => treat as "still same face"\n',
        'AUTO_FACE_RETRY_S   = 2.0     # retry period while face is UNKNOWN/DENIED\n'
        'GRANTED_IOU_THRESH  = 0.3     # IOU above this => treat as "still same face"\n'
        '\n'
        '# How long a GRANTED/DENIED/error message stays on the TFT before the voice\n'
        '# worker goes back to "Listening…". Without this hold, the autonomous loop\n'
        '# overwrites the result almost instantly and it is never actually readable.\n'
        'VOICE_RESULT_HOLD_S = 2.5\n',
    ),
    (
        "2) hold after a GRANTED/DENIED verify result",
        '            _log_event(shared_logs, "voice",\n'
        '                       f"[AUTO] {tag} {best_name} score={best_score:.4f}")\n'
        '            _set_voice(f"{tag}: {best_name or \'unknown\'} ({best_score:.2f})")\n'
        '\n'
        '    except KeyboardInterrupt:',
        '            _log_event(shared_logs, "voice",\n'
        '                       f"[AUTO] {tag} {best_name} score={best_score:.4f}")\n'
        '            _set_voice(f"{tag}: {best_name or \'unknown\'} ({best_score:.2f})")\n'
        '            time.sleep(VOICE_RESULT_HOLD_S)\n'
        '\n'
        '    except KeyboardInterrupt:',
    ),
    (
        "3) hold after 'No speakers enrolled'",
        '                _set_voice("No speakers enrolled")\n'
        '                continue',
        '                _set_voice("No speakers enrolled")\n'
        '                time.sleep(VOICE_RESULT_HOLD_S)\n'
        '                continue',
    ),
    (
        "4) hold after an audio quality failure",
        '                _set_voice(f"Audio quality issue — {e}")\n'
        '                continue',
        '                _set_voice(f"Audio quality issue — {e}")\n'
        '                time.sleep(VOICE_RESULT_HOLD_S)\n'
        '                continue',
    ),
]

missing = [label for label, old, _ in REPLACEMENTS if old not in src]
if missing:
    print("[!] Aborting — could not find expected text for:")
    for m in missing:
        print(f"      - {m}")
    print("    No changes were written. The file may already be patched,")
    print("    or has been edited since this script was generated.")
    sys.exit(1)

for label, old, new in REPLACEMENTS:
    src = src.replace(old, new, 1)
    print(f"[OK] Applied: {label}")

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("\nDone. Review the changes, then restart the orchestrator service")
print("(e.g. `sudo systemctl restart orchestra` or however you run it)")
print("for the fix to take effect.")