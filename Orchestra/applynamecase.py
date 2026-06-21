#!/usr/bin/env python3
"""
Patches Orchestra.py so the FACE worker lowercases names the same way the
VOICE worker already does. Right now face_worker keeps whatever case you
typed ("Faizy") while voice_worker forces lowercase ("faizy") — so the same
person ends up as two different identities for list/delete purposes.

Run this from inside your Orchestra/ directory, with the orchestrator
service STOPPED:
    python3 apply_name_casing_fix.py

It edits Orchestra.py in place. If the target text isn't found EXACTLY as
expected, it aborts and writes nothing.

IMPORTANT: this only changes behavior for NEW enroll/delete/lookup calls
going forward. If you already have face entries enrolled with capital
letters in faces_db.pkl, run migrate_faces_db_casing.py too (separately
provided) to lowercase the existing data so it lines up with this fix.
"""
import sys

PATH = "Orchestra.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

OLD = (
    '        nonlocal db   # needed by the reload_db branch\n'
    '        op   = cmd.get("op")\n'
    '        name = cmd.get("name", "").strip()\n'
)
NEW = (
    '        nonlocal db   # needed by the reload_db branch\n'
    '        op   = cmd.get("op")\n'
    '        name = cmd.get("name", "").strip().lower()   # match voice worker\'s casing\n'
)

if OLD not in src:
    print("[!] Aborting — expected text not found in Orchestra.py.")
    print("    No changes were written. The file may already be patched,")
    print("    or has been edited since this script was generated.")
    sys.exit(1)

if src.count(OLD) > 1:
    print(f"[!] Aborting — found {src.count(OLD)} matches, expected exactly 1.")
    print("    Refusing to guess which one to patch.")
    sys.exit(1)

src = src.replace(OLD, NEW, 1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("[OK] Patched face_worker_process to lowercase names on enroll/delete/lookup.")
print("\nNext steps:")
print("  1. If you have existing capitalized names in faces_db.pkl, run")
print("     migrate_faces_db_casing.py to bring them in line.")
print("  2. Restart the orchestrator service.")