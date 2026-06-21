#!/usr/bin/env python3
"""
One-time migration: lowercases all keys in faces_db.pkl so face identities
line up with the (already-lowercase) voice identities in
voiceprint_database_onnx/.

Run this ONCE, with the orchestrator service STOPPED, from inside your
Orchestra/ directory:
    python3 migrate_faces_db_casing.py

Safety:
  - Makes a faces_db.pkl.bak backup before writing anything.
  - If two existing keys would collapse onto the same lowercase name
    (e.g. both "Alice" and "alice" are already in the file), it stops
    and reports the collision instead of silently merging or
    overwriting one — you decide which embedding to keep, then re-run.
"""
import pickle
import shutil
import sys
from pathlib import Path

DB_PATH = Path("faces_db.pkl")

if not DB_PATH.exists():
    sys.exit(f"[!] {DB_PATH} not found in the current directory. "
              f"Run this from inside Orchestra/.")

with open(DB_PATH, "rb") as f:
    db = pickle.load(f)

if not isinstance(db, dict):
    sys.exit(f"[!] Expected a name -> embedding dict in {DB_PATH}, got "
              f"{type(db).__name__} instead. Stopping — this script only "
              f"knows how to handle that shape of data.")

lowered    = {}
collisions = {}
for name, embedding in db.items():
    key = name.strip().lower()
    if key in lowered and key not in collisions:
        collisions[key] = [k for k in db if k.strip().lower() == key]
    lowered[key] = embedding

if collisions:
    print("[!] Aborting — these existing names collide once lowercased:")
    for key, originals in collisions.items():
        print(f"      {originals}  ->  '{key}'")
    print("\n    These look like the same person enrolled twice under")
    print("    different capitalization. Manually delete the one you")
    print("    don't want (via the app's Delete User) and re-run this.")
    sys.exit(1)

changed = [name for name in db if name != name.strip().lower()]
if not changed:
    print("[OK] All keys already lowercase — nothing to do.")
    sys.exit(0)

backup_path = DB_PATH.with_suffix(".pkl.bak")
shutil.copy2(DB_PATH, backup_path)
print(f"[OK] Backed up original to {backup_path}")

with open(DB_PATH, "wb") as f:
    pickle.dump(lowered, f)

print(f"[OK] Lowercased {len(changed)} name(s) in {DB_PATH}:")
for old in changed:
    print(f"      '{old}'  ->  '{old.strip().lower()}'")

print("\nDone. Restart the orchestrator service.")