"""
rust_wrapper.py
Thin Python interface to the compiled Rust face embedding binary.

The Rust binary:
  - Accepts: <image_path>
  - Outputs: a Vec<f32> printed with Rust's {:?} format, e.g.  [0.123, -0.456, ...]
    which is valid JSON and is parsed directly.
  - Already applies L2 normalisation internally.
    We normalise again defensively in case of numerical drift.
"""

import subprocess
import json
import os
import time
import numpy as np

# Path to the compiled Rust binary (relative to where you run the Python script from)
RUST_BINARY_PATH = "./rust_face_engine/rust_face_engine"


def get_face_embedding(image_path: str):
    """
    Call the Rust binary on image_path and return a 128-D L2-normalised
    numpy float32 embedding, or None if anything goes wrong.
    """
    if not os.path.exists(RUST_BINARY_PATH):
        raise FileNotFoundError(
            f"Rust binary not found at: {RUST_BINARY_PATH}\n"
            f"Run 'cargo build --release' inside RustCodebase/ first."
        )

    try:
        t0     = time.time()
        result = subprocess.run(
            [RUST_BINARY_PATH, image_path],
            capture_output=True,
            text=True,
            check=True,          # raises CalledProcessError on non-zero exit
        )
        elapsed = time.time() - t0

        raw = result.stdout.strip()
        if not raw:
            print("❌  Rust binary produced no output.")
            return None

        # Rust's Vec<f32> {:?} format is valid JSON: [0.1, -0.2, ...]
        vec = json.loads(raw)
        emb = np.array(vec, dtype=np.float32)

        # Defensive re-normalisation (the binary already normalises, but floating
        # point and any small transport error won't hurt being caught here)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-10:
            emb = emb / norm
        else:
            print("❌  Rust returned a zero-norm embedding.")
            return None

        print(f"⚡  Rust inference: {elapsed:.3f}s  |  embedding dim={emb.shape[0]}")
        return emb

    except subprocess.CalledProcessError as exc:
        print(f"❌  Rust process error:\n{exc.stderr.strip()}")
        return None

    except json.JSONDecodeError as exc:
        print(f"❌  Could not parse Rust output as JSON: {exc}")
        print(f"    Raw output was: {result.stdout[:200]!r}")
        return None

    except Exception as exc:
        print(f"❌  Unexpected error in get_face_embedding: {exc}")
        return None


def compute_similarity(embed1, embed2) -> float:
    """
    Cosine similarity between two embeddings.
    Both are expected to be L2-normalised (which the rest of the code ensures),
    but we normalise defensively so this function is safe to call with raw vectors.

    Returns a float in [-1.0, 1.0].
    """
    a = np.asarray(embed1, dtype=np.float32).ravel()
    b = np.asarray(embed2, dtype=np.float32).ravel()

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na < 1e-10 or nb < 1e-10:
        return -1.0

    return float(np.dot(a / na, b / nb))