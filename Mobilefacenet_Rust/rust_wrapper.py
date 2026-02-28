import subprocess
import json
import os
import numpy as np
import time

# Path to your compiled Rust binary
RUST_BINARY_PATH = "./rust_face_engine/rust_face_engine"

def get_face_embedding(image_path):
    """
    Calls the Rust binary to get the 128-D face embedding.
    """
    if not os.path.exists(RUST_BINARY_PATH):
        raise FileNotFoundError(f"Rust binary not found at: {RUST_BINARY_PATH}")
    
    try:
        # Run the Rust binary
        # capture_output=True grabs the print() output from Rust
        start_time = time.time()
        result = subprocess.run(
            [RUST_BINARY_PATH, image_path],
            capture_output=True,
            text=True,
            check=True
        )
        duration = time.time() - start_time
        
        # Parse the JSON array output from Rust
        output_str = result.stdout.strip()
        embedding = json.loads(output_str)
        
        print(f"⚡ Inference took: {duration:.3f}s")
        return np.array(embedding, dtype=np.float32)

    except subprocess.CalledProcessError as e:
        print(f"❌ Rust Error: {e.stderr}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Failed to parse output: {result.stdout}")
        return None

def compute_similarity(embed1, embed2):
    """
    Calculates Cosine Similarity between two vectors.
    Range: -1.0 to 1.0 (1.0 is identical)
    """
    # Dot product (vectors are already normalized by Rust)
    return np.dot(embed1, embed2)