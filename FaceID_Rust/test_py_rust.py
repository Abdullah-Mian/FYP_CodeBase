import cv2
import numpy as np
from rust_wrapper import get_face_embedding, compute_similarity

# 1. Create a dummy face image (black square) for testing
dummy_img_path = "temp_test_face.jpg"
dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
cv2.imwrite(dummy_img_path, dummy_img)

print("🧪 Testing Python -> Rust Integration...")

# 2. Get embedding for the first image
vec1 = get_face_embedding(dummy_img_path)

if vec1 is not None:
    print(f"✅ Vector 1 received! Shape: {vec1.shape}")
    print(f"   Sample: {vec1[:5]}...")
    
    # 3. Get it again (to test similarity)
    vec2 = get_face_embedding(dummy_img_path)
    
    # 4. Compare
    score = compute_similarity(vec1, vec2)
    print(f"📊 Similarity Score (Should be 1.0): {score:.4f}")
else:
    print("❌ Failed to get embedding.")

import os
if os.path.exists(dummy_img_path):
    os.remove(dummy_img_path)
