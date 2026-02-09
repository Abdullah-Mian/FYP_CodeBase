# Multi-Face Caching System - Detailed Explanation

## How the Cache Tracks Multiple Faces Independently

### The Core Question

**"How does the cache know which box belongs to which face when there are 2+ faces in view?"**

---

## Overview

The caching system uses **Intersection over Union (IoU)** to track faces across frames by matching bounding box positions. Each face is tracked independently based on its location in the frame.

---

## Step-by-Step Example: Two Faces in View

### Frame 1 - Initial Detection

```
Camera Frame (640×480):
┌────────────────────────────────────────┐
│                                        │
│    ┌──────┐              ┌──────┐     │
│    │Alice │              │ Bob  │     │
│    │(100, │              │(400, │     │
│    │150,  │              │150,  │     │
│    │80,   │              │75,   │     │
│    │100)  │              │95)   │     │
│    └──────┘              └──────┘     │
│                                        │
└────────────────────────────────────────┘
```

**What happens:**

1. **MediaPipe detects 2 faces:**
   ```python
   faces = [
       (100, 150, 80, 100, 0.95),  # x, y, w, h, score
       (400, 150, 75, 95, 0.92)
   ]
   ```

2. **Processing Face 1 (Alice's position):**
   ```python
   bbox = (100, 150, 80, 100)
   
   # Check cache
   cached_name, cached_score, should_verify = get_cached_result(bbox)
   # Returns: (None, None, True) - not in cache yet!
   
   # Verify face
   name, sim = verify_face(face_crop)  # Calls C++ MobileFaceNet
   # Returns: ("Alice", 0.87)
   
   # Store in cache
   face_cache[(100, 150, 80, 100)] = ("Alice", 0.87, time.time(), True)
   ```

3. **Processing Face 2 (Bob's position):**
   ```python
   bbox = (400, 150, 75, 95)
   
   # Check cache
   cached_name, cached_score, should_verify = get_cached_result(bbox)
   # Returns: (None, None, True) - not in cache yet!
   
   # Verify face
   name, sim = verify_face(face_crop)  # Calls C++ MobileFaceNet
   # Returns: ("Bob", 0.82)
   
   # Store in cache
   face_cache[(400, 150, 75, 95)] = ("Bob", 0.82, time.time(), True)
   ```

**Cache state after Frame 1:**
```python
face_cache = {
    (100, 150, 80, 100): ("Alice", 0.87, 1234567890.0, True),
    (400, 150, 75, 95):  ("Bob",   0.82, 1234567890.0, True)
}
```

**Verifications performed:** 2 (one for each new face)

---

### Frame 2 - Faces Moved Slightly (0.1 seconds later)

```
Camera Frame:
┌────────────────────────────────────────┐
│                                        │
│    ┌──────┐              ┌──────┐     │
│    │Alice │              │ Bob  │     │
│    │(102, │              │(398, │     │  ← Moved left 2px
│    │152,  │              │149,  │     │  ← Moved up 1px
│    │80,   │              │75,   │     │
│    │100)  │              │95)   │     │
│    └──────┘              └──────┘     │
│                                        │
└────────────────────────────────────────┘
```

**What happens:**

1. **MediaPipe detects 2 faces (new positions):**
   ```python
   faces = [
       (102, 152, 80, 100, 0.94),  # Alice moved +2, +2
       (398, 149, 75, 95, 0.93)    # Bob moved -2, -1
   ]
   ```

2. **Processing Face at (102, 152, 80, 100):**
   ```python
   bbox_new = (102, 152, 80, 100)
   
   # Check cache - search all cached faces
   for cached_bbox in face_cache:
       iou = calculate_iou(bbox_new, cached_bbox)
       
   # Compare with Alice's cache: (100, 150, 80, 100)
   iou = calculate_iou((102, 152, 80, 100), (100, 150, 80, 100))
   # Returns: 0.89 (HIGH overlap - same face!)
   
   # Compare with Bob's cache: (400, 150, 75, 95)
   iou = calculate_iou((102, 152, 80, 100), (400, 150, 75, 95))
   # Returns: 0.0 (NO overlap - different face)
   
   # Match found with Alice's cache (IoU > 0.5)!
   # Alice is recognized → use cached result
   return ("Alice", 0.87, should_verify=False)
   ```
   **✅ No verification needed!**

3. **Processing Face at (398, 149, 75, 95):**
   ```python
   bbox_new = (398, 149, 75, 95)
   
   # Compare with Alice's cache: (100, 150, 80, 100)
   iou = calculate_iou((398, 149, 75, 95), (100, 150, 80, 100))
   # Returns: 0.0 (NO overlap)
   
   # Compare with Bob's cache: (400, 150, 75, 95)
   iou = calculate_iou((398, 149, 75, 95), (400, 150, 75, 95))
   # Returns: 0.91 (HIGH overlap - same face!)
   
   # Match found with Bob's cache!
   # Bob is recognized → use cached result
   return ("Bob", 0.82, should_verify=False)
   ```
   **✅ No verification needed!**

**Cache state after Frame 2 (updated positions):**
```python
face_cache = {
    (102, 152, 80, 100): ("Alice", 0.87, 1234567890.1, True),  # Updated position
    (398, 149, 75, 95):  ("Bob",   0.82, 1234567890.1, True)   # Updated position
}
# Old entries (100,150...) and (400,150...) automatically deleted
```

**Verifications performed:** 0 ✅

---

### Frame 10 - Third Person Enters (1 second later)

```
Camera Frame:
┌────────────────────────────────────────┐
│                                        │
│  ┌──────┐  ┌──────┐        ┌──────┐   │
│  │Alice │  │ NEW  │        │ Bob  │   │
│  │(105, │  │(250, │        │(395, │   │
│  │155,  │  │160,  │        │148,  │   │
│  │80,   │  │78,   │        │75,   │   │
│  │100)  │  │96)   │        │95)   │   │
│  └──────┘  └──────┘        └──────┘   │
│                                        │
└────────────────────────────────────────┘
```

**What happens:**

1. **MediaPipe detects 3 faces:**
   ```python
   faces = [
       (105, 155, 80, 100, 0.93),  # Alice
       (250, 160, 78, 96, 0.91),   # NEW person
       (395, 148, 75, 95, 0.94)    # Bob
   ]
   ```

2. **Processing Face 1 (Alice):**
   ```python
   # IoU with Alice's cache: 0.87 → Match found!
   return ("Alice", 0.87, should_verify=False)
   ```
   **✅ No verification**

3. **Processing Face 2 (NEW person):**
   ```python
   bbox_new = (250, 160, 78, 96)
   
   # Compare with Alice's cache: IoU = 0.0 (far apart)
   # Compare with Bob's cache: IoU = 0.0 (far apart)
   # No match found!
   
   return (None, None, should_verify=True)
   
   # Verify face
   name, sim = verify_face(face_crop)
   # Returns: ("Unknown", 0.35) - not in database
   
   # Store in cache
   face_cache[(250, 160, 78, 96)] = ("Unknown", 0.35, time.time(), True)
   ```
   **⚠️ Verification performed (new face)**

4. **Processing Face 3 (Bob):**
   ```python
   # IoU with Bob's cache: 0.89 → Match found!
   return ("Bob", 0.82, should_verify=False)
   ```
   **✅ No verification**

**Cache state after Frame 10:**
```python
face_cache = {
    (105, 155, 80, 100): ("Alice",   0.87, 1234567891.0, True),
    (250, 160, 78, 96):  ("Unknown", 0.35, 1234567891.0, True),  # NEW entry
    (395, 148, 75, 95):  ("Bob",     0.82, 1234567891.0, True)
}
```

**Verifications performed:** 1 (only for the new unknown face)

---

### Frame 70 - Unknown Face Re-verification (2 seconds after Frame 10)

```
Camera Frame: (same positions)
┌────────────────────────────────────────┐
│                                        │
│  ┌──────┐  ┌──────┐        ┌──────┐   │
│  │Alice │  │ ???  │        │ Bob  │   │
│  │      │  │(252, │        │      │   │
│  │      │  │162,  │        │      │   │
│  │      │  │78,   │        │      │   │
│  │      │  │96)   │        │      │   │
│  └──────┘  └──────┘        └──────┘   │
│                                        │
└────────────────────────────────────────┘
```

**What happens:**

1. **Processing Alice:** Cached, recognized → No verification ✅

2. **Processing Unknown face:**
   ```python
   bbox_new = (252, 162, 78, 96)
   
   # IoU with cache (250, 160, 78, 96): 0.88 → Match found!
   # Name = "Unknown"
   # Time since last verify: 2.0 seconds (threshold reached!)
   
   return ("Unknown", 0.35, should_verify=True)
   
   # Re-verify (maybe they were added to database)
   name, sim = verify_face(face_crop)
   # Returns: ("Charlie", 0.81) - NOW RECOGNIZED! ✅
   
   # Update cache
   face_cache[(252, 162, 78, 96)] = ("Charlie", 0.81, time.time(), True)
   ```
   **⚠️ Re-verification performed (2-second timer)**

3. **Processing Bob:** Cached, recognized → No verification ✅

**Cache state after Frame 70:**
```python
face_cache = {
    (105, 155, 80, 100): ("Alice",   0.87, timestamp, True),
    (252, 162, 78, 96):  ("Charlie", 0.81, timestamp, True),  # UPDATED!
    (395, 148, 75, 95):  ("Bob",     0.82, timestamp, True)
}
```

---

## The IoU Algorithm in Detail

### What is IoU (Intersection over Union)?

```
Face in Frame 1:           Face in Frame 2:
┌────────────┐             ┌────────────┐
│            │             │            │
│   Face A   │      ──►    │   Face A   │
│  (moved)   │             │  (moved)   │
│            │             │            │
└────────────┘             └────────────┘
  (100,150)                  (102,152)
  80×100                     80×100

Overlapping region:
    ┌──────────┐
    │ Overlap  │
    │  Area    │
    └──────────┘
```

### Calculation:

```python
def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1  # (100, 150, 80, 100)
    x2, y2, w2, h2 = bbox2  # (102, 152, 80, 100)
    
    # Find intersection rectangle
    x_left   = max(100, 102) = 102
    y_top    = max(150, 152) = 152
    x_right  = min(180, 182) = 180
    y_bottom = min(250, 252) = 250
    
    # Intersection area
    intersection = (180-102) × (250-152) = 78 × 98 = 7,644 pixels
    
    # Union area
    area1 = 80 × 100 = 8,000 pixels
    area2 = 80 × 100 = 8,000 pixels
    union = 8,000 + 8,000 - 7,644 = 8,356 pixels
    
    # IoU
    iou = 7,644 / 8,356 = 0.915 (91.5% overlap!)
    
    # Since 0.915 > 0.5 threshold → SAME FACE!
```

---

## Why This Works for Multiple Faces

### Key Insight: Spatial Separation

Faces in the same frame are **spatially separated**. Even if they're close, there's minimal overlap:

```
Two faces side-by-side:
┌──────┐  ┌──────┐
│Alice │  │ Bob  │
│(100, │  │(200, │
│150,  │  │150,  │
│80,   │  │80,   │
│100)  │  │100)  │
└──────┘  └──────┘

IoU calculation:
- Alice's box: x=100 to x=180
- Bob's box:   x=200 to x=280
- NO OVERLAP! IoU = 0.0

Result: Each face matches its own cache entry, not the other's!
```

### Cache Matching Process (Pseudocode)

```python
# For each detected face:
for detected_face in current_frame_faces:
    
    best_match = None
    best_iou = 0
    
    # Compare with ALL cached faces
    for cached_face in face_cache:
        iou = calculate_iou(detected_face.bbox, cached_face.bbox)
        
        if iou > 0.5 and iou > best_iou:
            best_match = cached_face
            best_iou = iou
    
    if best_match:
        # Use cached recognition result
        use_cache(best_match)
    else:
        # New face, verify it
        verify_and_cache(detected_face)
```

---

## Edge Cases Handled

### Case 1: Faces Very Close Together

```
┌──────┐
│Alice │┌──────┐
│      ││ Bob  │
└──────┘│      │
        └──────┘
(100,150) (120,150)

IoU between them: ~0.3 (some overlap, but < 0.5 threshold)
Result: Still treated as separate faces ✅
```

### Case 2: Face Leaves and Returns

```
Frame 1:  Alice present → cached
Frame 5:  Alice leaves  → cache entry ages
Frame 8:  Alice gone    → cache cleaned up (>3 seconds old)
Frame 15: Alice returns → treated as NEW face, re-verified
```

### Case 3: Faces Swap Positions (Rare)

```
Frame 1:  Alice(left)  Bob(right)
Frame 2:  Alice(right) Bob(left)  ← They crossed paths

What happens:
- Face at Alice's old position matches Bob's bbox → IoU low
- Face at Bob's old position matches Alice's bbox → IoU low
- Both get re-verified (acceptable, rare occurrence)
```

---

## Performance Analysis

### With 3 Faces in View:

**Frame 1 (first detection):**
- Verifications: 3 (all new)
- Time: ~50ms × 3 = 150ms

**Frames 2-29 (all recognized, no movement):**
- Verifications: 0
- Time: ~15ms (just detection + drawing)

**Frame 30 (2 seconds later, 1 unknown face):**
- Verifications: 1 (only the unknown face)
- Time: ~65ms (15ms + 50ms for one verification)

**Average over 30 frames:**
- Total verifications: 4
- Without caching would be: 90 (3 faces × 30 frames)
- **Reduction: 95.6%** ✅

---

## Code Location Reference

**Cache data structure (line 167):**
```python
face_cache = {}
```

**IoU calculation (lines 177-206):**
```python
def calculate_iou(bbox1, bbox2):
    # ... intersection calculation
    # ... union calculation
    return intersection_area / union_area
```

**Cache lookup (lines 208-262):**
```python
def get_cached_result(bbox):
    # Search all cached faces
    for cached_bbox in face_cache.items():
        iou = calculate_iou(bbox, cached_bbox)
        if iou > 0.5:
            # Found matching face!
            # Return cached result or flag for re-verification
```

**Main loop processing each face (lines 335-377):**
```python
for (x1, y1, w, h, score) in faces:  # ← Each face independent!
    # Get cached result for THIS face's position
    cached_name, _, should_verify = get_cached_result((x1, y1, w, h))
    
    if should_verify:
        verify_face(...)
        face_cache[(x1, y1, w, h)] = result  # ← Unique cache entry
```

---

## Summary

**How it tracks multiple faces:**
1. Each face has a unique position (bounding box)
2. IoU algorithm matches faces across frames by position overlap
3. Faces far apart have IoU ≈ 0 → separate cache entries
4. Each face tracked independently in the dictionary

**The cache is essentially:**
```
A dictionary mapping positions to identities:
  Position (x,y,w,h) → (Name, Score, Timestamp)
  
  (100,150,80,100) → ("Alice",   0.87, ...)
  (250,160,78,96)  → ("Charlie", 0.81, ...)
  (395,148,75,95)  → ("Bob",     0.82, ...)
```

**This naturally handles any number of faces!**
