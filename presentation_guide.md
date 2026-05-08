# 🕳️ Pothole Detection & Severity Analysis System — Evaluator Presentation Guide

---

## 1. Project Introduction & Problem Statement

**Problem:** Potholes cause vehicle damage, accidents, and injuries. Current systems rely entirely on manual road inspection — which is slow, inconsistent, and has no severity measurement.

**Our Solution:** An **AI-powered web application** that:
1. Allows **citizens** to photograph and report potholes with GPS location
2. Uses **two deep learning models** to automatically detect potholes and estimate their severity
3. Provides a **PWD (Public Works Department) dashboard** with AI analysis, severity scoring, and map visualization
4. Has an **AI-powered repair verification** system to close complaints

**Key Innovation:** We combine **instance segmentation** (YOLOv8) with **monocular depth estimation** (Depth Anything V2) to compute a severity score — no physical depth sensors required.

---

## 2. Technology Stack

| Layer | Technology | Why We Chose It |
|-------|-----------|-----------------|
| **Web Framework** | Streamlit | Rapid prototyping, built-in widgets for file upload, camera, maps |
| **Object Detection** | YOLOv8s-seg (Ultralytics) | State-of-the-art; produces both bounding boxes AND pixel-level masks |
| **Depth Estimation** | Depth Anything V2 Small (Hugging Face) | Best monocular depth model; works from a single photo |
| **Deep Learning Backend** | PyTorch | Powers both YOLO and Depth Anything V2 |
| **Detection Parsing** | Supervision (Roboflow) | Clean API for parsing YOLO outputs, mask annotation, tracking |
| **Image Processing** | OpenCV + Pillow | Color conversion, histogram comparison, drawing |
| **Geolocation** | Geopy + Nominatim + PyDeck | GPS → address conversion; 3D interactive maps |
| **Data Storage** | JSON + Filesystem | Lightweight, no database server needed |

---

## 3. System Architecture (High-Level)

```
┌───────────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB APPLICATION                      │
│                                                                   │
│   ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│   │  📸 Public   │  │  🛠️ PWD Official │  │  📊 Complaint    │  │
│   │  Reporter    │  │  Dashboard       │  │  Status Tracker   │  │
│   └──────┬───────┘  └────────┬─────────┘  └───────────────────┘  │
│          │                   │                                     │
│          ▼                   ▼                                     │
│   ┌──────────────────────────────────────────┐                    │
│   │         AI DETECTION PIPELINE            │                    │
│   │  YOLOv8 → Depth Anything V2 → Scoring   │                    │
│   └──────────────────────────────────────────┘                    │
│                                                                   │
│   ┌───────────────────┐  ┌──────────────────────────────────┐    │
│   │  📍 Geolocation & │  │  💾 JSON Database               │    │
│   │  Map Rendering    │  │  (submissions.json + images/)    │    │
│   └───────────────────┘  └──────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
```

---

## 4. The AI Detection Pipeline — Step by Step

This is the **core of the project**. When a PWD official clicks on a report, the following pipeline executes:

### Step 1: Image Preprocessing

```
User uploads/captures image (any format: PNG, JPEG)
              │
              ▼
   PIL opens image → Converts to RGB numpy array
              │
              ▼
   cv2.cvtColor(RGB → BGR)   ← YOLO requires BGR input
```

**Why BGR?** OpenCV and YOLO use BGR color ordering internally. If we pass RGB, the model's accuracy drops significantly.

---

### Step 2: YOLOv8 Instance Segmentation (Pothole Detection)

```python
model = YOLO("best.pt")                              # Load custom-trained model
results = model(bgr_frame, conf=confidence)[0]       # Run inference
detections = sv.Detections.from_ultralytics(results)  # Parse outputs
```

**What YOLO produces for each detected pothole:**
- **Bounding box** `[x1, y1, x2, y2]` — rectangle around the pothole
- **Confidence score** — how certain the model is (0–100%)
- **Segmentation mask** — pixel-level outline of the exact pothole shape
- **Class label** — "pothole"

**How instance segmentation differs from basic detection:**

| Basic Detection | Instance Segmentation (Our Model) |
|----------------|----------------------------------|
| Draws a rectangle | Outlines the exact pothole shape pixel-by-pixel |
| Cannot measure area accurately | Precise area measurement possible |
| May include non-pothole pixels | Only pothole pixels are marked |

The `MaskAnnotator` from the Supervision library renders these masks as colored overlays on the image.

---

### Step 3: Depth Anything V2 (Depth Estimation)

**Goal:** From a single 2D photograph, infer how deep each pixel is — creating a "depth map."

```
Original RGB image → Depth Anything V2 → Dense depth map (same resolution)
```

**Internal steps:**

```python
# 1. Convert to PIL Image
pil_image = Image.fromarray(frame)

# 2. Preprocess with model's tokenizer/processor
inputs = self.image_processor(images=pil_image, return_tensors="pt")

# 3. Forward pass (no gradient needed — inference only)
with torch.no_grad():
    outputs = self.model(**inputs)
    predicted_depth = outputs.predicted_depth   # Raw depth tensor

# 4. Resize depth map to match original image size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=(H, W),           # Match original image dimensions
    mode="bicubic",
    align_corners=False,
).squeeze()

# 5. Normalize to [0, 1] range  
depth_map = (depth_map - min) / (max - min)
```

**Output:** A 2D array where each pixel has a value between 0 (closest to camera) and 1 (farthest from camera).

**Why this matters:** A genuine pothole creates a depression in the road surface. In the depth map, the pothole region will show a different depth profile compared to the flat surrounding road — this difference becomes our severity signal.

---

### Step 4: Severity Score Computation

The severity is a **composite score** from three independent signals:

#### Signal 1: Depth Contrast (Weight: 30%)

**Question answered:** "How much deeper is the pothole compared to the surrounding road?"

```
pothole_mean = mean(depth_map[pothole_region])
road_mean    = mean(depth_map[surrounding_road])    ← 80% expanded bbox, excluding pothole
local_range  = max(surrounding) - min(surrounding)

depth_contrast = |pothole_mean − road_mean| / local_range
```

- The surrounding region is computed by **expanding the bounding box by 80%** in each direction
- The pothole pixels are **excluded** from the surrounding calculation using a boolean mask
- A higher contrast → the pothole is significantly deeper than the road

**In the final score, this is amplified ×5:** `min(1.0, depth_contrast × 5)` — because raw depth differences from monocular estimation are subtle.

---

#### Signal 2: Depth Variance (Weight: 30%)

**Question answered:** "How rough/uneven is the pothole surface internally?"

```
pothole_std = std(depth_map[pothole_region])
road_std    = std(depth_map[road_pixels])

depth_variance = pothole_std / road_std
depth_variance = min(depth_variance, 3.0) / 3.0   ← clamp & normalize to [0,1]
```

- Smooth road → low std. Rough, broken pothole → high std.
- The ratio tells us: **how many times rougher is the pothole compared to normal road?**
- Capped at 3× and normalized to [0, 1]

---

#### Signal 3: Size Factor (Weight: 40%)

**Question answered:** "How large is the pothole relative to the camera view?"

```
pothole_area = (x2 - x1) × (y2 - y1)
image_area   = image_height × image_width
size_ratio   = pothole_area / image_area

size_score = min(1.0, size_ratio × 20)
```

| Pothole covers... | Size Score |
|-------------------|-----------|
| 1% of image | 0.2 |
| 2.5% of image | 0.5 |
| 5% of image | 1.0 (maximum) |

**Why size gets the highest weight (40%):** Larger potholes pose more danger to vehicles and pedestrians regardless of depth.

---

#### Composite Formula

```
severity_score = 0.40 × size_score
               + 0.30 × min(1.0, depth_contrast × 5)
               + 0.30 × depth_variance

severity_score = clamp(severity_score, 0.0, 1.0)
```

#### Severity Classification

| Score Range | Label | Color |
|-------------|-------|-------|
| < 0.30 | 🟢 **Low** | Green |
| 0.30 – 0.55 | 🟠 **Medium** | Orange |
| ≥ 0.55 | 🔴 **High** | Red |

---

### Step 5: Annotation & Visualization

```python
# Draw colored bounding box
cv2.rectangle(annotated, (x1, y1), (x2, y2), severity_color, 2)

# Draw label with background
label = f"{severity} ({score:.0%})"     # e.g., "High (72%)"
cv2.rectangle(annotated, ...)            # Label background
cv2.putText(annotated, label, ...)       # Label text
```

The final output is the original image with:
- **Segmentation masks** (colored overlay on pothole pixels)
- **Bounding boxes** (colored by severity — green/orange/red)
- **Labels** showing severity level and percentage score

---

## 5. Complete Application Workflow

### 5.1 Public Reporter Flow

```
Citizen opens app → "Public Reporter" tab
    │
    ├── Step 1: Clicks "Get Location" → browser GPS captures lat/lon
    │
    ├── Step 2: Takes photo (camera) OR uploads image file
    │
    └── Step 3: Clicks "Submit Report"
              │
              ├── Image saved to submissions/ directory
              ├── Metadata saved to submissions.json
              │     {id, filepath, latitude, longitude, timestamp, status: "Pending Analysis"}
              └── Success message + balloons animation
```

### 5.2 PWD Official Flow

```
Official opens app → "PWD Official" tab → Logs in (admin/pwd123)
    │
    ├── Sees tabular list of ALL reports (ID, time, location, GPS, status)
    │
    ├── Clicks a Report ID → drills into that report
    │     │
    │     ├── Shows original image (left) + AI-analyzed image (right)
    │     │     └── AI Pipeline runs: YOLO → Depth → Severity → Annotation
    │     │
    │     ├── Shows per-pothole stats:
    │     │     • Severity badge (Low/Medium/High with color)
    │     │     • Score percentage
    │     │     • Bounding box dimensions in pixels
    │     │     • Size as % of image
    │     │     • Depth contrast value
    │     │
    │     ├── Shows interactive 3D map (PyDeck)
    │     │     • Marker color = overall severity
    │     │     • Zoomed to street level (zoom=17, pitch=45°)
    │     │
    │     └── Status management dropdown:
    │           Pending Analysis → In Progress → Resolved
    │                                              │
    │                                    ┌─────────┘
    │                                    ▼
    │                        REPAIR VERIFICATION (3-step)
    │
    └── Can logout
```

### 5.3 Repair Verification Flow (3-Step)

When a PWD official tries to mark a complaint as **"Resolved"**, these checks run:

```
Step 1: LOCATION CHECK (Haversine Formula)
    ├── Official's current GPS vs. original report GPS
    ├── Must be within 500 meters
    ├── Formula: d = 2R × atan2(√a, √(1-a))
    │     where a = sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2)
    │     R = 6,371,000 meters (Earth's radius)
    └── FAIL → "You are X meters away from the pothole location"

Step 2: SCENE SIMILARITY (HSV Histogram Comparison)
    ├── Both images resized to 256×256
    ├── Converted to HSV color space
    ├── 2D histograms computed (Hue: 50 bins, Saturation: 60 bins)
    ├── Compared using cv2.HISTCMP_CORREL  
    ├── Must have ≥ 30% correlation
    └── FAIL → "The scenes look completely different"

Step 3: AI POTHOLE CHECK (YOLO on repair image)
    ├── The repaired road image is run through YOLO
    ├── If ANY potholes detected → FAIL
    ├── If ZERO potholes → PASS ✅
    └── Status updated to "Resolved"
        Repair image saved as repair_{id}.jpg
```

### 5.4 Complaint Status Tracker

Any user (no login required) can view:
- **Summary metrics** — Total, Pending, In Progress, Resolved counts
- **Filterable table** of all complaints
- **Before/After image comparison** for resolved complaints (original pothole vs. repaired road)

---

## 6. YOLOv8 Model — Training Details

### How We Trained

| Parameter | Value | Why |
|-----------|-------|-----|
| Architecture | YOLOv8s-seg | Small variant — good accuracy vs. speed tradeoff |
| Task | Instance Segmentation | Need pixel-level masks, not just boxes |
| Pretrained | Yes (COCO) | Transfer learning — starts with general object knowledge |
| Epochs | 200 (with early stopping at patience=20) | Prevent overfitting |
| Batch Size | 64 | Maximizes GPU utilization |
| Image Size | 640×640 | Standard YOLO input resolution |
| Optimizer | AdamW (auto-selected) | Modern optimizer with weight decay |
| Initial LR | 0.01 → Final LR: 0.0001 | Learning rate decay over training |

### Data Augmentation

The model uses aggressive augmentation to prevent overfitting on a small dataset:

| Technique | Effect |
|-----------|--------|
| **Mosaic** (100%) | Combines 4 training images into 1 — forces model to learn varied contexts |
| **HSV shifts** | Hue ±1.5%, Saturation ±70%, Brightness ±40% — handles different lighting |
| **Horizontal Flip** (50%) | Potholes can appear on either side of the road |
| **Random Scale** (±50%) | Handles potholes at different distances from camera |
| **Random Erasing** (40%) | Occlusion robustness — potholes may be partially hidden |
| **RandAugment** | Automated augmentation policy |

### Loss Function

The model optimizes three losses simultaneously:

```
Total Loss = 7.5 × Box Loss + 0.5 × Classification Loss + 1.5 × DFL Loss
```

| Loss | Weight | What It Optimizes |
|------|--------|-------------------|
| **Box Loss** | 7.5 | Bounding box accuracy (CIoU) |
| **Classification Loss** | 0.5 | Pothole vs. background classification |
| **DFL (Distribution Focal Loss)** | 1.5 | Fine-grained box boundary refinement |

### Training Outputs

The `runs/` directory contains curves showing model performance:
- **Precision-Recall curves** — for both bounding boxes and segmentation masks
- **F1 curves** — optimal confidence threshold identification
- **Confusion matrix** — false positive/negative analysis
- **Training loss curves** — convergence verification

---

## 7. Depth Anything V2 — How It Works

### Architecture

Depth Anything V2 is a **Vision Transformer (ViT)** based encoder-decoder model:

```
Input Image (RGB)
      │
      ▼
┌─────────────┐
│  ViT Encoder │ — Splits image into patches, processes with self-attention
│  (DINOv2)    │ — Pre-trained on millions of images for robust features
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  DPT Decoder │ — Dense Prediction Transformer
│  (Head)      │ — Reconstructs per-pixel depth from ViT features
└──────┬──────┘
       │
       ▼
  Dense Depth Map (H × W)  — One depth value per pixel
```

### Why "Small" Variant?

| Variant | Parameters | Speed | Accuracy |
|---------|-----------|-------|----------|
| Small | 24.8M | ⚡ Fast | Good |
| Base | 97.5M | Medium | Better |
| Large | 335.3M | Slow | Best |

We chose **Small** for real-time usability on consumer hardware (especially CPU-only machines).

### Key Insight for Evaluators

> **Monocular depth estimation** means inferring 3D depth from a **single 2D image** — no stereo cameras, no LiDAR, no depth sensors. The model has learned depth cues from millions of images: perspective, object size, texture gradients, occlusion, etc.

---

## 8. Data Flow Summary

```
┌─────────────────────────────────────────────────┐
│  USER                                            │
│  Uploads photo + GPS                             │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  app.py                                          │
│  save_submission() → submissions.json + images   │
└──────┬──────────────────────────────────────────┘
       │ (When PWD official views report)
       ▼
┌─────────────────────────────────────────────────┐
│  callback()                                      │
│  RGB→BGR → YOLO → Detections → Mask Annotation   │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  depth_estimator.analyze_frame()                 │
│  ├── estimate_depth() → depth map [0,1]          │
│  └── For each bbox:                              │
│       ├── compute_severity_score()               │
│       │    • depth_contrast (30%)                 │
│       │    • depth_variance (30%)                 │
│       │    • size_score (40%)                     │
│       └── classify_severity() → Low/Med/High     │
└──────┬──────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  VISUALIZATION                                   │
│  ├── Annotated image with masks + labels + boxes │
│  ├── Severity badges (color-coded)               │
│  ├── 3D PyDeck map with marker                   │
│  └── Per-pothole statistics                      │
└─────────────────────────────────────────────────┘
```

---

## 9. Key Design Decisions (For Q&A)

| Question | Answer |
|----------|--------|
| **Why YOLO over Faster R-CNN or SSD?** | YOLO is single-pass (faster), and the `-seg` variant gives us pixel-level masks, not just bounding boxes |
| **Why Depth Anything V2 over MiDaS?** | V2 is newer, more accurate, and has a better Hugging Face integration. We previously used MiDaS and upgraded |
| **Why three severity signals?** | No single signal is reliable alone — depth estimation from a single photo is noisy, so we combine size (reliable) with depth signals (complementary) |
| **Why JSON over a database?** | Simplicity for a prototype. No database setup required. Can easily migrate to SQLite or PostgreSQL later |
| **Why ByteTrack is disabled for images?** | ByteTrack is an object tracker designed for video (maintains IDs across frames). On single independent images, it discards all detections because there's no "tracking history" |
| **Why 500m for repair verification?** | GPS accuracy on mobile phones is typically 5–50m. 500m gives enough tolerance while ensuring the official is near the actual location |
| **Why HSV histogram comparison?** | HSV separates color (hue) from brightness (value), making the comparison robust to lighting changes between the original and repair photo |

---

## 10. Project File Structure

```
Pothole-Detection/
├── app.py                 ← Main Streamlit app (771 lines)
│                            All UI views, routing, detection pipeline
├── depth_estimator.py     ← Depth Anything V2 module (204 lines)
│                            Depth map generation + severity scoring
├── best.pt                ← Custom-trained YOLOv8s-seg weights (~23 MB)
├── requirements.txt       ← Python dependencies
├── submissions.json       ← JSON database of pothole reports
├── submissions/           ← Uploaded pothole images + repair images
├── runs/                  ← YOLO training artifacts (curves, metrics)
│   ├── args.yaml          ← Full training configuration
│   ├── results.csv        ← Epoch-by-epoch metrics
│   ├── results.png        ← Training loss & mAP curves
│   ├── confusion_matrix.png
│   ├── *P_curve.png       ← Precision curves
│   ├── *R_curve.png       ← Recall curves
│   ├── *F1_curve.png      ← F1 curves
│   └── *PR_curve.png      ← Precision-Recall curves
└── venv/                  ← Python virtual environment
```

---

## 11. Presentation Tips

1. **Start with the problem** — show a real pothole image, explain the danger
2. **Demo the citizen flow** — upload a pothole photo, show it appearing in the system
3. **Demo the PWD dashboard** — show the AI analysis side-by-side with the original
4. **Explain the severity score** by walking through one pothole's three signals
5. **Show the depth map** — it's visually impressive and demonstrates the AI at work
6. **Demo repair verification** — show how the system prevents false resolutions
7. **Show training curves** from the `runs/` folder — proves the model was properly trained
8. **Be ready for:** "What if the depth estimation is inaccurate?" → "That's why we use three complementary signals, not just depth alone"
