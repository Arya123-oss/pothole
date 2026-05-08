# 🕳️ Pothole Detection & Severity Analysis System

## Complete Project Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack & Libraries](#2-technology-stack--libraries)
3. [Project File Structure](#3-project-file-structure)
4. [System Architecture](#4-system-architecture)
5. [YOLO Model — Object Detection](#5-yolo-model--object-detection)
6. [Depth Estimation — Depth Anything V2](#6-depth-estimation--depth-anything-v2)
7. [Severity Scoring — Equations & Calculations](#7-severity-scoring--equations--calculations)
8. [Application Workflow](#8-application-workflow)
9. [User Roles & Views](#9-user-roles--views)
10. [Repair Verification System](#10-repair-verification-system)
11. [Geolocation & Mapping](#11-geolocation--mapping)
12. [Data Storage & Submissions](#12-data-storage--submissions)
13. [Video Processing Pipeline](#13-video-processing-pipeline)
14. [Training Configuration](#14-training-configuration)
15. [Utility & Debug Scripts](#15-utility--debug-scripts)
16. [How to Run](#16-how-to-run)
17. [License](#17-license)

---

## 1. Project Overview

This project is an **AI-powered Pothole Detection and Severity Analysis System** built as a web application using **Streamlit**. It enables:

- **Citizens (Public Reporters)** to photograph and report potholes from their mobile devices with GPS location.
- **PWD (Public Works Department) Officials** to log in, review AI-analyzed pothole reports, see severity scores, and manage complaint status.
- **Complaint Tracking** for anyone to view the status of all filed complaints.

The system uses a **custom-trained YOLOv8 instance segmentation model** to detect potholes in images/video, and a **Depth Anything V2 monocular depth estimation model** to infer the depth/severity of detected potholes without any physical sensors.

---

## 2. Technology Stack & Libraries

### Core Dependencies (`requirements.txt`)

| Library | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | 8.2.92 | YOLOv8 model loading, inference, and training |
| `opencv-python` | 4.10.0.84 | Image/video reading, color conversion, drawing, histogram comparison |
| `numpy` | 1.25.2 | Array operations, numerical computation |
| `supervision` | 0.23.0 | Detection parsing (`Detections.from_ultralytics`), `ByteTrack` tracker, `MaskAnnotator` |
| `streamlit` | 1.38.0 | Web application framework (UI, widgets, state management) |
| `transformers` | latest | Hugging Face — loads Depth Anything V2 model + image processor |

### Additional Libraries (imported in `app.py`)

| Library | Purpose |
|---------|---------|
| `torch` (PyTorch) | Deep learning backbone for both YOLO and Depth Anything V2 |
| `PIL` (Pillow) | Image opening, format conversion, saving |
| `pydeck` | Interactive 3D map rendering in Streamlit (ScatterplotLayer) |
| `geopy` | Reverse geocoding — converts GPS coordinates to place names via Nominatim |
| `pandas` | DataFrame for complaint status table rendering |
| `streamlit_geolocation` | Browser-based GPS location capture widget |
| `streamlit.components.v1` | Injecting custom JavaScript (auto-scroll to selected report) |
| `json` | Complaint submission persistence (read/write JSON) |
| `math` | Haversine distance formula for repair location verification |
| `datetime` | Timestamps for submissions |

### AI / ML Models Used

| Model | Type | Source |
|-------|------|--------|
| **YOLOv8s-seg** | Instance Segmentation | Ultralytics — custom-trained on pothole dataset, saved as `best.pt` |
| **Depth Anything V2 Small** | Monocular Depth Estimation | Hugging Face (`depth-anything/Depth-Anything-V2-Small-hf`) |

---

## 3. Project File Structure

```
Pothole-Detection/
├── app.py                      # Main Streamlit application (775 lines)
├── depth_estimator.py          # Depth Anything V2 depth & severity module (204 lines)
├── best.pt                     # Custom-trained YOLOv8s-seg model weights (~23 MB)
├── requirements.txt            # Python dependencies
├── submissions.json            # JSON database of all pothole reports
├── submissions/                # Directory storing uploaded pothole images
├── runs/                       # YOLOv8 training output (metrics, curves, visualizations)
│   ├── args.yaml               # Full training hyperparameter configuration
│   ├── results.csv             # Epoch-by-epoch training metrics
│   ├── results.png             # Training loss & mAP curves
│   ├── confusion_matrix.png    # Class confusion matrix
│   ├── BoxP_curve.png          # Box Precision curve
│   ├── BoxR_curve.png          # Box Recall curve
│   ├── BoxF1_curve.png         # Box F1 curve
│   ├── BoxPR_curve.png         # Box Precision-Recall curve
│   ├── MaskP_curve.png         # Mask Precision curve
│   ├── MaskR_curve.png         # Mask Recall curve
│   ├── MaskF1_curve.png        # Mask F1 curve
│   ├── MaskPR_curve.png        # Mask Precision-Recall curve
│   ├── labels.jpg              # Label distribution visualization
│   ├── labels_correlogram.jpg  # Label correlogram
│   ├── train_batch*.jpg        # Sample training batches
│   └── val_batch*_*.jpg        # Validation predictions vs ground truth
├── test.py                     # Standalone video detection test script
├── convert.py                  # Export model to ONNX/TorchScript/NCNN formats
├── debug_shapes.py             # Debug — test YOLO on different image shapes
├── debug_tracking.py           # Debug — ByteTrack tracker behavior on single images
├── debug_yolo_compare.py       # Debug — compare detection across different images
├── debug_yolo_dash.py          # Debug — dashboard detection pipeline test
├── debug_yolo_rgb_bgr.py       # Debug — RGB vs BGR color space detection test
├── debug_yolo_upload.py        # Debug — simulate uploaded file detection
├── verify_fix.py               # Verify callback fix with mocked Streamlit
├── actual_verify.py            # Verify callback with real DepthEstimator
├── test_color.py               # Test color format handling
├── test_df_select.py           # Test DataFrame selectbox behavior
├── test_links.py               # Test link/button rendering
├── test_links2.py              # Test link rendering (variant)
├── test_select.py              # Test selectbox UI behavior
├── uploaded_image.jpg          # Sample test image
├── uploaded_image.png          # Sample test image (PNG)
├── LICENSE                     # MIT License
├── README.md                   # Project readme
├── .gitignore                  # Git ignore rules
└── venv/                       # Python virtual environment
```

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT WEB APP (app.py)                  │
│                                                                     │
│   ┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐  │
│   │  Public       │   │  PWD Official    │   │  Complaint Status │  │
│   │  Reporter     │   │  Dashboard       │   │  Tracker          │  │
│   │  View         │   │  View            │   │  View             │  │
│   └──────┬───────┘   └────────┬─────────┘   └───────────────────┘  │
│          │                    │                                      │
│          ▼                    ▼                                      │
│   ┌──────────────────────────────────────────┐                      │
│   │          DETECTION PIPELINE              │                      │
│   │                                          │                      │
│   │  1. Image Input (RGB)                    │                      │
│   │  2. RGB → BGR conversion                 │                      │
│   │  3. YOLOv8s-seg inference                │                      │
│   │  4. Detections parsed via Supervision    │                      │
│   │  5. Mask annotation on frame             │                      │
│   │  6. Depth Anything V2 depth map          │                      │
│   │  7. Severity score computation           │                      │
│   │  8. Label + bounding box overlay         │                      │
│   └──────────────────────────────────────────┘                      │
│                                                                     │
│   ┌───────────────────┐   ┌──────────────────────────────────────┐  │
│   │  GEOLOCATION &    │   │  SUBMISSIONS DATABASE               │  │
│   │  REVERSE GEOCODING│   │  (submissions.json + submissions/)  │  │
│   │  (geopy/pydeck)   │   │                                      │  │
│   └───────────────────┘   └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Internal Pipeline Flow (per image)

```
User uploads image
       │
       ▼
  Image loaded as RGB numpy array
       │
       ▼
  cv2.cvtColor(RGB → BGR)  ← YOLO expects BGR input
       │
       ▼
  model(bgr_frame, conf=confidence)  ← YOLOv8 inference
       │
       ▼
  sv.Detections.from_ultralytics(results)  ← parse bounding boxes + masks
       │
       ▼
  mask_annotator.annotate(frame, detections)  ← draw segmentation masks
       │
       ▼
  depth_estimator.analyze_frame(frame, detections)
       │
       ├──→  estimate_depth(frame) → Depth Anything V2 → normalized depth map [0,1]
       │
       └──→  For each detected pothole bounding box:
                 │
                 ├── compute_severity_score(depth_map, bbox, image_shape)
                 │       │
                 │       ├── Signal 1: Depth Contrast
                 │       ├── Signal 2: Depth Variance
                 │       ├── Signal 3: Size Factor
                 │       └── Composite Score (weighted combination)
                 │
                 └── classify_severity(score) → "Low" / "Medium" / "High"
       │
       ▼
  Draw severity labels & colored bounding boxes on annotated frame
       │
       ▼
  Return (annotated_frame, pothole_info_list) to UI
```

---

## 5. YOLO Model — Object Detection

### Model Architecture

- **Model**: YOLOv8s-seg (YOLOv8 Small — Segmentation variant)
- **Task**: Instance Segmentation (detects bounding boxes **AND** pixel-level masks)
- **Config file**: `yolov8s-seg.yaml` (Ultralytics built-in architecture)
- **Trained weights**: `best.pt` (~23.8 MB)

### What YOLO Does

1. Takes a BGR image as input
2. Outputs bounding boxes (`[x1, y1, x2, y2]`), confidence scores, class labels, and segmentation masks for each detected pothole
3. The `supervision` library parses these outputs into a `Detections` object

### How YOLO is Used in Code

```python
# Load model (cached globally once)
model = YOLO("best.pt")

# Inference
bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
results = model(bgr_frame, conf=confidence)[0]  # conf = confidence threshold

# Parse detections
detections = sv.Detections.from_ultralytics(results)

# Annotate with masks
annotated = mask_annotator.annotate(frame.copy(), detections=detections)
```

### Segmentation Masks

Unlike basic object detection (bounding box only), instance segmentation produces pixel-level masks that precisely outline the pothole shape. The `MaskAnnotator` from `supervision` renders these masks as colored overlays on the image.

---

## 6. Depth Estimation — Depth Anything V2

### What It Is

**Depth Anything V2** is a state-of-the-art **monocular depth estimation** model from Hugging Face. It infers a depth map from a single 2D image — predicting how far away each pixel is from the camera.

### Model Details

| Property | Value |
|----------|-------|
| Model Name | `depth-anything/Depth-Anything-V2-Small-hf` |
| Source | Hugging Face Transformers |
| Input | Single RGB image |
| Output | Dense depth map (same resolution as input) |
| Normalization | Min-max normalized to [0, 1] |
| Device | CUDA (GPU) if available, else CPU |

### How It Works Internally

```python
# 1. Convert numpy array to PIL Image
pil_image = Image.fromarray(frame)

# 2. Preprocess with the model's image processor
inputs = self.image_processor(images=pil_image, return_tensors="pt")

# 3. Run inference (no gradient computation)
with torch.no_grad():
    outputs = self.model(**inputs)
    predicted_depth = outputs.predicted_depth

# 4. Resize depth map to match original image dimensions
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=frame.shape[:2],  # (H, W)
    mode="bicubic",
    align_corners=False,
).squeeze()

# 5. Normalize to [0, 1]
depth_map = (depth_map - depth_min) / (depth_max - depth_min)
```

### Why Depth Estimation is Used

Potholes are physical depressions in the road surface. A depth map reveals depth variation: a genuine pothole will have a different depth profile compared to the surrounding flat road. This signal is combined with size information to produce a meaningful severity score — without any physical depth sensor.

---

## 7. Severity Scoring — Equations & Calculations

The severity of each detected pothole is computed as a **composite score** from three independent signals. This is implemented in `depth_estimator.py` → `compute_severity_score()`.

### Signal 1: Depth Contrast

**What it measures**: How much the pothole's depth differs from the surrounding road surface.

**Calculation**:
```
pothole_mean = mean(depth_map[pothole_region])
road_mean    = mean(depth_map[surrounding_region - pothole_region])
local_range  = max(surrounding_depth) - min(surrounding_depth)

depth_contrast = |pothole_mean - road_mean| / local_range
```

- The surrounding region is computed by expanding the bounding box by **80%** in each direction
- A mask excludes the pothole itself from the surrounding calculation
- Higher depth contrast → deeper pothole relative to road

### Signal 2: Depth Variance

**What it measures**: How rough/uneven the pothole surface is internally.

**Calculation**:
```
pothole_std = std(depth_map[pothole_region])
road_std    = std(depth_map[surrounding_road_pixels])

depth_variance = pothole_std / road_std
depth_variance = min(depth_variance, 3.0) / 3.0   # clamp & normalize to [0, 1]
```

- Rougher, deeper potholes have more internal depth variation compared to smooth road surfaces
- The ratio is capped at 3.0 and normalized to [0, 1]

### Signal 3: Size Factor

**What it measures**: How large the pothole is relative to the overall image.

**Calculation**:
```
pothole_area = (x2 - x1) × (y2 - y1)
image_area   = image_height × image_width
size_ratio   = pothole_area / image_area

size_score = min(1.0, size_ratio × 20)
```

- Scale: a pothole covering **1% of the image → 0.2**, **5% of the image → 1.0**
- Larger potholes are considered more severe

### Composite Severity Score

The three signals are combined using a **weighted average**:

```
composite = 0.40 × size_score
          + 0.30 × min(1.0, depth_contrast × 5)
          + 0.30 × depth_variance

composite = clamp(composite, 0.0, 1.0)
```

| Weight | Signal | Reasoning |
|--------|--------|-----------|
| **40%** | Size Score | Larger potholes pose more danger to vehicles and pedestrians |
| **30%** | Depth Contrast (amplified ×5) | Deeper potholes are more severe; depth signal is subtle so it's amplified |
| **30%** | Depth Variance | Surface roughness indicates structural degradation |

### Severity Classification Thresholds

| Score Range | Severity Label | Color (BGR) | Color (Display) |
|-------------|---------------|-------------|-----------------|
| `< 0.30` | 🟢 **Low** | `(0, 200, 0)` | Green |
| `0.30 – 0.55` | 🟠 **Medium** | `(255, 140, 0)` | Orange |
| `≥ 0.55` | 🔴 **High** | `(220, 0, 0)` | Red |

---

## 8. Application Workflow

### Complete End-to-End Flow

```
                    ┌─────────────────────────────┐
                    │   User opens Streamlit App   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Sidebar Navigation         │
                    │   • Public Reporter           │
                    │   • PWD Official              │
                    │   • Complaint Status           │
                    └──────────────┬──────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────┐        ┌──────────────────┐      ┌────────────────┐
│  PUBLIC       │        │  PWD OFFICIAL    │      │  COMPLAINT     │
│  REPORTER     │        │  LOGIN/DASHBOARD │      │  STATUS        │
│               │        │                  │      │  TRACKER       │
│ 1. Get GPS    │        │ 1. Login         │      │                │
│ 2. Take/Upload│        │ 2. View reports  │      │ • View all     │
│    photo      │        │ 3. AI Analysis   │      │   complaints   │
│ 3. Submit     │        │ 4. Update status │      │ • Filter by    │
│    report     │        │ 5. Repair verify │      │   status       │
└──────┬───────┘        └────────┬─────────┘      │ • Before/After │
       │                         │                 │   images       │
       ▼                         ▼                 └────────────────┘
┌──────────────┐        ┌──────────────────┐
│ Save to       │        │ For each report: │
│ submissions/  │        │ • Load image     │
│ + JSON entry  │        │ • Run YOLO       │
└──────────────┘        │ • Run Depth Est. │
                        │ • Compute score  │
                        │ • Show on map    │
                        └──────────────────┘
```

---

## 9. User Roles & Views

### 9.1 Public Reporter View (`public_reporter_view()`)

**Purpose**: Allows citizens to report potholes.

**Steps**:
1. **Get Location** — Uses `streamlit_geolocation` to capture browser GPS (latitude/longitude)
2. **Upload/Capture Image** — Camera capture or file upload (PNG/JPEG/JPG)
3. **Submit Report** — Saves image to `submissions/` directory and metadata to `submissions.json`

**Validations**:
- Image is required
- Low-resolution warning if image < 300×300 pixels
- RGBA/P images are auto-converted to RGB
- Location is optional but encouraged

### 9.2 PWD Official Dashboard (`official_dashboard_view()`)

**Purpose**: Authenticated dashboard for officials to review and manage reports.

**Login**: Username: `admin`, Password: `pwd123`

**Features**:
- **Tabular listing** of all reports with ID, timestamp, location, GPS, and status
- **Clickable report IDs** to drill into individual reports
- **AI Analysis** panel showing:
  - Original image + AI-annotated image side by side
  - Per-pothole severity score, size, depth contrast
  - Color-coded severity badges (Green/Orange/Red)
  - Interactive 3D map with severity-colored markers
- **Status management**: Pending Analysis → In Progress → Resolved
- **Repair Verification** workflow (see Section 10)

### 9.3 Complaint Status Tracker (`complaint_status_view()`)

**Purpose**: Public-facing status overview.

**Features**:
- Summary metrics (Total, Pending, In Progress, Resolved)
- Filterable DataFrame table
- Before/After image comparison for resolved complaints

---

## 10. Repair Verification System

When a PWD official attempts to mark a report as **"Resolved"**, a multi-step verification is triggered:

### Step 1: Location Verification (Haversine Distance)

The official must be at the same physical location as the original report.

**Haversine Formula** (distance between two GPS coordinates on Earth's surface):

```
R = 6,371,000 meters (Earth's radius)

φ₁ = radians(lat₁)
φ₂ = radians(lat₂)
Δφ = radians(lat₂ - lat₁)
Δλ = radians(lon₂ - lon₁)

a = sin²(Δφ/2) + cos(φ₁) × cos(φ₂) × sin²(Δλ/2)
d = 2 × R × atan2(√a, √(1-a))
```

- **Threshold**: Must be within **500 meters** of the original report location
- If original report has no GPS: location check is skipped

### Step 2: Image Scene Similarity (HSV Histogram Comparison)

Compares the original and repair images to verify they are from the same scene.

**Method**:
```
1. Resize both images to 256×256
2. Convert BGR → HSV color space
3. Compute 2D histograms (Hue: 50 bins, Saturation: 60 bins)
4. Normalize histograms
5. Compare using cv2.compareHist with HISTCMP_CORREL (Correlation method)
```

- **Threshold**: Similarity must be ≥ **0.30** (30% correlation)
- If below threshold: verification fails ("scenes look completely different")

### Step 3: AI Pothole Detection on Repair Image

The repaired road image is run through the YOLO model:

```python
repair_results = model(repair_bgr, conf=confidence)[0]
num_potholes = len(repair_results.boxes)
```

- If **any potholes detected**: verification fails
- If **zero potholes detected**: verification passes → status updated to "Resolved"
- Repair image saved as `repair_{report_id}.jpg`

---

## 11. Geolocation & Mapping

### Reverse Geocoding

Converts GPS coordinates to human-readable place names.

```python
geolocator = Nominatim(user_agent="pothole_detector_app")
reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

location = reverse_geocode((lat, lon), language='en')
place_name = location.address
```

- **Provider**: OpenStreetMap Nominatim (free, no API key needed)
- **Rate limiting**: 1 request per second to avoid throttling
- Results are cached with `@st.cache_data`

### Map Visualization

Uses **PyDeck** to render an interactive 3D map:

```python
st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=17, pitch=45),
    layers=[
        pdk.Layer('ScatterplotLayer',
            data=[{"lat": lat, "lon": lon, "color": marker_color}],
            get_position='[lon, lat]',
            get_fill_color='color',
            get_radius=50,
        )
    ]
))
```

- **Marker color** = overall severity color:
  - Green `[40, 167, 69]` for Low
  - Orange `[253, 126, 20]` for Medium
  - Red `[220, 53, 69]` for High

---

## 12. Data Storage & Submissions

### Storage Architecture

- **No database server** — uses flat-file JSON storage
- **Images**: Saved to `submissions/` directory as `{timestamp}_{filename}`
- **Metadata**: Appended to `submissions.json`

### Submission JSON Schema

```json
{
    "id": "20260228_190353",          // Unique ID (timestamp-based)
    "filepath": "submissions/...",     // Relative path to saved image
    "latitude": 8.526734,             // GPS latitude (nullable)
    "longitude": 76.892944,           // GPS longitude (nullable)
    "timestamp": "2026-02-28 19:03:53", // Human-readable timestamp
    "status": "Pending Analysis"      // Status: Pending Analysis | In Progress | Resolved
}
```

### Status Flow

```
Pending Analysis  ──→  In Progress  ──→  Resolved
                                          (requires verification)
```

---

## 13. Video Processing Pipeline

The system supports real-time video analysis via `video_input()`:

1. **Input**: Upload video (MP4/MOV/AVI/MKV/WebM) or use sample video
2. **Frame loop**: Read frames with OpenCV `VideoCapture`
3. **Per frame**:
   - Resize to specified dimensions
   - Convert BGR → RGB
   - Run through `callback()` (YOLO + Depth + Severity)
   - Display annotated frame + pothole info
4. **FPS tracking**: Computed as `1 / (current_time - previous_time)`
5. **Display**: Shows height, width, FPS metrics in columns

**Note**: ByteTrack tracking is initialized but deliberately skipped for static image analysis (dashboard mode), since it discards detections that don't persist across consecutive frames.

---

## 14. Training Configuration

The YOLO model was trained with the following configuration (from `runs/args.yaml`):

### Key Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `task` | `segment` | Instance segmentation (not just detection) |
| `model` | `yolov8s-seg.yaml` | YOLOv8 Small segmentation architecture |
| `data` | `data.yaml` | Dataset configuration (pothole images + annotations) |
| `epochs` | 200 | Maximum training epochs |
| `patience` | 20 | Early stopping patience |
| `batch` | 64 | Batch size |
| `imgsz` | 640 | Input image size (640×640 pixels) |
| `pretrained` | true | Transfer learning from COCO pretrained weights |

### Optimizer & Learning Rate

| Parameter | Value |
|-----------|-------|
| `optimizer` | auto (AdamW selected automatically) |
| `lr0` | 0.01 (initial learning rate) |
| `lrf` | 0.01 (final learning rate factor) |
| `momentum` | 0.937 |
| `weight_decay` | 0.0005 |
| `warmup_epochs` | 3.0 |
| `warmup_momentum` | 0.8 |
| `warmup_bias_lr` | 0.1 |

### Loss Function Weights

| Loss Component | Weight | Description |
|---------------|--------|-------------|
| `box` | 7.5 | Bounding box regression loss |
| `cls` | 0.5 | Classification loss |
| `dfl` | 1.5 | Distribution Focal Loss (box refinement) |

### Data Augmentation

| Augmentation | Value | Description |
|-------------|-------|-------------|
| `hsv_h` | 0.015 | Hue shift |
| `hsv_s` | 0.7 | Saturation shift |
| `hsv_v` | 0.4 | Value (brightness) shift |
| `translate` | 0.1 | Random translation |
| `scale` | 0.5 | Random scaling |
| `fliplr` | 0.5 | Horizontal flip probability |
| `mosaic` | 1.0 | Mosaic augmentation (combines 4 images) |
| `erasing` | 0.4 | Random erasing |
| `auto_augment` | `randaugment` | Random augmentation policy |

### Other Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| `amp` | true | Automatic Mixed Precision training |
| `iou` | 0.7 | IoU threshold for NMS |
| `max_det` | 300 | Maximum detections per image |
| `overlap_mask` | true | Allow overlapping masks |
| `mask_ratio` | 4 | Mask downsampling ratio |
| `close_mosaic` | 10 | Disable mosaic for last 10 epochs |
| `cos_lr` | false | Cosine learning rate scheduler (disabled) |
| `deterministic` | true | Reproducible training |

### Training Output

The `runs/` directory contains training artifacts:
- **Precision/Recall/F1 curves** for both bounding boxes and masks
- **Confusion matrix** (regular and normalized)
- **Label distribution** and correlogram
- **Training batch visualizations** (early and late epochs)
- **Validation batch** predictions vs ground truth
- **Full training metrics** in `results.csv` (loss values per epoch)

---

## 15. Utility & Debug Scripts

### `convert.py` — Model Export

Exports the trained YOLO model to multiple deployment formats:
- **ONNX** — cross-platform inference
- **TorchScript** — PyTorch JIT
- **NCNN** — mobile/edge deployment

### `test.py` — Standalone Video Test

Runs pothole detection on a video file using OpenCV window display (non-Streamlit), with ByteTrack tracking enabled.

### Debug Scripts

| Script | Purpose |
|--------|---------|
| `debug_shapes.py` | Tests YOLO detection on images of different resolutions |
| `debug_tracking.py` | Verifies ByteTrack behavior — shows that trackers discard detections on single images |
| `debug_yolo_compare.py` | Compares detection results across sample and uploaded images |
| `debug_yolo_dash.py` | Simulates the dashboard detection pipeline |
| `debug_yolo_rgb_bgr.py` | Tests RGB vs BGR color space impact on detection accuracy |
| `debug_yolo_upload.py` | Simulates user upload detection flow |
| `verify_fix.py` | Verifies callback function with mocked Streamlit |
| `actual_verify.py` | Verifies callback with real DepthEstimator (no mocks) |

---

## 16. How to Run

### Prerequisites

- Python 3.9+
- GPU recommended (CUDA) for faster inference; CPU is also supported

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Pothole-Detection

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install streamlit-geolocation pydeck geopy pandas pillow torch torchvision
```

### Running the Application

```bash
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

### Configuration

- **Confidence Threshold**: Adjustable via the sidebar slider (0.1–1.0, default: 0.3)
- Lower confidence = more detections (but possibly more false positives)
- Higher confidence = fewer but more certain detections

---

## 17. License

**MIT License** — Copyright (c) 2024 Santhosh

This project is open source. You are free to use, modify, and distribute it under the MIT License terms.

---

## Summary of Internal Working

| Component | Technology | What It Does |
|-----------|-----------|--------------|
| **Web Interface** | Streamlit | Multi-page app with navigation, file uploads, maps, and interactive tables |
| **Object Detection** | YOLOv8s-seg (custom trained) | Detects potholes with pixel-level segmentation masks |
| **Depth Estimation** | Depth Anything V2 Small | Infers monocular depth from a single image |
| **Severity Scoring** | Custom algorithm | Combines depth contrast (30%), depth variance (30%), and size (40%) |
| **Severity Classification** | Threshold-based | Low (<0.30), Medium (0.30–0.55), High (≥0.55) |
| **Location Services** | Geopy + Nominatim + PyDeck | GPS capture, reverse geocoding, 3D map rendering |
| **Repair Verification** | Haversine + HSV histogram + YOLO | 3-step location, scene, and AI-based verification |
| **Data Persistence** | JSON + filesystem | No database — flat JSON file + image directory |
| **Video Analysis** | OpenCV VideoCapture | Frame-by-frame detection with real-time FPS display |
