# Architecture Overview

This document provides a detailed explanation of the Avocadet codebase architecture and how each component works together.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         run.py                                  │
│                    (Entry Point / CLI)                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LivestreamProcessor                           │
│                      (stream.py)                                │
│  - Manages video capture from various sources                   │
│  - Orchestrates detection, analysis, and visualization          │
│  - Handles user input and interactive controls                  │
└───────┬─────────────────┬─────────────────┬─────────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────────────────┐
│AvocadoDetector│ │ ColorAnalyzer │ │      Visualizer           │
│ (detector.py) │ │ SizeEstimator │ │    (visualizer.py)        │
│               │ │ (analyzer.py) │ │                           │
│ - YOLO detect │ │ - HSV analysis│ │ - Draws bounding boxes    │
│ - Segmentation│ │ - Ripeness    │ │ - Renders stats panel     │
│ - Hybrid mode │ │ - Size calc   │ │ - Color swatches          │
└───────┬───────┘ └───────────────┘ └───────────────────────────┘
        │
        ▼
┌───────────────────────┐
│  ColorBasedSegmenter  │
│   (segmenter.py)      │
│                       │
│ - HSV color filtering │
│ - Contour detection   │
│ - Shape analysis      │
└───────────────────────┘
```

## Component Details

### 1. Entry Point (`run.py`)

The main entry point that provides CLI argument parsing and initializes the processing pipeline.

**Key Arguments:**
- `--source`: Video source (webcam ID, file path, or stream URL)
- `--model`: Path to custom YOLO model weights
- `--mode`: Detection mode (`yolo`, `segment`, or `hybrid`)
- `--confidence`: Minimum detection confidence threshold

**Flow:**
1. Parse command-line arguments
2. Create `LivestreamProcessor` with specified configuration
3. Start the processing loop
4. Handle keyboard interrupts gracefully

---

### 2. Livestream Processor (`stream.py`)

The central orchestrator that manages the video processing pipeline.

**Class: `LivestreamProcessor`**

```python
def __init__(
    source: Union[int, str],      # Video source
    model_path: Optional[str],     # YOLO model path
    confidence_threshold: float,   # Detection threshold
    process_every_n_frames: int,   # Frame skip for performance
    mode: str                      # Detection mode
)
```

**Responsibilities:**
- Opens and manages video capture from multiple source types
- Creates trackbar controls for real-time parameter adjustment
- Coordinates detection, analysis, and visualization
- Handles keyboard input (quit, pause, screenshot, fullscreen)

**Processing Loop:**
```
For each frame:
    1. Read trackbar values → Update detector parameters
    2. Run detection → Get bounding boxes
    3. For each detection:
        - Analyze color → Determine ripeness
        - Estimate size → Categorize as small/medium/large
    4. Visualize results → Draw overlays
    5. Display frame → Handle user input
```

---

### 3. Avocado Detector (`detector.py`)

Handles object detection using multiple strategies.

**Class: `AvocadoDetector`**

**Detection Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `yolo` | Uses YOLOv8 only | Best with custom-trained model |
| `segment` | Color-based segmentation only | Fast, no model needed |
| `hybrid` | Combines both methods | Default, best coverage |

**YOLO Detection Flow:**
```python
1. Load YOLOv8 model (default: yolov8n.pt)
2. Run inference on frame
3. Filter detections by confidence
4. Filter by class (fruit-like objects or 'avocado')
5. Return list of Detection objects
```

**Hybrid Mode:**
```python
1. Run segmentation → Get color-based detections
2. Run YOLO → Get model-based detections
3. Merge results using IoU overlap detection
4. Remove duplicates (IoU > 0.5)
5. Return combined detections
```

**Data Class: `Detection`**
```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float                 # 0.0 - 1.0
    class_name: str                   # "avocado"
```

---

### 4. Color-Based Segmenter (`segmenter.py`)

Detects avocados based on color and shape characteristics.

**Class: `ColorBasedSegmenter`**

**HSV Color Range:**
```python
HSV_LOWER = [25, 40, 40]   # Dark yellow-green
HSV_UPPER = [85, 200, 180] # Darker greens (excludes bright leaves)
```

**Detection Pipeline:**
```
1. Convert frame to HSV color space
2. Create binary mask using color range
3. Apply morphological operations (close + open)
4. Find contours in the mask
5. For each contour:
   - Filter by area (min: 800, max: 30000)
   - Calculate circularity (min: 0.45)
   - Check aspect ratio (0.5 - 2.0)
   - Check convexity (min: 0.7)
6. Return valid contours as SegmentedObject
```

**Why These Filters?**
- **Area**: Filters out small noise and large background regions
- **Circularity**: Avocados are rounder than leaves (which are elongated)
- **Aspect Ratio**: Avocados are oval, not thin like leaves
- **Convexity**: Avocados are smooth, jagged leaves have low convexity

---

### 5. Color Analyzer (`analyzer.py`)

Analyzes detected regions to determine ripeness and dominant color.

**Class: `ColorAnalyzer`**

**Ripeness Classification:**

| Ripeness | Hue Range | Description |
|----------|-----------|-------------|
| `unripe` | 35-85 | Bright green |
| `nearly_ripe` | 25-35 | Yellow-green |
| `ripe` | 15-25 | Brown-green |
| `overripe` | 0-15, 160+ | Dark brown/black |

**Algorithm:**
```python
1. Extract ROI (region of interest) from bounding box
2. Convert to HSV color space
3. Apply K-means clustering (k=3) to find dominant colors
4. Select the largest cluster as dominant color
5. Map hue value to ripeness category
```

**Class: `SizeEstimator`**

**Size Categories:**
```python
SMALL_THRESHOLD = 0.02   # < 2% of frame area
LARGE_THRESHOLD = 0.08   # > 8% of frame area
```

Sizes are relative to frame dimensions, not absolute measurements.

---

### 6. Visualizer (`visualizer.py`)

Renders detection results and statistics overlay.

**Class: `Visualizer`**

**Visual Elements:**
1. **Bounding Boxes**: Color-coded by ripeness
2. **Labels**: Detection ID, ripeness, size, confidence
3. **Color Swatches**: Shows dominant color of each detection
4. **Stats Panel**: Semi-transparent overlay with:
   - Total avocado count
   - Current FPS
   - Ripeness breakdown

**Color Scheme:**
```python
RIPENESS_COLORS = {
    'unripe':      (0, 255, 0),    # Green
    'nearly_ripe': (0, 255, 255),  # Yellow
    'ripe':        (0, 165, 255),  # Orange
    'overripe':    (0, 0, 255)     # Red
}
```

---

## Training Pipeline

### Annotation Tool (`tools/annotate.py`)

Interactive tool for creating training data from video frames.

**Workflow:**
1. Extract frames at specified interval
2. For each frame, user draws bounding boxes
3. Save images to `datasets/avocado_custom/images/train/`
4. Save YOLO-format labels to `datasets/avocado_custom/labels/train/`
5. Generate `data.yaml` configuration file

**YOLO Label Format:**
```
<class_id> <center_x> <center_y> <width> <height>
```
All values normalized to 0-1 range.

### Training Script (`tools/train.py`)

Fine-tunes YOLOv8 on custom dataset.

**Key Parameters:**
- `--epochs`: Training iterations (default: 50)
- `--batch`: Batch size (reduce if OOM)
- `--imgsz`: Image size (default: 640)
- `--device`: GPU ID or "cpu"

**Output:**
- Best model saved to specified output path
- Training metrics and plots in `runs/train/`

---

## Data Flow Summary

```
Video Frame
    │
    ▼
┌─────────────────────┐
│  AvocadoDetector    │
│  detect(frame)      │
└─────────┬───────────┘
          │
          ▼
    List[Detection]
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│   ColorAnalyzer     │     │   SizeEstimator     │
│   analyze(roi)      │     │   estimate(bbox)    │
└─────────┬───────────┘     └─────────┬───────────┘
          │                           │
          ▼                           ▼
    AvocadoAnalysis ◄─────────────────┘
    (color, ripeness, size)
          │
          ▼
┌─────────────────────┐
│     Visualizer      │
│   draw(frame, ...)  │
└─────────┬───────────┘
          │
          ▼
    Annotated Frame
```

---

## Performance Considerations

1. **Frame Skipping**: Use `--skip-frames N` to process every Nth frame
2. **Model Size**: Smaller models (yolov8n) are faster but less accurate
3. **Segmentation Mode**: Faster than YOLO, good for real-time on CPU
4. **Resolution**: Lower resolution = faster processing

## Extending the System

### Adding New Detection Classes

1. Modify `ColorBasedSegmenter` HSV ranges
2. Train custom YOLO model with new classes
3. Update `AvocadoDetector.FRUIT_CLASSES`

### Custom Ripeness Logic

Modify `ColorAnalyzer.RIPENESS_RANGES` to adjust hue thresholds.

### Adding New Video Sources

`LivestreamProcessor` accepts any OpenCV-compatible source string.
