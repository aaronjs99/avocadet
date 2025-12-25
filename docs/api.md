# API Reference

Complete API documentation for the Avocadet package.

## Core Classes

### LivestreamProcessor

Main class for processing video streams.

```python
from avocadet import LivestreamProcessor

processor = LivestreamProcessor(
    source=0,                        # int, str - video source
    model_path=None,                 # Optional[str] - YOLO model path
    confidence_threshold=0.5,        # float - detection confidence
    process_every_n_frames=1,        # int - frame skip
    on_frame_callback=None,          # Optional[Callable] - per-frame callback
    mode="hybrid"                    # str - detection mode
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `run(show_window=True)` | Start processing loop |
| `run_async(show_window=True)` | Start in background thread |
| `stop()` | Stop processing |
| `toggle_pause()` | Pause/resume |
| `process_single_frame(frame)` | Process one frame |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_running` | bool | Processing status |
| `is_paused` | bool | Pause status |

---

### AvocadoDetector

Object detection using YOLO and/or color segmentation.

```python
from avocadet import AvocadoDetector

detector = AvocadoDetector(
    model_path=None,                 # Optional[str]
    confidence_threshold=0.5,        # float
    device="auto",                   # str - "auto", "cpu", "cuda"
    mode="hybrid"                    # str - "yolo", "segment", "hybrid"
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `detect(frame)` | List[Detection] | Detect objects in frame |
| `detect_batch(frames)` | List[List[Detection]] | Batch detection |

---

### ColorAnalyzer

Analyzes color and determines ripeness.

```python
from avocadet import ColorAnalyzer

analyzer = ColorAnalyzer()
color, color_name, ripeness = analyzer.analyze(frame, bbox)
```

**Returns:**
- `color`: Tuple[int, int, int] - BGR color
- `color_name`: str - Human-readable color name
- `ripeness`: Ripeness - Enum value

---

### SizeEstimator

Estimates relative size of detections.

```python
from avocadet import SizeEstimator

estimator = SizeEstimator()
estimator.update_frame_size(width, height)
size_category, relative_size = estimator.estimate(bbox)
```

**Returns:**
- `size_category`: SizeCategory - SMALL, MEDIUM, or LARGE
- `relative_size`: float - Proportion of frame area

---

### Visualizer

Renders detection overlays.

```python
from avocadet import Visualizer

viz = Visualizer()
annotated = viz.draw(frame, detections, analyses, count, fps)
```

---

## Data Classes

### Detection

```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str = "avocado"
    
    # Properties
    center -> Tuple[int, int]
    width -> int
    height -> int
    area -> int
```

### AvocadoAnalysis

```python
@dataclass
class AvocadoAnalysis:
    dominant_color: Tuple[int, int, int]  # BGR
    dominant_color_name: str
    ripeness: Ripeness
    size_category: SizeCategory
    relative_size: float
```

### FrameResult

```python
@dataclass
class FrameResult:
    frame: np.ndarray
    detections: List[Detection]
    analyses: List[AvocadoAnalysis]
    count: int
    fps: float
    timestamp: float
```

---

## Enums

### Ripeness

```python
class Ripeness(Enum):
    UNRIPE = "unripe"
    NEARLY_RIPE = "nearly_ripe"
    RIPE = "ripe"
    OVERRIPE = "overripe"
```

### SizeCategory

```python
class SizeCategory(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
```

---

## Example Usage

### Basic Detection

```python
from avocadet import LivestreamProcessor

# Process webcam
processor = LivestreamProcessor(source=0)
processor.run()
```

### Custom Callback

```python
def on_detection(result):
    print(f"Detected {result.count} avocados at {result.fps:.1f} FPS")
    for det, analysis in zip(result.detections, result.analyses):
        print(f"  - {analysis.ripeness.value}, {analysis.size_category.value}")

processor = LivestreamProcessor(
    source="video.mp4",
    on_frame_callback=on_detection
)
processor.run()
```

### Single Frame Processing

```python
import cv2
from avocadet import LivestreamProcessor

processor = LivestreamProcessor(source=0)
frame = cv2.imread("image.jpg")
result = processor.process_single_frame(frame)

print(f"Found {result.count} avocados")
```

### Headless Mode

```python
processor = LivestreamProcessor(
    source="video.mp4",
    on_frame_callback=lambda r: print(f"Count: {r.count}")
)
processor.run(show_window=False)
```
