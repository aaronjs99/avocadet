# Avocadet

Real-time avocado detection, counting, and ripeness analysis from video feeds.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)

## Demo

https://github.com/user-attachments/assets/demo/video.mp4

## Features

- **Object Detection** - YOLOv8-based detection with custom training support
- **Ripeness Analysis** - Color-based classification (unripe, nearly ripe, ripe, overripe)
- **Size Estimation** - Relative size categorization (small, medium, large)
- **Live Statistics** - Real-time count, FPS, and analysis overlay
- **Interactive Controls** - Adjustable parameters via trackbars
- **Multi-source Input** - Webcam, video files, RTSP/HTTP streams

## Installation

```bash
git clone https://github.com/aaronjs99/avocadet.git
cd avocadet
pip install -e .
```

## Quick Start

```bash
# Webcam
python run.py

# Video file
python run.py --source path/to/video.mp4

# RTSP stream
python run.py --source rtsp://camera-ip:554/stream

# With custom trained model
python run.py --model path/to/model.pt --source video.mp4
```

## Detection Modes

| Mode | Description | Command |
|------|-------------|---------|
| `hybrid` | Combines color segmentation + YOLO (default) | `--mode hybrid` |
| `yolo` | YOLO only, best with custom model | `--mode yolo` |
| `segment` | Color segmentation only, no model needed | `--mode segment` |

## Training a Custom Model

1. **Annotate frames from your video:**
   ```bash
   python tools/annotate.py --video your_video.mp4 --every 20
   ```

2. **Train YOLOv8 on your annotations:**
   ```bash
   python tools/train.py --dataset datasets/avocado_custom --epochs 50
   ```

3. **Run with your trained model:**
   ```bash
   python run.py --model path/to/model.pt --mode yolo
   ```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause/Resume |
| `s` | Save screenshot |
| `f` | Toggle fullscreen |

## Documentation

- [Architecture Overview](docs/architecture.md) - Detailed codebase explanation
- [API Reference](docs/api.md) - Complete API documentation

## Package Structure

```
avocadet/
├── run.py              # Main entry point
├── tools/
│   ├── train.py        # Model training script
│   └── annotate.py     # Frame annotation tool
├── src/avocadet/
│   ├── detector.py     # YOLOv8 + segmentation detector
│   ├── analyzer.py     # Color and size analysis
│   ├── segmenter.py    # Color-based segmentation
│   ├── stream.py       # Video stream processor
│   └── visualizer.py   # UI overlay and display
├── docs/               # Documentation
└── tests/              # Unit tests
```

## API Usage

```python
from avocadet import LivestreamProcessor

# Basic usage
processor = LivestreamProcessor(source="video.mp4")
processor.run()

# With custom settings
processor = LivestreamProcessor(
    source=0,
    model_path="path/to/model.pt",
    confidence_threshold=0.5,
    mode="hybrid"
)
processor.run()
```

## License

MIT
