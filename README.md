# Avocadet - ROS Package

Real-time avocado detection, counting, and ripeness analysis for ROS.

![ROS](https://img.shields.io/badge/ROS-Noetic-blue.svg)
![Python](https://img.shields.io/badge/Python-2.7%20%7C%203.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Note**: This is the ROS1 branch (master). For ROS2 (Humble/Iron/Jazzy), see the [ros2 branch](https://github.com/aaronjs99/avocadet/tree/ros2).

## Overview

Avocadet is a ROS package for automated avocado detection in agricultural robotics applications. The system combines deep learning-based object detection (YOLOv8) with classical computer vision techniques for:

- Real-time fruit detection and localization
- Ripeness classification (unripe, nearly ripe, ripe, overripe)
- Relative size estimation
- Multi-modal detection (YOLO, color segmentation, or hybrid)

## Installation

### Prerequisites

- ROS Noetic (Ubuntu 20.04)
- Python 3.8+
- OpenCV 4.x
- CUDA (optional, for GPU acceleration)

### Build from Source

```bash
# Create workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# Clone the package
git clone https://github.com/aaronjs99/avocadet.git

# Install Python dependencies
cd avocadet
pip3 install ultralytics opencv-python numpy

# Build the workspace
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Usage

### Basic Launch

```bash
# Launch detector (subscribes to /camera/image_raw)
roslaunch avocadet detector.launch

# With custom camera topic
roslaunch avocadet detector.launch image_topic:=/camera/color/image_raw

# With custom model and confidence
roslaunch avocadet detector.launch \
    model_path:=/path/to/model.pt \
    confidence:=0.6 \
    mode:=yolo
```

### Gazebo Integration

```bash
# Terminal 1: Launch Gazebo simulation
roslaunch your_robot_package gazebo.launch

# Terminal 2: Launch avocadet
roslaunch avocadet detector.launch image_topic:=/robot/camera/image_raw
```

## ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Input camera stream |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/avocadet/detections` | `std_msgs/String` | JSON-formatted detection results |
| `/avocadet/annotated_image` | `sensor_msgs/Image` | Visualized output with bounding boxes |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `~image_topic` | string | `/camera/image_raw` | Camera topic to subscribe |
| `~model_path` | string | `""` | Custom YOLO model path |
| `~confidence_threshold` | float | `0.5` | Detection confidence [0.0-1.0] |
| `~mode` | string | `hybrid` | Detection mode |
| `~publish_annotated` | bool | `true` | Publish annotated images |

### Detection Modes

| Mode | Description |
|------|-------------|
| `hybrid` | Combines YOLO detection + color segmentation (default) |
| `yolo` | YOLOv8 object detection only |
| `segment` | Color-based segmentation only (fast, CPU-friendly) |

## Message Format

Detection results are published as JSON:

```json
{
  "header": {
    "stamp": {"secs": 1703520000, "nsecs": 123456789},
    "frame_id": "camera_optical_frame"
  },
  "count": 3,
  "detections": [
    {
      "bbox": {"x": 100, "y": 150, "width": 80, "height": 120},
      "confidence": 0.923,
      "ripeness": "ripe",
      "size_category": "medium",
      "relative_size": 0.0512,
      "color": {"r": 34, "g": 85, "b": 28}
    }
  ]
}
```

## Package Structure

```
avocadet/
├── package.xml             # ROS package manifest
├── CMakeLists.txt          # Build configuration
├── setup.py                # Python setup for catkin
├── msg/                    # Custom message definitions
│   ├── BoundingBox.msg
│   ├── Color.msg
│   ├── AvocadoDetection.msg
│   └── AvocadoDetectionArray.msg
├── launch/
│   └── detector.launch     # Launch configuration
├── config/                 # Parameter files
├── scripts/
│   └── detector_node.py    # Main detector node
├── src/avocadet/           # Core detection library
│   ├── detector.py         # YOLO + segmentation
│   ├── segmenter.py        # Color-based segmentation
│   ├── analyzer.py         # Ripeness & size analysis
│   └── visualizer.py       # Visualization utilities
├── tools/                  # Training utilities
└── tests/                  # Unit tests
```

## Custom Model Training

Train a custom detection model for your specific avocado varieties:

```bash
# 1. Annotate frames from your video
python3 tools/annotate.py --video input.mp4 --every 20

# 2. Train YOLOv8
python3 tools/train.py --dataset datasets/custom --epochs 50

# 3. Use trained model
roslaunch avocadet detector.launch model_path:=/path/to/best.pt mode:=yolo
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{avocadet2024,
  author = {Aaron JS},
  title = {Avocadet: Real-time Avocado Detection for ROS},
  year = {2024},
  url = {https://github.com/aaronjs99/avocadet}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## See Also

- [ROS2 Version (ros2 branch)](https://github.com/aaronjs99/avocadet/tree/ros2)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
