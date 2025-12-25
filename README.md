# Avocadet - ROS2 Package

Real-time avocado detection, counting, and ripeness analysis for ROS2.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ROS2](https://img.shields.io/badge/ROS2-Humble%2FIron%2FJazzy-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)

> **Note**: This is the ROS2 branch. For the standalone Python package, see the [master branch](https://github.com/aaronjs99/avocadet/tree/master).

## Overview

This ROS2 package provides avocado detection capabilities for robotic systems. It subscribes to camera image topics and publishes detection results including:

- Bounding boxes
- Ripeness classification (unripe, nearly_ripe, ripe, overripe)
- Size estimation (small, medium, large)
- Confidence scores
- Dominant color

## Installation

### Prerequisites

- ROS2 Humble / Iron / Jazzy
- Python 3.8+
- OpenCV
- cv_bridge

### Build

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone this package
git clone -b ros2 https://github.com/aaronjs99/avocadet.git

# Install Python dependencies
cd avocadet
pip install ultralytics opencv-python numpy

# Build
cd ~/ros2_ws
colcon build --packages-select avocadet
source install/setup.bash
```

## Usage

### Launch Detector

```bash
# Basic launch (subscribes to /camera/image_raw)
ros2 launch avocadet detector.launch.py

# With custom camera topic
ros2 launch avocadet detector.launch.py image_topic:=/camera/color/image_raw

# With custom model
ros2 launch avocadet detector.launch.py model_path:=/path/to/model.pt mode:=yolo
```

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | Input (subscribe) |
| `/avocadet/detections_json` | std_msgs/String | Detection results as JSON |
| `/avocadet/annotated_image` | sensor_msgs/Image | Visualized output |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_topic` | `/camera/image_raw` | Camera topic to subscribe |
| `model_path` | `""` | Custom YOLO model path |
| `confidence_threshold` | `0.5` | Detection confidence |
| `mode` | `hybrid` | Detection mode |
| `publish_annotated` | `true` | Publish annotated images |

## Gazebo Integration

```bash
# Terminal 1: Launch Gazebo with camera
ros2 launch your_robot gazebo.launch.py

# Terminal 2: Launch avocadet
ros2 launch avocadet detector.launch.py image_topic:=/robot/camera/image_raw
```

## Message Format

Detection results are published as JSON:

```json
{
  "count": 3,
  "detections": [
    {
      "bbox": {"x": 100, "y": 150, "width": 80, "height": 120},
      "confidence": 0.92,
      "ripeness": "ripe",
      "size_category": "medium",
      "relative_size": 0.05,
      "color": {"r": 34, "g": 85, "b": 28}
    }
  ]
}
```

## Custom Messages

This package also defines custom ROS2 messages in `msg/`:

- `BoundingBox.msg`
- `Color.msg`
- `AvocadoDetection.msg`
- `AvocadoDetectionArray.msg`

## Package Structure

```
avocadet/
├── package.xml           # ROS2 package manifest
├── CMakeLists.txt        # Build configuration
├── msg/                  # Custom message definitions
├── launch/               # Launch files
├── config/               # Configuration files
├── avocadet_ros/         # ROS2 nodes
│   └── detector_node.py
└── src/avocadet/         # Core detection library
    ├── detector.py
    ├── analyzer.py
    ├── segmenter.py
    └── visualizer.py
```

## License

MIT
