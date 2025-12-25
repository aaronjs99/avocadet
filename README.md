# Avocadet - ROS2 Package

Real-time avocado detection, counting, and ripeness analysis for ROS2.

![ROS2](https://img.shields.io/badge/ROS2-Humble%20%7C%20Iron%20%7C%20Jazzy-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Note**: This is the ROS2 branch. For ROS1 (Noetic), see the [master branch](https://github.com/aaronjs99/avocadet/tree/master).

## Overview

Avocadet is a ROS2 package for automated avocado detection in agricultural robotics applications. The system combines deep learning-based object detection (YOLOv8) with classical computer vision techniques for:

- Real-time fruit detection and localization
- Ripeness classification (unripe, nearly ripe, ripe, overripe)
- Relative size estimation
- Multi-modal detection (YOLO, color segmentation, or hybrid)

## Installation

### Prerequisites

- ROS2 Humble / Iron / Jazzy
- Python 3.8+
- OpenCV 4.x
- CUDA (optional, for GPU acceleration)

### Build from Source

```bash
# Create workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# Clone the package
git clone -b ros2 https://github.com/aaronjs99/avocadet.git

# Install Python dependencies
cd avocadet
pip3 install ultralytics opencv-python numpy

# Build the workspace
cd ~/catkin_ws
colcon build --packages-select avocadet --symlink-install
source install/setup.bash
```

## Usage

### Basic Launch

```bash
# Launch detector (subscribes to /camera/image_raw)
ros2 launch avocadet detector.launch.py

# With custom camera topic
ros2 launch avocadet detector.launch.py image_topic:=/camera/color/image_raw

# With custom model and confidence
ros2 launch avocadet detector.launch.py \
    model_path:=/path/to/model.pt \
    confidence:=0.6 \
    mode:=yolo
```

### Gazebo Integration

```bash
# Terminal 1: Launch Gazebo simulation
ros2 launch your_robot_package gazebo.launch.py

# Terminal 2: Launch avocadet with sim time
ros2 launch avocadet detector.launch.py \
    image_topic:=/robot/camera/image_raw \
    use_sim_time:=true
```

## ROS2 Interface

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
| `image_topic` | string | `/camera/image_raw` | Camera topic to subscribe |
| `model_path` | string | `""` | Custom YOLO model path |
| `confidence_threshold` | float | `0.5` | Detection confidence [0.0-1.0] |
| `mode` | string | `hybrid` | Detection mode |
| `publish_annotated` | bool | `true` | Publish annotated images |
| `use_sim_time` | bool | `false` | Use Gazebo simulation time |

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
    "stamp": {"sec": 1703520000, "nanosec": 123456789},
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
├── package.xml             # ROS2 package manifest
├── CMakeLists.txt          # Build configuration
├── msg/                    # Custom message definitions
│   ├── BoundingBox.msg
│   ├── Color.msg
│   ├── AvocadoDetection.msg
│   └── AvocadoDetectionArray.msg
├── launch/
│   └── detector.launch.py  # Launch configuration
├── config/                 # Parameter files
├── avocadet_ros/           # ROS2 nodes
│   ├── __init__.py
│   └── detector_node.py    # Main detector node
├── src/avocadet/           # Core detection library
│   ├── detector.py         # YOLO + segmentation
│   ├── segmenter.py        # Color-based segmentation
│   ├── analyzer.py         # Ripeness & size analysis
│   ├── stream.py           # Video stream processing
│   └── visualizer.py       # Visualization utilities
├── tools/                  # Training utilities
│   ├── annotate.py         # Dataset annotation
│   └── train.py            # Model training
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
ros2 launch avocadet detector.launch.py model_path:=/path/to/best.pt mode:=yolo
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{avocadet2024,
  author = {Aaron JS},
  title = {Avocadet: Real-time Avocado Detection for ROS2},
  year = {2024},
  url = {https://github.com/aaronjs99/avocadet}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## See Also

- [ROS1 Version (master branch)](https://github.com/aaronjs99/avocadet/tree/master)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
