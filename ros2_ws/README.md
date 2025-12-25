# Avocadet ROS2 Integration

This branch contains ROS2 integration for the Avocadet avocado detection system.

## Package Structure

```
ros2_ws/
├── src/
│   ├── avocadet_msgs/          # Custom message definitions
│   │   └── msg/
│   │       ├── BoundingBox.msg
│   │       ├── Color.msg
│   │       ├── AvocadoDetection.msg
│   │       └── AvocadoDetectionArray.msg
│   └── avocadet_ros/           # Main ROS2 package
│       ├── avocadet_ros/
│       │   └── detector_node.py
│       ├── launch/
│       │   └── detector.launch.py
│       └── config/
```

## Installation

### Prerequisites

- ROS2 Humble/Iron/Jazzy
- Python 3.8+
- OpenCV
- cv_bridge

### Build

```bash
# Clone repository
git clone -b ros2 https://github.com/aaronjs99/avocadet.git
cd avocadet

# Install base package
pip install -e .

# Build ROS2 workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Usage

### Launch Detector Node

```bash
# Basic launch
ros2 launch avocadet_ros detector.launch.py

# With custom parameters
ros2 launch avocadet_ros detector.launch.py \
    image_topic:=/camera/color/image_raw \
    mode:=yolo \
    confidence:=0.6
```

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | sensor_msgs/Image | Input camera image (subscribe) |
| `/avocadet/detections` | avocadet_msgs/AvocadoDetectionArray | Detection results (publish) |
| `/avocadet/annotated_image` | sensor_msgs/Image | Annotated output image (publish) |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_topic` | string | `/camera/image_raw` | Input image topic |
| `model_path` | string | `""` | Path to custom YOLO model |
| `confidence_threshold` | float | `0.5` | Detection confidence |
| `mode` | string | `hybrid` | Detection mode |
| `publish_annotated` | bool | `true` | Publish annotated images |

## Gazebo Integration

### Using with Gazebo Camera

```bash
# Launch Gazebo (example with TurtleBot3)
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Launch avocadet with Gazebo camera topic
ros2 launch avocadet_ros detector.launch.py \
    image_topic:=/camera/image_raw
```

### Custom Gazebo World with Avocados

Create a Gazebo world with avocado models and camera sensors for simulation testing.

## Message Definitions

### AvocadoDetection.msg

```
avocadet_msgs/BoundingBox bbox
float32 confidence
string ripeness          # unripe, nearly_ripe, ripe, overripe
string size_category     # small, medium, large
float32 relative_size
avocadet_msgs/Color dominant_color
```

### AvocadoDetectionArray.msg

```
std_msgs/Header header
avocadet_msgs/AvocadoDetection[] detections
uint32 count
```

## Example: Processing Detections

```python
import rclpy
from rclpy.node import Node
from avocadet_msgs.msg import AvocadoDetectionArray


class AvocadoCounter(Node):
    def __init__(self):
        super().__init__('avocado_counter')
        self.sub = self.create_subscription(
            AvocadoDetectionArray,
            '/avocadet/detections',
            self.callback,
            10
        )

    def callback(self, msg):
        ripe_count = sum(1 for d in msg.detections if d.ripeness == 'ripe')
        self.get_logger().info(f'Total: {msg.count}, Ripe: {ripe_count}')


def main():
    rclpy.init()
    node = AvocadoCounter()
    rclpy.spin(node)
    rclpy.shutdown()
```

## License

MIT
