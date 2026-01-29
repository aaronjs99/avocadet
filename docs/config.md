# Avocadet Configuration Guide

Avocadet uses a modular configuration system located in the `config/` directory.

## Core Configuration Files

### `detector.yaml`
Controls the detection model and backend.
- `model_path`: Path to YOLO weights or TensorRT engine.
- `backend`: `ultralytics` (dev), `tensorrt` (production), or `onnx`.
- `input_size`: [width, height] for inference.

### `geometry.yaml`
Manages lens models and image rectification.
- `lens_model`: `fisheye` or `pinhole`.
- `rectify_enabled`: If true, incoming images are unwarped before detection.
- `calibration_source`: `camera_info` (preferred) or `yaml` manual values.

### `tiling.yaml`
Fallback strategy for high-resolution detection.
- `tiling_enabled`: Activates sliding window inference.
- `grid`: [rows, cols]
- `overlap_px`: Pixel overlap between tiles.

### `runtime.yaml`
Performance tuning.
- `target_hz`: Goal frame rate.
- `max_end_to_end_latency_ms`: Max allowable latency before dropping frames.
- `worker_threads`: Number of inference threads.

## Overrides
All parameters can be overridden via ROS 2 launch arguments or parameter server.
Example:
```bash
ros2 launch avocadet detector.launch.py model_path:=models/custom.pt backend:=tensorrt
```
