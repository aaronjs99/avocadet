# Jetson Deployment Guide

For maximum performance on NVIDIA Jetson, use the TensorRT backend.

## Prerequisites
- JetPack 5.x or 6.x
- `tensorrt` python bindings
- `pycuda`

## Steps to Deploy

1. **Export YOLO Model to ONNX**
   ```bash
   yolo export model=models/yolov8n.pt format=onnx opset=12
   ```

2. **Build TensorRT Engine**
   You can use `trtexec` (recommended) or allow Avocadet to build it on first run (experimental).
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=models/yolov8n.onnx --saveEngine=models/yolov8n.engine --fp16
   ```

3. **Configure Avocadet**
   Edit `config/backends.yaml` or just use `detector.yaml`:
   ```yaml
   # config/detector.yaml
   backend: "tensorrt"
   model_path: "models/yolov8n.engine"
   ```

4. **Run**
   ```bash
   ros2 launch avocadet detector.launch.py config_dir:=./config
   ```

## Performance Notes
- **Precision**: FP16 is valid for almost all detection tasks and offers 2x speedup on Jetson Orin/Xavier.
- **Latency**: Ensure `runtime.yaml` has `drop_frames: true` to prevent buffer lag regardless of throughput.
- **Power Mode**: Run `sudo nvpmodel -m 0` (MAXN) or appropriate mode for best results.
