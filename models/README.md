# Models

This directory contains YOLO model weights for avocado detection.

## Default Model

The package uses `yolov8n.pt` (YOLOv8 nano) by default, which is automatically downloaded on first run.

## Custom Models

To train a custom avocado detection model:

```bash
# 1. Annotate your video frames
python tools/annotate.py --video your_video.mp4 --every 20

# 2. Train on your annotations
python tools/train.py --dataset datasets/avocado_custom --epochs 50

# 3. Your model will be saved here
```

## Usage

```bash
python run.py --model models/your_model.pt --mode yolo
```

## Notes

- Model files (`.pt`) are gitignored to keep the repository lightweight
- Train your own model for best results on your specific use case
- Larger models (yolov8s, yolov8m) are more accurate but slower
