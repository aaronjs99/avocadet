# Models directory

This directory is for storing trained model weights.

## Default Model

By default, avocadet uses `yolov8n.pt` from Ultralytics, which is automatically 
downloaded on first run.

## Custom Models

For better avocado-specific detection, you can:

1. Train a custom model on an avocado dataset
2. Place the weights file (e.g., `avocado_yolov8.pt`) in this directory
3. Use it with: `avocadet --model models/avocado_yolov8.pt`

## Training Resources

- [Roboflow Avocado Datasets](https://universe.roboflow.com/search?q=avocado)
- [Ultralytics Training Guide](https://docs.ultralytics.com/modes/train/)
