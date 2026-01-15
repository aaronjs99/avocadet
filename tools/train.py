#!/usr/bin/env python3
"""
YOLOv8 Fine-tuning Script for Avocado Detection

This script downloads an avocado dataset from Roboflow and fine-tunes YOLOv8.
"""

import os
from pathlib import Path


def download_dataset():
    """Download avocado dataset from Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        os.system("pip install roboflow")
        from roboflow import Roboflow

    # Initialize Roboflow - you'll need an API key
    # Get one free at: https://roboflow.com/
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        print("\n" + "=" * 60)
        print("ROBOFLOW API KEY REQUIRED")
        print("=" * 60)
        print("\n1. Create a free account at: https://roboflow.com/")
        print("2. Go to Settings -> API Keys")
        print("3. Copy your Private API Key")
        print("4. Run: export ROBOFLOW_API_KEY='your-key-here'")
        print("5. Then run this script again")
        print("\nAlternatively, you can download manually:")
        print("  https://universe.roboflow.com/brad-dwyer/avocado-dacgw")
        print("  Download in YOLOv8 format to: datasets/avocado/")
        print("=" * 60 + "\n")
        return None

    rf = Roboflow(api_key=api_key)

    # Download avocado detection dataset
    # Source: https://universe.roboflow.com/brad-dwyer/avocado-dacgw
    project = rf.workspace("brad-dwyer").project("avocado-dacgw")
    version = project.version(1)
    dataset = version.download("yolov8", location="datasets/avocado")

    print(f"\nDataset downloaded to: {dataset.location}")
    return dataset.location


def train_model(
    dataset_path: str = "datasets/avocado",
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",  # GPU device, or "cpu"
):
    """
    Fine-tune YOLOv8 on the avocado dataset.

    Args:
        dataset_path: Path to downloaded dataset
        model_size: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: GPU device ID or "cpu"
    """
    from ultralytics import YOLO

    # Find data.yaml in dataset
    data_yaml = Path(dataset_path) / "data.yaml"
    if not data_yaml.exists():
        # Try one level deeper
        for p in Path(dataset_path).rglob("data.yaml"):
            data_yaml = p
            break

    if not data_yaml.exists():
        print(f"Error: Could not find data.yaml in {dataset_path}")
        print("Please ensure the dataset is downloaded correctly.")
        return None

    print(f"\n{'='*60}")
    print("STARTING YOLOV8 FINE-TUNING FOR AVOCADO DETECTION")
    print(f"{'='*60}")
    print(f"  Model: yolov8{model_size}.pt")
    print(f"  Dataset: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Load pre-trained model
    model = YOLO(f"yolov8{model_size}.pt")

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/train",
        name="avocado_detector",
        exist_ok=True,
        patience=20,  # Early stopping
        save=True,
        plots=True,
    )

    # Save best model to models directory
    best_model_path = Path("runs/train/avocado_detector/weights/best.pt")
    if best_model_path.exists():
        output_path = Path("models/avocado_yolov8.pt")
        output_path.parent.mkdir(exist_ok=True)
        import shutil

        shutil.copy(best_model_path, output_path)
        print(f"\n✅ Best model saved to: {output_path}")
        print(
            f"\nTo use: python run.py --model models/avocado_yolov8.pt --source your_video.mp4"
        )

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for avocado detection"
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Only download dataset"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip dataset download"
    )
    parser.add_argument(
        "--model-size",
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size (default: n)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument("--device", default="0", help="Device: GPU id or 'cpu'")
    parser.add_argument("--dataset", default="datasets/avocado", help="Dataset path")

    args = parser.parse_args()

    # Download dataset
    if not args.skip_download:
        dataset_path = download_dataset()
        if dataset_path:
            args.dataset = dataset_path

    if args.download_only:
        return

    # Train model
    train_model(
        dataset_path=args.dataset,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
