#!/usr/bin/env python3
"""
Avocadet - Flower Detection from Livestream
(Non-ROS Standalone Runner)

Main entry point for running flower detection on video streams without ROS.

Usage:
    python run.py --model path/to/model.pt # Use webcam (default)
    python run.py --source 0               # Use specific webcam
    python run.py --source video.mp4       # Use video file
    python run.py --backend onnx           # Use ONNX backend
"""

import argparse
import sys
import os
import time

# Add src to path for development without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from avocadet_lib import LivestreamProcessor


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Avocadet - Real-time flower detection from livestream video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --model models/best.pt  # Use webcam with specific model
  python run.py --source 0              # Use webcam 0
  python run.py --source video.mp4      # Use video file
  python run.py --backend onnx          # Use ONNX backend
        """,
    )

    parser.add_argument(
        "--source",
        "-s",
        default="0",
        help="Video source: webcam ID (0, 1, ...), file path, or stream URL",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Capture width (for webcam). Default: 640",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Capture height (for webcam). Default: 480",
    )

    parser.add_argument(
        "--model",
        "-m",
        default=None,
        required=True,
        help="Path to model weights (required)",
    )

    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.25,
        help="Minimum confidence threshold 0.0-1.0 (default: 0.25)",
    )

    parser.add_argument(
        "--skip-frames",
        "-k",
        type=int,
        default=1,
        help="Process every Nth frame for performance (default: 1)",
    )

    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without display window (headless mode)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detection details to console",
    )

    parser.add_argument(
        "--backend",
        choices=["ultralytics", "onnx", "tensorrt"],
        default="ultralytics",
        help="Inference backend (default: ultralytics)",
    )

    args = parser.parse_args()

    # Parse source - try to convert to int for webcam
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass  # Keep as string (file path or URL)

    # Print banner
    print("=" * 50)
    print("  🌸/🥑 AVOCADET - Unified Object Detection (Standalone)")
    print("=" * 50)
    print(f"  Source: {source}")
    print(f"  Backend: {args.backend}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Model: {args.model}")
    print("=" * 50)
    print()

    if not args.no_window:
        print("Controls: q=quit, p=pause, s=screenshot")
        print()

    # Callback for verbose mode
    last_print = 0

    def on_frame(result):
        nonlocal last_print
        now = time.time()
        if result.count > 0:
            print(f"[{result.fps:.1f} FPS] Detected {result.count} object(s)")
            for i, (det, analysis) in enumerate(
                zip(result.detections, result.analyses)
            ):
                print(f"  #{i+1}: {det.class_name} " f"conf={det.confidence:.0%}")
            last_print = now
        elif now - last_print > 1.0:  # Heartbeat every 1s
            print(f"[{result.fps:.1f} FPS] No detections...")
            last_print = now

    # Create and run processor
    try:
        # Note: LivestreamProcessor uses FlowerDetector internally now
        # We need to make sure we passed the right args.
        # Looking at stream.py earlier, it was initializing:
        # FlowerDetector(model_path=model_path, confidence_threshold=confidence_threshold, backend="ultralytics")
        # Wait, I hardcoded backend="ultralytics" in stream.py in the previous step!
        # I need to fix stream.py to accept backend argument first.

        processor = LivestreamProcessor(
            source=source,
            model_path=args.model,
            confidence_threshold=args.confidence,
            process_every_n_frames=args.skip_frames,
            on_frame_callback=on_frame if args.verbose else None,
            backend=args.backend,
            width=args.width,
            height=args.height,
        )
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return

    try:
        processor.run(show_window=not args.no_window)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        processor.stop()
        print("Done.")


if __name__ == "__main__":
    main()
