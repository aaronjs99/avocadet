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

# Add current directory to path if needed (implicit for script execution)
# sys.path.insert(0, os.path.dirname(__file__))

from avocadet_lib import LivestreamProcessor, ConfigLoader


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Avocadet - Real-time flower detection (Standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--source", "-s", default=None, help="Video source (overrides config)"
    )
    parser.add_argument(
        "--config-dir", default="config", help="Path to config directory"
    )
    parser.add_argument(
        "--model", "-m", default=None, help="Model path (overrides config)"
    )
    parser.add_argument("--backend", default=None, help="Backend (overrides config)")
    parser.add_argument(
        "--confidence", "-c", type=float, default=None, help="Confidence threshold"
    )
    parser.add_argument("--width", type=int, default=1920, help="Camera width")
    parser.add_argument("--height", type=int, default=1080, help="Camera height")
    parser.add_argument("--no-window", action="store_true", help="Headless mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-frames", type=int, default=1, help="Skip N frames")

    args = parser.parse_args()

    # Load Configuration
    config_loader = ConfigLoader(args.config_dir)

    # Resolve Source
    source = args.source
    if source is None:
        # Try config
        # We don't have a 'source' key in camera.yaml explicitly standard,
        # but we can look for it or default to 0
        source = 0

    # Try converting source to int
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # Print banner
    print("=" * 50)
    print("  🌸/🥑 AVOCADET - Standalone Runner")
    print("=" * 50)
    print(f"  Config Dir: {args.config_dir}")
    print(f"  Source: {source}")
    print("=" * 50)

    # Verbose callback
    last_print = 0

    def on_frame(result):
        nonlocal last_print
        now = time.time()
        if result.count > 0:
            print(f"[{result.fps:.1f} FPS] Detected {result.count} object(s)")
            for i, (det, analysis) in enumerate(
                zip(result.detections, result.analyses)
            ):
                print(f"  #{i+1}: {det.class_name} conf={det.confidence:.0%}")
            last_print = now
        elif now - last_print > 1.0:
            print(f"[{result.fps:.1f} FPS] No detections...")
            last_print = now

    try:
        processor = LivestreamProcessor(
            source=source,
            model_path=args.model,
            confidence_threshold=args.confidence,
            process_every_n_frames=args.skip_frames,
            on_frame_callback=on_frame if args.verbose else None,
            backend=args.backend,
            width=args.width,
            height=args.height,
            config=config_loader.config,
        )

        processor.run(show_window=not args.no_window)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
