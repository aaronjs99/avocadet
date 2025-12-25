#!/usr/bin/env python3
"""
Avocadet Demo Script

Demonstrates avocado detection on various video sources.
"""

import argparse
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from avocadet import LivestreamProcessor


def on_frame_detected(result):
    """Callback for each processed frame."""
    if result.count > 0:
        print(f"Detected {result.count} object(s) | FPS: {result.fps:.1f}")
        for i, (det, analysis) in enumerate(zip(result.detections, result.analyses)):
            print(f"  #{i+1}: {analysis.ripeness.value}, {analysis.size_category.value}, "
                  f"color: {analysis.dominant_color_name}")


def main():
    parser = argparse.ArgumentParser(description="Avocadet Demo")
    parser.add_argument(
        "--source", "-s",
        default=0,
        help="Video source (0 for webcam, path for file, URL for stream)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to custom YOLO model"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detection details to console"
    )
    
    args = parser.parse_args()
    
    # Parse source
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass
    
    print("=" * 50)
    print("  AVOCADET - Avocado Detection Demo")
    print("=" * 50)
    print(f"  Source: {source}")
    print(f"  Confidence: {args.confidence}")
    print("=" * 50)
    print()
    print("Controls:")
    print("  q - Quit")
    print("  p - Pause/Resume")
    print("  s - Save screenshot")
    print()
    
    # Create processor
    processor = LivestreamProcessor(
        source=source,
        model_path=args.model,
        confidence_threshold=args.confidence,
        on_frame_callback=on_frame_detected if args.verbose else None
    )
    
    try:
        processor.run(show_window=True)
    except KeyboardInterrupt:
        print("\nStopping demo...")
    finally:
        processor.stop()
        print("Demo ended.")


if __name__ == "__main__":
    main()
