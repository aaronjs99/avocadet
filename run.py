#!/usr/bin/env python3
"""
Avocadet - Avocado Detection from Livestream

Main entry point for running avocado detection on video streams.

Usage:
    python run.py                          # Use webcam (default)
    python run.py --source 0               # Use specific webcam
    python run.py --source video.mp4       # Use video file
    python run.py --source rtsp://ip/stream  # Use RTSP stream
    python run.py --help                   # Show all options
"""

import argparse
import sys
import os

# Add src to path for development without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from avocadet import LivestreamProcessor


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Avocadet - Real-time avocado detection from livestream video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                         # Use webcam
  python run.py --source 0              # Use webcam 0
  python run.py --source video.mp4      # Use video file
  python run.py --source rtsp://ip:554/stream  # Use RTSP stream
  python run.py --confidence 0.3        # Lower confidence threshold

Keyboard Controls (when window is open):
  q - Quit
  p - Pause/Resume
  s - Save screenshot
        """
    )
    
    parser.add_argument(
        "--source", "-s",
        default=0,
        help="Video source: webcam ID (0, 1, ...), file path, or stream URL"
    )
    
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to custom YOLO model weights (default: yolov8n.pt)"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Minimum confidence threshold 0.0-1.0 (default: 0.5)"
    )
    
    parser.add_argument(
        "--skip-frames", "-k",
        type=int,
        default=1,
        help="Process every Nth frame for performance (default: 1)"
    )
    
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without display window (headless mode)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detection details to console"
    )
    
    parser.add_argument(
        "--mode",
        choices=["yolo", "segment", "hybrid"],
        default="hybrid",
        help="Detection mode: yolo (YOLO only), segment (color-based), hybrid (both, default)"
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
    print("  🥑 AVOCADET - Avocado Detection")
    print("=" * 50)
    print(f"  Source: {source}")
    print(f"  Mode: {args.mode}")
    print(f"  Confidence: {args.confidence}")
    print(f"  Model: {args.model or 'yolov8n.pt (default)'}")
    print("=" * 50)
    print()
    
    if not args.no_window:
        print("Controls: q=quit, p=pause, s=screenshot")
        print()
    
    # Callback for verbose mode
    def on_frame(result):
        if result.count > 0:
            print(f"[{result.fps:.1f} FPS] Detected {result.count} object(s)")
            for i, (det, analysis) in enumerate(zip(result.detections, result.analyses)):
                print(f"  #{i+1}: {analysis.ripeness.value}, "
                      f"{analysis.size_category.value}, "
                      f"conf={det.confidence:.0%}")
    
    # Create and run processor
    processor = LivestreamProcessor(
        source=source,
        model_path=args.model,
        confidence_threshold=args.confidence,
        process_every_n_frames=args.skip_frames,
        on_frame_callback=on_frame if args.verbose else None,
        mode=args.mode
    )
    
    try:
        processor.run(show_window=not args.no_window)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        processor.stop()
        print("Done.")


if __name__ == "__main__":
    main()
