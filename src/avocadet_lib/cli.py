"""
Command-line interface for avocadet.
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Avocadet - Real-time avocado detection from livestream video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  avocadet --source 0                    # Use webcam
  avocadet --source video.mp4            # Use video file
  avocadet --source rtsp://ip:554/stream # Use RTSP stream
  avocadet --source 0 --confidence 0.3   # Lower confidence threshold
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
        help="Path to custom YOLO model weights"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (0.0-1.0)"
    )
    
    parser.add_argument(
        "--skip-frames", "-k",
        type=int,
        default=1,
        help="Process every Nth frame (for performance)"
    )
    
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without display window (headless mode)"
    )
    
    args = parser.parse_args()
    
    # Parse source - try to convert to int for webcam
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass  # Keep as string (file path or URL)
    
    # Import here to avoid slow startup for help text
    from .stream import LivestreamProcessor
    
    # Create and run processor
    processor = LivestreamProcessor(
        source=source,
        model_path=args.model,
        confidence_threshold=args.confidence,
        process_every_n_frames=args.skip_frames
    )
    
    try:
        processor.run(show_window=not args.no_window)
    except KeyboardInterrupt:
        print("\nStopping...")
        processor.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
