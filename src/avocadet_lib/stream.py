# -*- coding: utf-8 -*-
"""
Livestream Processor Module

This module provides the main video stream processing pipeline for
real-time avocado detection from various video sources.

Authors:
    Aaron John Sabu
    Sunwoong Choi
    Sriram Narasimhan

License:
    MIT License
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union, List
import time
import threading
import cv2
import numpy as np

from .detector import AvocadoDetector, Detection
from .analyzer import (
    ColorAnalyzer,
    SizeEstimator,
    AvocadoAnalysis,
    Ripeness,
    SizeCategory,
)


@dataclass
class FrameResult:
    """Result of processing a single frame."""

    frame: np.ndarray
    detections: List[Detection]
    analyses: List[AvocadoAnalysis]
    count: int
    fps: float
    timestamp: float


class LivestreamProcessor:
    """
    Processes livestream video for avocado detection.

    Supports various video sources:
    - Webcam (integer device ID)
    - Video file path
    - RTSP stream URL
    - HTTP/HLS stream URL
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        process_every_n_frames: int = 1,
        on_frame_callback: Optional[Callable[[FrameResult], None]] = None,
        mode: str = "hybrid",  # 'yolo', 'segment', or 'hybrid'
    ):
        """
        Initialize the livestream processor.

        Args:
            source: Video source - webcam ID, file path, or stream URL.
            model_path: Path to YOLO model weights.
            confidence_threshold: Minimum detection confidence.
            process_every_n_frames: Process every Nth frame for performance.
            on_frame_callback: Callback function for each processed frame.
            mode: Detection mode - 'yolo', 'segment', or 'hybrid'.
        """
        self.source = source
        self.process_every_n_frames = process_every_n_frames
        self.on_frame_callback = on_frame_callback

        # Initialize components
        self.detector = AvocadoDetector(
            model_path=model_path, confidence_threshold=confidence_threshold, mode=mode
        )
        self.color_analyzer = ColorAnalyzer()
        self.size_estimator = SizeEstimator()

        # State
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.fps = 0.0
        self._thread: Optional[threading.Thread] = None
        self._last_frame_time = 0.0

    def _open_source(self) -> bool:
        """Open the video source."""
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"Error: Could not open video source: {self.source}")
            return False

        # Get frame dimensions and update size estimator
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size_estimator.update_frame_size(width, height)

        return True

    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        """Process a single frame."""
        # Calculate FPS
        current_time = time.time()
        if self._last_frame_time > 0:
            self.fps = 1.0 / (current_time - self._last_frame_time)
        self._last_frame_time = current_time

        # Detect avocados
        detections = self.detector.detect(frame)

        # Analyze each detection
        analyses = []
        for detection in detections:
            # Analyze color
            color, color_name, ripeness = self.color_analyzer.analyze(
                frame, detection.bbox
            )

            # Estimate size
            size_category, relative_size = self.size_estimator.estimate(detection.bbox)

            analyses.append(
                AvocadoAnalysis(
                    dominant_color=color,
                    dominant_color_name=color_name,
                    ripeness=ripeness,
                    size_category=size_category,
                    relative_size=relative_size,
                )
            )

        return FrameResult(
            frame=frame,
            detections=detections,
            analyses=analyses,
            count=len(detections),
            fps=self.fps,
            timestamp=current_time,
        )

    def process_single_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Process a single frame externally provided.

        Args:
            frame: BGR image as numpy array.

        Returns:
            FrameResult with detections and analyses.
        """
        # Update size estimator with frame dimensions
        h, w = frame.shape[:2]
        self.size_estimator.update_frame_size(w, h)

        return self._process_frame(frame)

    def run(self, show_window: bool = True) -> None:
        """
        Run the livestream processor.

        Args:
            show_window: Whether to show the visualization window.
        """
        if not self._open_source():
            return

        self.running = True
        self.frame_count = 0

        # Import visualizer here to avoid circular imports
        from .visualizer import Visualizer

        visualizer = Visualizer()

        # Create resizable window
        window_name = "Avocadet - Avocado Detection"
        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

            # Add trackbars for parameter adjustment
            # Confidence threshold (0-100, displayed as 0.0-1.0)
            cv2.createTrackbar(
                "Confidence %",
                window_name,
                int(self.detector.confidence_threshold * 100),
                100,
                lambda v: None,
            )

            # Min area for segmentation (100-5000)
            if self.detector.segmenter is not None:
                cv2.createTrackbar(
                    "Min Area",
                    window_name,
                    self.detector.segmenter.min_area,
                    5000,
                    lambda v: None,
                )

                # Max area for segmentation (5000-100000)
                cv2.createTrackbar(
                    "Max Area",
                    window_name,
                    min(self.detector.segmenter.max_area, 100000),
                    100000,
                    lambda v: None,
                )

                # Circularity threshold (0-100, displayed as 0.0-1.0)
                cv2.createTrackbar(
                    "Circular %",
                    window_name,
                    int(self.detector.segmenter.circularity_threshold * 100),
                    100,
                    lambda v: None,
                )

        print(f"Starting avocado detection on source: {self.source}")
        print(
            "Press 'q' to quit, 'p' to pause, 's' to save screenshot, 'f' to fullscreen"
        )

        try:
            while self.running:
                if self.paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord("p"):
                        self.paused = False
                    elif key == ord("q"):
                        break
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    # End of video or stream error
                    if isinstance(self.source, str) and not self.source.startswith(
                        ("rtsp://", "http://")
                    ):
                        # Video file ended, loop back
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Stream ended or error occurred")
                        break

                self.frame_count += 1

                # Read trackbar values and update parameters
                if show_window:
                    conf = cv2.getTrackbarPos("Confidence %", window_name) / 100.0
                    self.detector.confidence_threshold = max(0.01, conf)

                    if self.detector.segmenter is not None:
                        self.detector.segmenter.min_area = cv2.getTrackbarPos(
                            "Min Area", window_name
                        )
                        self.detector.segmenter.max_area = cv2.getTrackbarPos(
                            "Max Area", window_name
                        )
                        circ = cv2.getTrackbarPos("Circular %", window_name) / 100.0
                        self.detector.segmenter.circularity_threshold = circ

                # Process frame (skip some for performance if needed)
                if self.frame_count % self.process_every_n_frames == 0:
                    result = self._process_frame(frame)

                    # Call callback if provided
                    if self.on_frame_callback:
                        self.on_frame_callback(result)

                    # Visualize
                    if show_window:
                        annotated_frame = visualizer.draw(
                            result.frame,
                            result.detections,
                            result.analyses,
                            result.count,
                            result.fps,
                        )
                        cv2.imshow(window_name, annotated_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    self.paused = True
                elif key == ord("s"):
                    # Save screenshot
                    filename = f"avocadet_screenshot_{int(time.time())}.png"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Screenshot saved: {filename}")
                elif key == ord("f"):
                    # Toggle fullscreen
                    cv2.setWindowProperty(
                        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                    )

        finally:
            self.stop()

    def run_async(self, show_window: bool = True) -> None:
        """
        Run the livestream processor in a background thread.

        Args:
            show_window: Whether to show the visualization window.
        """
        self._thread = threading.Thread(
            target=self.run, args=(show_window,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the livestream processor."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        self.paused = not self.paused

    @property
    def is_running(self) -> bool:
        """Check if the processor is running."""
        return self.running

    @property
    def is_paused(self) -> bool:
        """Check if the processor is paused."""
        return self.paused
