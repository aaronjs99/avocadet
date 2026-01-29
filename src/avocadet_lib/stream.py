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

from .detector import UnifiedDetector
from .backends.base import Detection
from .analyzer import (
    ColorAnalyzer,
    SizeEstimator,
    AnalysisAttributes,
    Ripeness,
    SizeCategory,
)


@dataclass
class FrameResult:
    """Result of processing a single frame."""

    frame: np.ndarray
    detections: List[Detection]
    analyses: List[AnalysisAttributes]
    count: int
    fps: float
    timestamp: float


class ThreadedVideoCapture:
    """
    Threaded video capture to ensure we always get the latest frame.
    Prevents buffering delay when processing is slower than capture.
    """

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.ret = False
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.latest_frame = frame
                self.ret = ret
            if not ret:
                # Keep trying or stop? For files, we might want to stop or loop.
                # For now, let the main loop handle 'not ret' by checking isOpened
                # But if read fails (end of file), we should probably stop or flag it.
                if isinstance(self.cap, cv2.VideoCapture) and not self.cap.isOpened():
                    self.running = False

            # small sleep to prevent CPU hogging if capture is fast,
            # but we want low latency so keep it minimal or 0.
            # time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.latest_frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.cap.isOpened()


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
        confidence_threshold: Optional[float] = None,
        process_every_n_frames: int = 1,
        on_frame_callback: Optional[Callable[[FrameResult], None]] = None,
        backend: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        config: Optional[dict] = None,
    ):
        """
        Initialize the livestream processor.

        Args:
            source: Video source - webcam ID, file path, or stream URL.
            model_path: Path to YOLO model weights.
            confidence_threshold: Minimum detection confidence.
            process_every_n_frames: Process every Nth frame for performance.
            on_frame_callback: Callback function for each processed frame.
            backend: Inference backend ('ultralytics', 'onnx', 'tensorrt').
            config: Full configuration dictionary.
        """
        self.source = source
        self.process_every_n_frames = process_every_n_frames
        self.on_frame_callback = on_frame_callback
        self.width = width
        self.height = height
        self.config = config or {}

        # Resolve config logic: Argument > Config > Default
        det_config = self.config.get("detector", {})
        
        final_model_path = model_path or det_config.get("model_path")
        final_conf = confidence_threshold if confidence_threshold is not None else det_config.get("confidence_threshold", 0.5)
        final_backend = backend or det_config.get("backend", "ultralytics")
        
        # Initialize components
        from .geometry import GeometryManager
        self.geometry_manager = GeometryManager(self.config.get("geometry", {}))
        
        self.detector = UnifiedDetector(
            model_path=final_model_path,
            confidence_threshold=final_conf,
            backend=final_backend,
            config=self.config
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
        # Use threaded capture for webcams/streams to reduce latency
        # For video files, synchronous might be better to avoid skipping frames if analysis is desired on all
        # But user wants low delay, so let's default to threaded for all for consistency,
        # or check if source is int (webcam) or rtsp str.

        use_threaded = False
        if isinstance(self.source, int):  # Webcam
            use_threaded = True
        elif isinstance(self.source, str) and (
            self.source.startswith("rtsp") or self.source.startswith("http")
        ):
            use_threaded = True

        if use_threaded:
            # Set resolution before starting thread if possible, or support it in ThreadedVideoCapture
            # Set resolution before starting thread
            threaded_cap = ThreadedVideoCapture(self.source)
            threaded_cap.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            threaded_cap.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap = threaded_cap.start()
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

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

        # Rectify if enabled
        # Note: GeometryManager handles checking is_enabled internally for performant no-op
        # but we can also check here.
        # However, we must ensure we are using the geometry manager we init'd.
        frame = self.geometry_manager.rectify(frame)

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
                AnalysisAttributes(
                    dominant_color=color,
                    dominant_color_name=color_name,
                    size_category=size_category,
                    relative_size=relative_size,
                    ripeness=ripeness,
                    # Defaults for attributes not yet estimated
                    sex="unknown",
                    quality=0.5,
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
                    # Threaded capture might return False if it hasn't captured first frame yet
                    # or if video ended.
                    if isinstance(self.cap, ThreadedVideoCapture):
                        if self.cap.isOpened():
                            # Just wait a bit and retry
                            time.sleep(0.01)
                            continue

                    # End of video or stream error
                    if isinstance(self.source, str) and not self.source.startswith(
                        ("rtsp://", "http://")
                    ):
                        # Video file ended, loop back (only works for non-threaded standard capture easily)
                        # For now, just stop or break.
                        if not isinstance(self.cap, ThreadedVideoCapture):
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            print("Video file ended")
                            break
                    else:
                        print("Stream ended or error occurred")
                        break

                if isinstance(self.cap, ThreadedVideoCapture) and frame is None:
                    continue

                self.frame_count += 1

                # Read trackbar values and update parameters
                if show_window:
                    conf = cv2.getTrackbarPos("Confidence %", window_name) / 100.0
                    self.detector.confidence_threshold = max(0.01, conf)

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
                    filename = f"data/avocadet_screenshot_{int(time.time())}.png"
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
