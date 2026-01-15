# -*- coding: utf-8 -*-
"""
Avocado Detector Module

This module implements the core detection functionality using YOLOv8
object detection and/or color-based segmentation.

Authors:
    Aaron John Sabu
    Sunwoong Choi
    Sriram Narasimhan

License:
    MIT License
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    """Represents a single avocado detection."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str = "avocado"

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        return self.width * self.height


class AvocadoDetector:
    """
    Avocado detector using YOLOv8 or color-based segmentation.

    Supports multiple detection modes:
    - 'yolo': Use YOLOv8 object detection (requires trained model)
    - 'segment': Use color-based segmentation (works for green avocados)
    - 'hybrid': Use both and combine results
    """

    # COCO class names that might represent avocados in a general detector
    FRUIT_CLASSES = {"apple", "orange", "banana", "sports ball"}

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "auto",
        mode: str = "hybrid",  # 'yolo', 'segment', or 'hybrid'
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLO model weights. If None, uses YOLOv8n.
            confidence_threshold: Minimum confidence for detections.
            device: Device to run inference on ('cpu', 'cuda', or 'auto').
            mode: Detection mode - 'yolo', 'segment', or 'hybrid'.
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.mode = mode
        self.model = None
        self.segmenter = None
        self.model_path = model_path or "yolov8n.pt"

        if mode in ("yolo", "hybrid"):
            self._load_model()
        if mode in ("segment", "hybrid"):
            self._load_segmenter()

    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            if self.device != "auto":
                self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def _load_segmenter(self) -> None:
        """Load the color-based segmenter."""
        from .segmenter import ColorBasedSegmenter

        self.segmenter = ColorBasedSegmenter()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect avocados in a frame.

        Args:
            frame: BGR image as numpy array (OpenCV format).

        Returns:
            List of Detection objects.
        """
        detections = []

        # Use segmentation mode
        if self.mode in ("segment", "hybrid") and self.segmenter is not None:
            segments = self.segmenter.segment(frame)
            detections.extend(self.segmenter.to_detections(segments))

        # Use YOLO mode
        if self.mode in ("yolo", "hybrid") and self.model is not None:
            yolo_detections = self._detect_yolo(frame)
            if self.mode == "yolo":
                detections = yolo_detections
            else:
                # Hybrid: merge results, avoiding duplicates
                detections = self._merge_detections(detections, yolo_detections)

        return detections

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Detect using YOLO model."""
        if self.model is None:
            return []

        # Run inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                # Get bounding box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.model.names[cls_id]

                # For general YOLO model, filter for fruit-like objects
                # With a custom avocado model, all detections would be avocados
                if cls_name in self.FRUIT_CLASSES or "avocado" in cls_name.lower():
                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2), confidence=conf, class_name="avocado"
                        )
                    )
                # Also accept any detection with high confidence for demo purposes
                elif conf > 0.7:
                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2), confidence=conf, class_name=cls_name
                        )
                    )

        return detections

    def _merge_detections(
        self,
        seg_detections: List[Detection],
        yolo_detections: List[Detection],
        iou_threshold: float = 0.5,
    ) -> List[Detection]:
        """Merge detections from segmentation and YOLO, removing duplicates."""
        if not seg_detections:
            return yolo_detections
        if not yolo_detections:
            return seg_detections

        merged = list(seg_detections)

        for yolo_det in yolo_detections:
            is_duplicate = False
            for seg_det in seg_detections:
                iou = self._calculate_iou(yolo_det.bbox, seg_det.bbox)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                merged.append(yolo_det)

        return merged

    def _calculate_iou(
        self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect avocados in multiple frames.

        Args:
            frames: List of BGR images.

        Returns:
            List of detection lists, one per frame.
        """
        return [self.detect(frame) for frame in frames]
