# -*- coding: utf-8 -*-
"""
Unified Detector Module

This module implements the core detection functionality using pluggable backends.
It supports filtering detections based on a configurable allowlist.
"""

from typing import List, Optional
import numpy as np

from .backends.base import Detection
from .backends.ultralytics_backend import UltralyticsBackend
from .backends.onnx_backend import OnnxBackend
from .backends.tensorrt_backend import TensorRTBackend


class UnifiedDetector:
    """
    Unified detector supporting multiple inference backends for Flowers and Fruits.

    Attributes:
        confidence_threshold (float): Minimum confidence threshold for detections.
        device (str): Computation device ('cpu', 'cuda', 'auto').
        backend_type (str): Name of the backend to use.
        config (dict): Configuration dictionary.
        backend (Backend): Instantiated backend object.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: str = "auto",
        backend: str = "ultralytics",
        config: Optional[dict] = None,
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to model weights (required).
            confidence_threshold: Minimum confidence for detections.
            device: Device to run inference on ('cpu', 'cuda', or 'auto').
            backend: Inference backend ('ultralytics', 'onnx', 'tensorrt').
            config: Full configuration dictionary (optional).

        Raises:
            ValueError: If model_path is not provided or backend is unknown.
        """
        if not model_path:
            raise ValueError("model_path is mandatory for UnifiedDetector.")

        self.confidence_threshold = confidence_threshold
        self.device = device
        self.backend_type = backend
        self.config = config or {}
        self.backend = None

        self._load_backend(model_path)

    def _load_backend(self, model_path: str) -> None:
        """
        Load the specified inference backend.

        Args:
            model_path: Path to the model file.

        Raises:
            ValueError: If the backend type is not supported.
        """
        if self.backend_type == "ultralytics":
            self.backend = UltralyticsBackend(
                model_path, self.confidence_threshold, self.device
            )
        elif self.backend_type == "onnx":
            self.backend = OnnxBackend(
                model_path, self.confidence_threshold, self.device
            )
        elif self.backend_type == "tensorrt":
            self.backend = TensorRTBackend(
                model_path, self.confidence_threshold, self.device, config=self.config
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend_type}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame: Input BGR image as a numpy array.

        Returns:
            List[Detection]: A list of detection objects found in the frame.
        """
        if self.backend is None:
            return []

        # Run inference
        detections = self.backend.infer(frame)

        # Filter by class allowlist if specified in config
        detector_config = self.config.get("detector", {})
        allowlist = detector_config.get("class_allowlist", [])

        if not allowlist:
            return detections

        # Case-insensitive matching for allowlist
        allowlist_lower = [c.lower() for c in allowlist]
        filtered_detections = []

        for det in detections:
            class_name_lower = det.class_name.lower()

            # Check for exact match or substring inclusion
            # "avocado" matches "avocado fruit"
            # "fruit" matches "avocado fruit"
            is_allowed = False

            if class_name_lower in allowlist_lower:
                is_allowed = True
            else:
                for allowed in allowlist_lower:
                    if allowed in class_name_lower:
                        is_allowed = True
                        break

            if is_allowed:
                filtered_detections.append(det)

        return filtered_detections
