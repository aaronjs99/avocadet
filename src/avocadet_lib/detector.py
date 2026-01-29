# -*- coding: utf-8 -*-
"""
Unified Detector Module

This module implements the core detection functionality using pluggable backends.
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
        """Load the specified backend."""
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
        Detect flowers in a frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            List of Detection objects.
        """
        if self.backend is None:
            return []

        return self.backend.infer(frame)
