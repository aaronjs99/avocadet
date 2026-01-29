from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Detection:
    """Represents a single object detection."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str

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


class BaseBackend(ABC):
    """Abstract base class for detection backends."""

    def __init__(self, model_path: str, confidence_threshold: float, device: str):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on the given frame."""
        pass
