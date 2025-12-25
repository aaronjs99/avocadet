"""
Color and size analysis for detected avocados.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import numpy as np
import cv2


class Ripeness(Enum):
    """Avocado ripeness levels based on color."""
    UNRIPE = "unripe"       # Bright green
    NEARLY_RIPE = "nearly_ripe"  # Dark green
    RIPE = "ripe"           # Dark brown/green
    OVERRIPE = "overripe"   # Very dark/black


class SizeCategory(Enum):
    """Relative size categories for avocados."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class AvocadoAnalysis:
    """Complete analysis of a detected avocado."""
    dominant_color: Tuple[int, int, int]  # BGR
    dominant_color_name: str
    ripeness: Ripeness
    size_category: SizeCategory
    relative_size: float  # 0.0 to 1.0


class ColorAnalyzer:
    """
    Analyzes color of detected avocados to determine ripeness.
    
    Uses HSV color space for more robust color analysis.
    """
    
    # HSV ranges for avocado colors (H: 0-180, S: 0-255, V: 0-255)
    COLOR_RANGES = {
        "bright_green": ((35, 50, 50), (85, 255, 255)),   # Unripe
        "dark_green": ((35, 30, 30), (85, 150, 150)),     # Nearly ripe
        "brown": ((10, 30, 30), (25, 200, 200)),          # Ripe
        "dark_brown": ((0, 20, 0), (20, 150, 100)),       # Overripe
    }
    
    def __init__(self):
        """Initialize the color analyzer."""
        pass
    
    def analyze(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int, int], str, Ripeness]:
        """
        Analyze the color of an avocado region.
        
        Args:
            image: Full BGR image.
            bbox: Bounding box (x1, y1, x2, y2).
            
        Returns:
            Tuple of (dominant_color_bgr, color_name, ripeness).
        """
        x1, y1, x2, y2 = bbox
        
        # Add padding to avoid edge artifacts
        h, w = image.shape[:2]
        pad = 5
        x1 = max(0, x1 + pad)
        y1 = max(0, y1 + pad)
        x2 = min(w, x2 - pad)
        y2 = min(h, y2 - pad)
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ((0, 128, 0), "green", Ripeness.UNRIPE)
        
        # Get dominant color
        dominant_color = self._get_dominant_color(roi)
        
        # Classify color and ripeness
        color_name, ripeness = self._classify_color(roi)
        
        return (dominant_color, color_name, ripeness)
    
    def _get_dominant_color(self, roi: np.ndarray) -> Tuple[int, int, int]:
        """Get the dominant color in a region using k-means."""
        # Reshape image to be a list of pixels
        pixels = roi.reshape(-1, 3).astype(np.float32)
        
        # Use k-means to find dominant color
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 3  # Number of clusters
        
        try:
            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Get the most common cluster
            _, counts = np.unique(labels, return_counts=True)
            dominant_idx = np.argmax(counts)
            dominant_color = centers[dominant_idx].astype(int)
            
            return tuple(dominant_color)
        except:
            # Fallback to mean color
            mean_color = np.mean(pixels, axis=0).astype(int)
            return tuple(mean_color)
    
    def _classify_color(self, roi: np.ndarray) -> Tuple[str, Ripeness]:
        """Classify the color of the ROI and determine ripeness."""
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate mean HSV values
        mean_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
        h, s, v = mean_hsv
        
        # Classify based on hue and saturation
        if h >= 35 and h <= 85:  # Green hues
            if s > 100 and v > 100:
                return ("bright_green", Ripeness.UNRIPE)
            else:
                return ("dark_green", Ripeness.NEARLY_RIPE)
        elif h >= 10 and h <= 35:  # Yellow-brown hues
            if v > 80:
                return ("brown", Ripeness.RIPE)
            else:
                return ("dark_brown", Ripeness.OVERRIPE)
        elif v < 60:  # Very dark
            return ("black", Ripeness.OVERRIPE)
        else:
            # Default to nearly ripe for ambiguous colors
            return ("green", Ripeness.NEARLY_RIPE)


class SizeEstimator:
    """
    Estimates relative size of detected avocados.
    
    Without camera calibration, provides relative sizing based on
    bounding box dimensions compared to frame size.
    """
    
    # Thresholds for size classification (as fraction of frame area)
    SIZE_THRESHOLDS = {
        "small": 0.01,   # < 1% of frame
        "medium": 0.03,  # 1-3% of frame
        "large": 0.03,   # > 3% of frame
    }
    
    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        """
        Initialize the size estimator.
        
        Args:
            frame_width: Width of the video frame.
            frame_height: Height of the video frame.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height
    
    def update_frame_size(self, width: int, height: int) -> None:
        """Update the frame dimensions."""
        self.frame_width = width
        self.frame_height = height
        self.frame_area = width * height
    
    def estimate(self, bbox: Tuple[int, int, int, int]) -> Tuple[SizeCategory, float]:
        """
        Estimate the size category of an avocado.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2).
            
        Returns:
            Tuple of (size_category, relative_size_0_to_1).
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Calculate relative size
        relative_size = area / self.frame_area if self.frame_area > 0 else 0
        
        # Normalize to 0-1 range (assuming max reasonable size is 10% of frame)
        normalized_size = min(1.0, relative_size / 0.1)
        
        # Classify
        if relative_size < self.SIZE_THRESHOLDS["small"]:
            return (SizeCategory.SMALL, normalized_size)
        elif relative_size < self.SIZE_THRESHOLDS["medium"]:
            return (SizeCategory.MEDIUM, normalized_size)
        else:
            return (SizeCategory.LARGE, normalized_size)
    
    def get_size_in_pixels(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get the width and height of a bounding box in pixels."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1, y2 - y1)
