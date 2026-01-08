# -*- coding: utf-8 -*-
"""
Segmentation Module

This module provides color-based segmentation and optional SAM
(Segment Anything Model) integration for avocado detection.

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
import cv2

from .detector import Detection


@dataclass 
class SegmentedObject:
    """Represents a segmented object."""
    mask: np.ndarray  # Binary mask
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: int
    confidence: float = 1.0
    

class ColorBasedSegmenter:
    """
    Color-based segmentation for avocados.
    
    Tuned to detect actual avocados (darker green/brown, oval shape)
    and filter out leaves (bright green, elongated) and bark/ground.
    """
    
    # HSV ranges for avocado colors - DARKER green, not bright leaves
    # Avocados: darker green to brownish-green
    HSV_LOWER_AVOCADO = np.array([25, 40, 40])    # Dark yellow-green
    HSV_UPPER_AVOCADO = np.array([85, 200, 180])  # Darker greens (exclude bright)
    
    def __init__(
        self,
        min_area: int = 800,          # Larger min to filter small leaf fragments
        max_area: int = 30000,         # Reasonable max for avocados
        circularity_threshold: float = 0.45,  # Higher - avocados are rounder than leaves
        min_aspect_ratio: float = 0.5,   # Avocados are roughly oval
        max_aspect_ratio: float = 2.0,   # Not too elongated
        convexity_threshold: float = 0.7  # Avocados are fairly convex
    ):
        """
        Initialize the segmenter.
        
        Args:
            min_area: Minimum contour area to consider.
            max_area: Maximum contour area to consider.
            circularity_threshold: Minimum circularity (0-1) for avocado-like shapes.
            min_aspect_ratio: Minimum height/width ratio.
            max_aspect_ratio: Maximum height/width ratio.
            convexity_threshold: Minimum convexity (area/convex_hull_area).
        """
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_threshold = circularity_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.convexity_threshold = convexity_threshold
    
    def segment(self, frame: np.ndarray) -> List[SegmentedObject]:
        """
        Segment avocados using color-based detection.
        
        Args:
            frame: BGR image.
            
        Returns:
            List of SegmentedObject instances.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for avocado-like colors (darker greens)
        mask = cv2.inRange(hsv, self.HSV_LOWER_AVOCADO, self.HSV_UPPER_AVOCADO)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        segments = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Filter by circularity (avocados are rounder than leaves)
            if circularity < self.circularity_threshold:
                continue
            
            # Get bounding box and check aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0:
                continue
            aspect_ratio = h / w
            
            # Filter by aspect ratio (avocados are oval, not elongated like leaves)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Check convexity (avocados are fairly convex, jagged leaves are not)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                convexity = area / hull_area
                if convexity < self.convexity_threshold:
                    continue
            
            # Create binary mask for this contour
            contour_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            
            # Calculate confidence based on how "avocado-like" the detection is
            confidence = (circularity / 0.8) * (convexity if hull_area > 0 else 0.8)
            confidence = min(1.0, confidence)
            
            segments.append(SegmentedObject(
                mask=contour_mask,
                bbox=(x, y, x + w, y + h),
                area=area,
                confidence=confidence
            ))
        
        return segments
    
    def to_detections(self, segments: List[SegmentedObject]) -> List[Detection]:
        """Convert segments to Detection objects."""
        return [
            Detection(
                bbox=seg.bbox,
                confidence=seg.confidence,
                class_name="avocado"
            )
            for seg in segments
        ]


class SAMSegmenter:
    """
    SAM (Segment Anything Model) based segmentation.
    
    Provides more accurate segmentation but requires more compute.
    Falls back to color-based segmentation if SAM is not available.
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize SAM segmenter.
        
        Args:
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b').
            checkpoint_path: Path to SAM checkpoint.
            device: Device to run on.
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.sam = None
        self.mask_generator = None
        self._fallback = ColorBasedSegmenter()
        self._load_model()
    
    def _load_model(self) -> None:
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            import torch
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if self.checkpoint_path is None:
                print("SAM checkpoint not provided, using color-based fallback")
                return
            
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                self.sam,
                points_per_side=16,  # Reduce for speed
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=500,
            )
            print(f"SAM loaded on {self.device}")
            
        except ImportError:
            print("SAM not installed, using color-based fallback")
        except Exception as e:
            print(f"Failed to load SAM: {e}, using color-based fallback")
    
    def segment(self, frame: np.ndarray) -> List[SegmentedObject]:
        """
        Segment objects in frame using SAM.
        
        Args:
            frame: BGR image.
            
        Returns:
            List of SegmentedObject instances.
        """
        if self.mask_generator is None:
            return self._fallback.segment(frame)
        
        # Convert BGR to RGB for SAM
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        masks = self.mask_generator.generate(rgb_frame)
        
        segments = []
        for mask_data in masks:
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            bbox = mask_data['bbox']  # x, y, w, h format
            x, y, w, h = [int(v) for v in bbox]
            
            # Filter by aspect ratio (avocados are roughly 1:1.5)
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                continue
            
            segments.append(SegmentedObject(
                mask=mask,
                bbox=(x, y, x + w, y + h),
                area=int(mask_data['area']),
                confidence=mask_data['predicted_iou']
            ))
        
        return segments
    
    def to_detections(self, segments: List[SegmentedObject]) -> List[Detection]:
        """Convert segments to Detection objects."""
        return [
            Detection(
                bbox=seg.bbox,
                confidence=seg.confidence,
                class_name="avocado"
            )
            for seg in segments
        ]
