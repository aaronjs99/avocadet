# -*- coding: utf-8 -*-
"""
Visualization Module

This module provides visualization utilities for rendering detection
results, bounding boxes, and statistics overlays.

Authors:
    Aaron John Sabu
    Sunwoong Choi
    Sriram Narasimhan

License:
    MIT License
"""

from typing import List, Tuple
import cv2
import numpy as np

from .detector import Detection
from .analyzer import AvocadoAnalysis, Ripeness, SizeCategory


class Visualizer:
    """
    Draws detection results and analysis overlays on frames.
    """
    
    # Colors for different ripeness levels (BGR format)
    RIPENESS_COLORS = {
        Ripeness.UNRIPE: (0, 255, 0),       # Bright green
        Ripeness.NEARLY_RIPE: (0, 180, 0),  # Dark green
        Ripeness.RIPE: (0, 140, 70),        # Brown-green
        Ripeness.OVERRIPE: (30, 50, 100),   # Dark brown
    }
    
    # Size category display text
    SIZE_TEXT = {
        SizeCategory.SMALL: "S",
        SizeCategory.MEDIUM: "M",
        SizeCategory.LARGE: "L",
    }
    
    def __init__(
        self,
        font_scale: float = 0.5,
        line_thickness: int = 2,
        show_confidence: bool = True,
        show_size: bool = True,
        show_ripeness: bool = True,
        show_stats_panel: bool = True
    ):
        """
        Initialize the visualizer.
        
        Args:
            font_scale: Font scale for text.
            line_thickness: Line thickness for bounding boxes.
            show_confidence: Show confidence scores.
            show_size: Show size classifications.
            show_ripeness: Show ripeness classifications.
            show_stats_panel: Show statistics panel.
        """
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.show_confidence = show_confidence
        self.show_size = show_size
        self.show_ripeness = show_ripeness
        self.show_stats_panel = show_stats_panel
    
    def draw(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        analyses: List[AvocadoAnalysis],
        count: int,
        fps: float
    ) -> np.ndarray:
        """
        Draw all overlays on a frame.
        
        Args:
            frame: Original BGR frame.
            detections: List of detections.
            analyses: List of analyses (same order as detections).
            count: Total count of avocados.
            fps: Current FPS.
            
        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        
        # Draw each detection
        for i, (detection, analysis) in enumerate(zip(detections, analyses)):
            self._draw_detection(annotated, detection, analysis, i + 1)
        
        # Draw stats panel
        if self.show_stats_panel:
            self._draw_stats_panel(annotated, count, fps, analyses)
        
        return annotated
    
    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection,
        analysis: AvocadoAnalysis,
        index: int
    ) -> None:
        """Draw a single detection with its analysis."""
        x1, y1, x2, y2 = detection.bbox
        
        # Get color based on ripeness
        color = self.RIPENESS_COLORS.get(analysis.ripeness, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
        
        # Build label text
        label_parts = [f"#{index}"]
        
        if self.show_ripeness:
            label_parts.append(analysis.ripeness.value)
        
        if self.show_size:
            label_parts.append(self.SIZE_TEXT[analysis.size_category])
        
        if self.show_confidence:
            label_parts.append(f"{detection.confidence:.0%}")
        
        label = " | ".join(label_parts)
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, 1
        )
        
        # Position label above bounding box
        label_y = max(y1 - 5, text_height + 5)
        
        cv2.rectangle(
            frame,
            (x1, label_y - text_height - 5),
            (x1 + text_width + 10, label_y + 5),
            color,
            -1  # Filled
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 5, label_y),
            self.font,
            self.font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Draw color swatch showing dominant color
        swatch_size = 15
        swatch_x = x2 - swatch_size - 5
        swatch_y = y1 + 5
        # Convert numpy ints to native Python ints for OpenCV
        swatch_color = tuple(int(c) for c in analysis.dominant_color)
        cv2.rectangle(
            frame,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size, swatch_y + swatch_size),
            swatch_color,
            -1
        )
        cv2.rectangle(
            frame,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size, swatch_y + swatch_size),
            (255, 255, 255),
            1
        )
    
    def _draw_stats_panel(
        self,
        frame: np.ndarray,
        count: int,
        fps: float,
        analyses: List[AvocadoAnalysis]
    ) -> None:
        """Draw the statistics panel."""
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_width = 200
        panel_height = 140
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Draw semi-transparent panel background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw panel border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (100, 100, 100),
            1
        )
        
        # Draw title
        cv2.putText(
            frame,
            "AVOCADET STATS",
            (panel_x + 10, panel_y + 25),
            self.font,
            0.5,
            (0, 255, 200),
            1,
            cv2.LINE_AA
        )
        
        # Draw separator
        cv2.line(
            frame,
            (panel_x + 10, panel_y + 35),
            (panel_x + panel_width - 10, panel_y + 35),
            (100, 100, 100),
            1
        )
        
        # Draw stats
        stats = [
            f"Count: {count}",
            f"FPS: {fps:.1f}",
        ]
        
        # Count ripeness distribution
        if analyses:
            ripeness_counts = {}
            for a in analyses:
                r = a.ripeness.value
                ripeness_counts[r] = ripeness_counts.get(r, 0) + 1
            
            for ripeness, cnt in ripeness_counts.items():
                stats.append(f"  {ripeness}: {cnt}")
        
        y_offset = 55
        for stat in stats:
            cv2.putText(
                frame,
                stat,
                (panel_x + 10, panel_y + y_offset),
                self.font,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_offset += 18
    
    def draw_simple(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        count: int
    ) -> np.ndarray:
        """
        Draw simple bounding boxes without analysis.
        
        Args:
            frame: Original BGR frame.
            detections: List of detections.
            count: Total count.
            
        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"#{i+1} ({detection.confidence:.0%})"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                self.font,
                self.font_scale,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
        
        # Draw count
        cv2.putText(
            annotated,
            f"Count: {count}",
            (10, 30),
            self.font,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        return annotated
