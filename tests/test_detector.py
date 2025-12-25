"""
Unit tests for avocadet package.
"""

import numpy as np
import pytest


class TestDetection:
    """Tests for the Detection dataclass."""
    
    def test_detection_properties(self):
        """Test Detection dataclass properties."""
        from avocadet.detector import Detection
        
        det = Detection(
            bbox=(100, 100, 200, 250),
            confidence=0.85
        )
        
        assert det.center == (150, 175)
        assert det.width == 100
        assert det.height == 150
        assert det.area == 15000
    

class TestColorAnalyzer:
    """Tests for the ColorAnalyzer."""
    
    def test_ripeness_classification(self):
        """Test that ripeness classification works."""
        from avocadet.analyzer import ColorAnalyzer, Ripeness
        
        analyzer = ColorAnalyzer()
        
        # Create a green image
        green_image = np.zeros((100, 100, 3), dtype=np.uint8)
        green_image[:, :] = (0, 200, 0)  # BGR green
        
        color, name, ripeness = analyzer.analyze(green_image, (10, 10, 90, 90))
        
        # Should detect as unripe (green)
        assert ripeness in [Ripeness.UNRIPE, Ripeness.NEARLY_RIPE]


class TestSizeEstimator:
    """Tests for the SizeEstimator."""
    
    def test_size_categories(self):
        """Test size category classification."""
        from avocadet.analyzer import SizeEstimator, SizeCategory
        
        estimator = SizeEstimator(frame_width=1920, frame_height=1080)
        
        # Small detection (tiny fraction of frame)
        small_bbox = (0, 0, 50, 50)
        category, _ = estimator.estimate(small_bbox)
        assert category == SizeCategory.SMALL
        
        # Large detection (significant portion of frame)
        large_bbox = (0, 0, 400, 400)
        category, _ = estimator.estimate(large_bbox)
        assert category == SizeCategory.LARGE
    
    def test_frame_size_update(self):
        """Test updating frame dimensions."""
        from avocadet.analyzer import SizeEstimator
        
        estimator = SizeEstimator()
        estimator.update_frame_size(1280, 720)
        
        assert estimator.frame_width == 1280
        assert estimator.frame_height == 720
        assert estimator.frame_area == 1280 * 720


class TestVisualizer:
    """Tests for the Visualizer."""
    
    def test_draw_simple(self):
        """Test simple visualization drawing."""
        from avocadet.visualizer import Visualizer
        from avocadet.detector import Detection
        
        visualizer = Visualizer()
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.9),
        ]
        
        result = visualizer.draw_simple(frame, detections, count=1)
        
        # Result should be same shape
        assert result.shape == frame.shape
        # Result should be modified (not all zeros)
        assert not np.array_equal(result, frame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
