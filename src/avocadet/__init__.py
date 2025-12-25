"""
Avocadet - Real-time avocado detection from livestream video feeds.
"""

from .detector import AvocadoDetector
from .analyzer import ColorAnalyzer, SizeEstimator
from .stream import LivestreamProcessor
from .visualizer import Visualizer
from .segmenter import ColorBasedSegmenter, SAMSegmenter

__version__ = "0.1.0"
__all__ = [
    "AvocadoDetector",
    "ColorAnalyzer", 
    "SizeEstimator",
    "LivestreamProcessor",
    "Visualizer",
    "ColorBasedSegmenter",
    "SAMSegmenter",
]
