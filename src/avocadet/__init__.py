# -*- coding: utf-8 -*-
"""
Avocadet - Real-time Avocado Detection Package

A computer vision system for real-time avocado detection, counting,
and ripeness analysis from video streams.

Authors:
    Aaron John Sabu
    Sunwoong Choi
    Sriram Narasimhan

License:
    MIT License
"""

from .detector import AvocadoDetector
from .analyzer import ColorAnalyzer, SizeEstimator
from .stream import LivestreamProcessor
from .visualizer import Visualizer
from .segmenter import ColorBasedSegmenter, SAMSegmenter

__version__ = "1.0.0"
__author__ = "Aaron John Sabu, Sunwoong Choi, Sriram Narasimhan"
__license__ = "MIT"

__all__ = [
    "AvocadoDetector",
    "ColorAnalyzer",
    "SizeEstimator",
    "LivestreamProcessor",
    "Visualizer",
    "ColorBasedSegmenter",
    "SAMSegmenter",
]
