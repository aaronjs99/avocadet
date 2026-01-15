#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download avocado dataset from Roboflow Universe for training.

Usage:
    python tools/download_dataset.py

Requires:
    export ROBOFLOW_API_KEY='your-key-here'
"""

import os
import sys

from roboflow import Roboflow


def main():
    # Get API key from environment
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        print(
            """
ERROR: ROBOFLOW_API_KEY environment variable not set.

To set it:
    export ROBOFLOW_API_KEY='your-key-here'

Or add to ~/.bashrc:
    echo "export ROBOFLOW_API_KEY='your-key-here'" >> ~/.bashrc
    source ~/.bashrc
"""
        )
        sys.exit(1)

    rf = Roboflow(api_key=api_key)

    print(
        """
To download an avocado dataset:

1. Go to: https://universe.roboflow.com/search?q=avocado
2. Pick a dataset (e.g., "Avocado Detection" or "Avocado Ripeness")
3. Click "Download Dataset" → "YOLOv8"
4. Copy the Python code snippet and modify it to use the environment variable

Example usage after finding a dataset:
    from roboflow import Roboflow
    import os
    
    rf = Roboflow(api_key=os.environ.get('ROBOFLOW_API_KEY'))
    project = rf.workspace("workspace-name").project("project-name")
    version = project.version(1)
    dataset = version.download("yolov8", location="datasets/avocado_roboflow")
"""
    )


if __name__ == "__main__":
    main()
