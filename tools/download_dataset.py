#!/usr/bin/env python3
"""
Download avocado dataset from Roboflow Universe for training.
"""

from roboflow import Roboflow

# Initialize with API key
rf = Roboflow(api_key="RCBS9Z3Ao7J1P73zBanz")

# To download a public dataset, you need to:
# 1. Go to https://universe.roboflow.com
# 2. Search for "avocado"
# 3. Find a dataset you like (e.g., "avocado-ripeness")
# 4. Click "Download" and select "YOLOv8"
# 5. Copy the code snippet it gives you

# Example snippet from Roboflow Universe:
# (Replace with the actual snippet from the website)

print("""
To download an avocado dataset:

1. Go to: https://universe.roboflow.com/search?q=avocado
2. Pick a dataset (e.g., "Avocado Detection" or "Avocado Ripeness")
3. Click "Download Dataset" → "YOLOv8"
4. Copy the Python code snippet
5. Run it here!

Example:
    from roboflow import Roboflow
    rf = Roboflow(api_key="RCBS9Z3Ao7J1P73zBanz")
    project = rf.workspace("workspace-name").project("project-name")
    version = project.version(1)
    dataset = version.download("yolov8")
""")
