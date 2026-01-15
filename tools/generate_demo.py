#!/usr/bin/env python3
"""Generate demo video with detection overlays."""

import sys
import os

# Add src to path correctly (parent of tools directory)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import cv2
import imageio
from avocadet import AvocadoDetector, ColorAnalyzer, SizeEstimator, Visualizer
from avocadet.analyzer import AvocadoAnalysis

# Config
INPUT_VIDEO = os.path.join(ROOT_DIR, "data/slowed/2s.mp4")
OUTPUT_VIDEO = os.path.join(ROOT_DIR, "demo/demo.mp4")
OUTPUT_GIF = os.path.join(ROOT_DIR, "demo/demo.gif")
MODEL_PATH = os.path.join(ROOT_DIR, "models/avocado_yolov8.pt")

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}, falling back to yolov8n.pt")
    MODEL_PATH = os.path.join(ROOT_DIR, "models/yolov8n.pt")

# Initialize
print(f"Using model: {MODEL_PATH}")
detector = AvocadoDetector(model_path=MODEL_PATH, confidence_threshold=0.5, mode="yolo")
analyzer = ColorAnalyzer()
size_estimator = SizeEstimator()
visualizer = Visualizer()

# Open video
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Error opening video file {INPUT_VIDEO}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

size_estimator.update_frame_size(width, height)

# Use imageio to write video (better compatibility with ffmpeg)
print(f"Processing {INPUT_VIDEO} -> {OUTPUT_VIDEO}...")
writer = imageio.get_writer(OUTPUT_VIDEO, fps=fps, codec="libx264", quality=8)

frames_for_gif = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    detections = detector.detect(frame)

    # Analyze and visualize
    analyses = []
    for det in detections:
        color, color_name, ripeness = analyzer.analyze(frame, det.bbox)
        size_cat, rel_size = size_estimator.estimate(det.bbox)
        analyses.append(
            AvocadoAnalysis(
                dominant_color=color,
                dominant_color_name=color_name,
                ripeness=ripeness,
                size_category=size_cat,
                relative_size=rel_size,
            )
        )

    # Draw overlays
    frame = visualizer.draw(frame, detections, analyses, len(detections), fps)

    # Convert BGR to RGB for imageio
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Write to video
    writer.append_data(rgb_frame)

    # Store frames for GIF generation (subsample to reduce size)
    if frame_count % 3 == 0:  # Take every 3rd frame
        # Resize for GIF to keep size down
        h, w = frame.shape[:2]
        new_w = 480
        new_h = int(h * (new_w / w))
        small_frame = cv2.resize(rgb_frame, (new_w, new_h))
        frames_for_gif.append(small_frame)

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
writer.close()
print(f"Video saved to {OUTPUT_VIDEO}")

# Generate GIF
print(f"Generating GIF from {len(frames_for_gif)} frames...")
imageio.mimsave(OUTPUT_GIF, frames_for_gif, fps=10, loop=0)
print(f"GIF saved to {OUTPUT_GIF}")

print("Done!")
