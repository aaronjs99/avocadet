#!/usr/bin/env python3
"""
Frame Extraction and Annotation Tool for Creating Avocado Training Data

This script:
1. Extracts frames from videos
2. Opens a simple annotation interface to label avocados
3. Saves annotations in YOLO format for training
"""

import os
import cv2
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class BoundingBox:
    """A bounding box annotation."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int = 0  # 0 = avocado
    class_name: str = "avocado"


@dataclass  
class FrameAnnotation:
    """Annotations for a single frame."""
    image_path: str
    width: int
    height: int
    boxes: List[BoundingBox] = field(default_factory=list)


class AnnotationTool:
    """Simple annotation tool for labeling avocados in frames."""
    
    CLASSES = ["avocado"]  # Add more classes if needed
    
    def __init__(self, output_dir: str = "datasets/avocado_custom"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images" / "train"
        self.labels_dir = self.output_dir / "labels" / "train"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self.current_boxes: List[BoundingBox] = []
        self.drawing = False
        self.start_point = None
        self.current_frame = None
        self.display_frame = None
        self.frame_count = 0
    
    def extract_frames(self, video_path: str, every_n_frames: int = 30):
        """Extract frames from a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Extracting frames from: {video_path}")
        print(f"  Total frames: {total_frames}, FPS: {fps:.1f}")
        print(f"  Extracting every {every_n_frames} frames...")
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % every_n_frames == 0:
                frames.append((frame_idx, frame))
            
            frame_idx += 1
        
        cap.release()
        print(f"  Extracted {len(frames)} frames")
        return frames
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Draw preview rectangle
            self.display_frame = self.current_frame.copy()
            self._draw_boxes(self.display_frame)
            cv2.rectangle(self.display_frame, self.start_point, (x, y), (0, 255, 0), 2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                # Ensure proper order
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Only add if box is reasonably sized
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.current_boxes.append(BoundingBox(x1, y1, x2, y2))
                    print(f"  Added box: ({x1}, {y1}) -> ({x2}, {y2})")
            
            self.start_point = None
            self._update_display()
    
    def _draw_boxes(self, frame):
        """Draw all current bounding boxes on frame."""
        for i, box in enumerate(self.current_boxes):
            cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
            cv2.putText(frame, f"#{i+1} avocado", (box.x1, box.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _update_display(self):
        """Update the display frame with current boxes."""
        self.display_frame = self.current_frame.copy()
        self._draw_boxes(self.display_frame)
        
        # Draw instructions
        h = self.current_frame.shape[0]
        instructions = [
            "CONTROLS:",
            "  Click & drag: Draw box",
            "  'z': Undo last box",
            "  'c': Clear all boxes", 
            "  'n'/'SPACE': Next frame (save)",
            "  's': Skip (no save)",
            "  'q': Quit",
            f"  Boxes: {len(self.current_boxes)}"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(self.display_frame, text, (10, h - 150 + i*18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _save_annotation(self, frame, frame_idx: int):
        """Save frame and YOLO format annotation."""
        if not self.current_boxes:
            return
        
        h, w = frame.shape[:2]
        
        # Save image
        img_name = f"frame_{self.frame_count:04d}.jpg"
        img_path = self.images_dir / img_name
        cv2.imwrite(str(img_path), frame)
        
        # Save YOLO format labels
        label_name = f"frame_{self.frame_count:04d}.txt"
        label_path = self.labels_dir / label_name
        
        with open(label_path, 'w') as f:
            for box in self.current_boxes:
                # Convert to YOLO format (normalized center x, y, width, height)
                cx = ((box.x1 + box.x2) / 2) / w
                cy = ((box.y1 + box.y2) / 2) / h
                bw = (box.x2 - box.x1) / w
                bh = (box.y2 - box.y1) / h
                f.write(f"{box.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        
        self.frame_count += 1
        print(f"  Saved: {img_name} with {len(self.current_boxes)} boxes")
    
    def annotate_frames(self, frames: List[Tuple[int, any]]):
        """Open annotation interface for frames."""
        if not frames:
            print("No frames to annotate")
            return
        
        window_name = "Avocado Annotation Tool"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        print("\n" + "="*50)
        print("ANNOTATION TOOL")
        print("="*50)
        print("Draw boxes around avocados by clicking and dragging")
        print("="*50 + "\n")
        
        idx = 0
        while idx < len(frames):
            frame_idx, frame = frames[idx]
            self.current_frame = frame.copy()
            self.current_boxes = []
            self._update_display()
            
            print(f"\nFrame {idx+1}/{len(frames)} (video frame #{frame_idx})")
            
            while True:
                cv2.imshow(window_name, self.display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting annotation...")
                    cv2.destroyAllWindows()
                    return
                
                elif key == ord('z'):
                    # Undo last box
                    if self.current_boxes:
                        self.current_boxes.pop()
                        print("  Undid last box")
                        self._update_display()
                
                elif key == ord('c'):
                    # Clear all boxes
                    self.current_boxes = []
                    print("  Cleared all boxes")
                    self._update_display()
                
                elif key in [ord('n'), ord(' ')]:
                    # Next frame (save current)
                    self._save_annotation(frame, frame_idx)
                    idx += 1
                    break
                
                elif key == ord('s'):
                    # Skip frame (no save)
                    print("  Skipped frame")
                    idx += 1
                    break
        
        cv2.destroyAllWindows()
        self._create_data_yaml()
        print(f"\n✅ Annotation complete! {self.frame_count} frames saved.")
    
    def _create_data_yaml(self):
        """Create data.yaml for YOLO training."""
        yaml_content = f"""# Avocado Custom Dataset
path: {self.output_dir.absolute()}
train: images/train
val: images/train  # Using same for now, split later if needed

names:
  0: avocado
"""
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"Created: {yaml_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and annotate frames for training")
    parser.add_argument("--video", "-v", required=True, help="Video file to extract frames from")
    parser.add_argument("--every", "-e", type=int, default=30, 
                       help="Extract every Nth frame (default: 30)")
    parser.add_argument("--output", "-o", default="datasets/avocado_custom",
                       help="Output directory for dataset")
    
    args = parser.parse_args()
    
    tool = AnnotationTool(output_dir=args.output)
    
    # Extract frames
    frames = tool.extract_frames(args.video, every_n_frames=args.every)
    
    if frames:
        # Start annotation
        tool.annotate_frames(frames)
        
        print("\n" + "="*50)
        print("NEXT STEPS")
        print("="*50)
        print(f"1. Your dataset is in: {args.output}/")
        print(f"2. Train the model:")
        print(f"   python train.py --skip-download --dataset {args.output} --epochs 50")
        print("="*50)


if __name__ == "__main__":
    main()
