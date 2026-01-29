from typing import List
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .base import BaseBackend, Detection


class UltralyticsBackend(BaseBackend):
    def __init__(
        self, model_path: str, confidence_threshold: float = 0.5, device: str = "auto"
    ):
        super().__init__(model_path, confidence_threshold, device)
        if YOLO is None:
            raise ImportError("ultralytics package is required for UltralyticsBackend")

        try:
            self.model = YOLO(model_path)
            if device != "auto":
                self.model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.model.names[cls_id]

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2), confidence=conf, class_name=cls_name
                    )
                )
        return detections
