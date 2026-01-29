from typing import List
import numpy as np
from .base import BaseBackend, Detection


class OnnxBackend(BaseBackend):
    def __init__(
        self, model_path: str, confidence_threshold: float = 0.5, device: str = "cpu"
    ):
        super().__init__(model_path, confidence_threshold, device)
        # TODO: Implement ONNX Runtime session loading
        print(
            f"Warning: OnnxBackend initialized with {model_path} but not implemented yet."
        )
        self.session = None

    def infer(self, frame: np.ndarray) -> List[Detection]:
        # TODO: Implement ONNX inference
        raise NotImplementedError("ONNX inference is not yet implemented.")
