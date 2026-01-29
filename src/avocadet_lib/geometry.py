import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


class GeometryManager:
    """
    Manages camera geometry, lens models (pinhole vs fisheye), and rectification.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lens_model = config.get("lens_model", "pinhole")
        self.rectify_enabled = config.get("rectify_enabled", False)
        self.output_geometry = config.get("output_geometry", "pinhole_rectified")

        # Intrinsics
        intrinsics = config.get("intrinsics", {})
        self.K = np.array(
            [
                [intrinsics.get("fx", 1000.0), 0, intrinsics.get("cx", 960.0)],
                [0, intrinsics.get("fy", 1000.0), intrinsics.get("cy", 540.0)],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Distortion
        dist_config = config.get("distortion", {})
        self.D = np.array(dist_config.get("coeffs", []), dtype=np.float32)

        # Caching maps
        self._map1 = None
        self._map2 = None
        self._image_size = None  # (w, h)

    def update_from_camera_info(self, camera_info) -> None:
        """Updates intrinsics/distortion from a ROS CameraInfo message."""
        self.K = np.array(camera_info.k).reshape(3, 3)
        self.D = np.array(camera_info.d)

        # Reset maps if geometry changes
        self._map1 = None
        self._map2 = None

    def rectify(self, image: np.ndarray) -> np.ndarray:
        """
        Rectifies the image if enabled and configured.
        """
        # Return as is if rectification is disabled or not applicable
        if not self.rectify_enabled:
            return image

        h, w = image.shape[:2]

        # Initialize maps if needed or if size changed
        if self._map1 is None or self._image_size != (w, h):
            self._image_size = (w, h)
            self._init_maps(w, h)

        if self._map1 is not None and self._map2 is not None:
            return cv2.remap(image, self._map1, self._map2, cv2.INTER_LINEAR)

        return image

    def _init_maps(self, w: int, h: int):
        """Initializes remap look-up tables."""
        if self.lens_model == "fisheye":
            # Estimate new camera matrix for undistortion
            # Simple balance=0 or 1 approach, or maintain K
            # For simplicity in this v1, we try to preserve K unless we want strictly 'valid' crop
            new_K = (
                cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.K, self.D, (w, h), np.eye(3), balance=1.0
                )
                if len(self.D) >= 4
                else self.K
            )

            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
        elif self.lens_model == "pinhole":
            new_K, roi = cv2.getOptimalNewCameraMatrix(
                self.K, self.D, (w, h), 1, (w, h)
            )
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
        else:
            # Unknown model, no-op
            self._map1 = None
            self._map2 = None
