import pytest
import numpy as np
from avocadet_lib.geometry import GeometryManager


class MockCameraInfo:
    def __init__(self):
        self.k = [1000.0, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0]
        self.d = [0.0, 0.0, 0.0, 0.0, 0.0]


def test_geometry_manager_init():
    config = {
        "lens_model": "pinhole",
        "rectify_enabled": False,
        "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240},
    }
    gm = GeometryManager(config)
    assert gm.lens_model == "pinhole"
    assert gm.rectify_enabled == False
    assert gm.K[0, 0] == 500


def test_update_from_camera_info():
    config = {"lens_model": "pinhole"}
    gm = GeometryManager(config)

    msg = MockCameraInfo()
    gm.update_from_camera_info(msg)

    assert gm.K[0, 0] == 1000.0
    assert gm.D.shape == (5,)


def test_rectify_passthrough():
    config = {"rectify_enabled": False}
    gm = GeometryManager(config)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    rect = gm.rectify(img)

    # Should return same object if disabled
    assert rect is img


def test_rectify_logic():
    # Basic test to ensure it doesn't crash
    config = {
        "lens_model": "fisheye",
        "rectify_enabled": True,
        "intrinsics": {"fx": 100, "fy": 100, "cx": 50, "cy": 50},
        "distortion": {"coeffs": [0.1, 0.1, 0, 0]},
    }
    gm = GeometryManager(config)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    rect = gm.rectify(img)

    assert rect.shape == img.shape
    assert rect is not img
