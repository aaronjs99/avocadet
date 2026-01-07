
import cv2
import numpy as np

def test_detect():
    img = cv2.imread('/home/ig-handle/.gemini/antigravity/brain/1cd36689-c3d5-4357-b469-ba0e0d4555e4/camera_debug.png')
    if img is None:
        print("Image not found")
        return

    # Dictionary 4x4_250 is int value 2
    # But let's use the enum to be safe or try all
    # cv2.aruco.DICT_4X4_250
    
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        aruco_params = cv2.aruco.DetectorParameters()
        
        # Use ArucoDetector directly for OpenCV 4.7+
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(img)
        
        print(f"Detected IDs: {ids}")
        
        if ids is not None:
             print(f"Found {len(ids)} markers")
        else:
             print("No markers found")
            
    except Exception as e:
        print(f"Error: {e}")

test_detect()
