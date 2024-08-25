import cv2
import json
from daisykit.utils import get_asset_file, to_py_type
from daisykit import HumanPoseMoveNetFlow
import os
from typing import Dict, List
import numpy as np


class DaisykitHumanPoseDetector():
    """Daisykit Human Pose Detector
    Detect multiple human poses from image 
    """

    def __init__(self, config):
        self.human_pose_flow = HumanPoseMoveNetFlow(json.dumps(config))

    def detect(self, img, threshold=0.3, debug=False):
        """Detect human poses
        """

        # Convert image to RGB color space before pushing into Daisykit
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        poses = self.human_pose_flow.Process(img)

        if debug:
            draw = img.copy()
            self.human_pose_flow.DrawResult(draw, poses)
            draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
            cv2.imshow("Pose Result", draw)
            cv2.waitKey(1)

        #Convert poses to Python list of dict
        poses = to_py_type(poses)

        keypoints = []
        for pose in poses:
        # Extract keypoints for each pose
            kp = [[p["x"], p["y"], p["confidence"]] for p in pose["keypoints"]]
            keypoints.append(kp)

        return np.squeeze(keypoints,0)


# TEST

config = {
    "person_detection_model": {
        "model": get_asset_file(r"models/human_detection/ssd_mobilenetv2/ssd_mobilenetv2.param"),
        "weights": get_asset_file(r"models/human_detection/ssd_mobilenetv2/ssd_mobilenetv2.bin"),
        "input_width": 320,
        "input_height": 320,
        "use_gpu": False
    },
    "human_pose_model": {
        "model": get_asset_file(r"models/human_pose_detection/movenet/lightning.param"),
        "weights": get_asset_file(r"models/human_pose_detection/movenet/lightning.bin"),
        "input_width": 192,
        "input_height": 192,
        "use_gpu": False
    }
}
# Initialize the detector with your configuration
detector = DaisykitHumanPoseDetector(config)

# Load an image from file
image_path = r"C:\Users\buing\Downloads\anticheating\dataset\sitting\sitting (287).jpg"

#use cv2
img = cv2.imread(image_path)#output is numpy array with dimension of 
# print(img)


# Detect poses in the image
keypoints = detector.detect(img, debug=True)

# Print detected keypoints
print("Detected keypoints:", keypoints)

cv2.imshow("Original Image", img)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows() 
