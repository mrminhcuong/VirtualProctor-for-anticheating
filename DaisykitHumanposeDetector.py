import cv2
import json
from daisykit.utils import get_asset_file, to_py_type
from daisykit import HumanPoseMoveNetFlow
import os
from typing import Dict, List, Tuple
import numpy as np
from data import BodyPart, Person, person_from_keypoints_with_scores, config

class DaisykitPoseDetector:
    """Daisykit Human Pose Detector - Detect multiple human poses from an image."""

    def __init__(self, config) -> None:
        """Initialize the DaisykitPoseDetector."""
        self.human_pose_flow = HumanPoseMoveNetFlow(json.dumps(config))
        self._input_height = 256
        self._input_width = 256

    def _pad_and_resize(self, image: np.ndarray) -> (np.ndarray, np.ndarray):
        
        """Resizes the image while maintaining aspect ratio, and pads to match the target size."""
        target_size= (256, 256)
        height, width = image.shape[:2]
        scale = min(target_size[1] / width, target_size[0] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        padded_image = cv2.copyMakeBorder(
            resized_image,
            top=(target_size[0] - new_height) // 2,
            bottom=(target_size[0] - new_height + 1) // 2,
            left=(target_size[1] - new_width) // 2,
            right=(target_size[1] - new_width + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        return resized_image, padded_image


    def _run_detector(self, image: np.ndarray) -> Tuple[List[object], np.ndarray]:
        """Runs model inference on the resized and padded image."""
        resizeimage, padimage = self._pad_and_resize(image)
        # Convert image to RGB color space before processing
        paddedimage = cv2.cvtColor(padimage, cv2.COLOR_BGR2RGB)

        # Detect pose
        poses = self.human_pose_flow.Process(paddedimage)
        # convert pose(list of object to list of dict)
        poses_dict = to_py_type(poses)
        keypoints = []
        for pose in poses_dict:
            # Extract keypoints for each pose
            kp = [[p["x"], p["y"], p["confidence"]] for p in pose["keypoints"]]
            keypoints.append(kp)
        keypoints=np.squeeze(np.array(keypoints),0)
        return poses, keypoints

    def detect(self, input_image: np.ndarray) -> np.ndarray:
        """Run detection on an input image."""
        image_height, image_width, _ = input_image.shape
        poses, keypoint_with_scores = self._run_detector(input_image)
        return person_from_keypoints_with_scores(keypoint_with_scores, image_height, image_width)
    
    def visualize(self, input_image: np.ndarray):
        """Visualize image with detected keypoints after pose estimation
        Args: numpy array of input image
             """
        resizeimage, padimage = self._pad_and_resize(input_image)
        poses, keypoint_with_scores = self._run_detector(input_image)
        draw = padimage.copy()
        self.human_pose_flow.DrawResult(draw, poses)
        draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
        cv2.imshow("Pose Result", draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 


# ## TEST
detector1 = DaisykitPoseDetector(config)
# detector2 = draftposedetector(config)
image_path = r"C:\Users\buing\Downloads\anticheating\dataset\train\cheating\cheating (418).jpg"
img = cv2.imread(image_path)#original image is numpy array with dimension of (720, 1280, 3) 
print(img.shape) 
#depends on the image original size
# detector1.visualize(img)

# res, pad = detector1._pad_and_resize(img) 
# print("after resizing", res.shape) #(144, 256, 3)
# print("after resize and padding", pad.shape)  #after resize and padding (256, 256, 3)
#detector1.visualize(pad) 

# keypointsrun = detector1._run_detector(img)
# print(keypointsrun.shape)
keypoints = detector1.detect(img)
# print("Detected keypoints:", keypoints)