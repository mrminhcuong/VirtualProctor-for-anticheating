from utils import * 
import sys
import cv2
import os
from sys import platform
import argparse
import math
import numpy as np
import random
import copy
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import concurrent.futures
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

from daisykit import HumanPoseMoveNetFlow

class CheatDetection:
    def __init__(self):
        # Initialize DaisyKit HumanPoseMoveNetFlow
        config_str = '{"model": "movenet", "type": "thunder", "input_size": [256, 256]}'  # Example config
        self.pose_detector = HumanPoseMoveNetFlow(config_str)
        
        # Starting XGBoost
        self.model = XGBClassifier()
        xgboost_model_path = dir_path + "./XGB_BiCD_Tuned_GPU_05.model"
        self.model.load_model(xgboost_model_path)
        self.model.set_params(**{"predictor": "gpu_predictor"})

    def GeneratePose(self, img):
        # Process image with DaisyKit
        self.poses = self.pose_detector.Process(img)
        # Drawing the result on the image
        output_img = self.pose_detector.DrawResult(img, self.poses)
        return output_img

    def DetectCheat(self, ShowPose=True, img=None):
        cheating = False
        if ShowPose:
            output_img = img.copy()
        else:
            output_img = img

        if self.poses:
            pose_collection = self._extract_keypoints(self.poses)
            original_posecollection = copy.deepcopy(pose_collection)
            pose_collection = NormalizePoseCollection(pose_collection)
            pose_collection = ReshapePoseCollection(pose_collection)
            pose_collection = ConvertToDataFrame(pose_collection)
            preds = self.model.predict(pose_collection)
            for idx, pred in enumerate(preds):
                if pred:
                    cheating = True
                    output_img = DrawBoundingRectangle(
                        output_img, GetBoundingBoxCoords(original_posecollection[idx])
                    )
        return output_img, cheating

    def _extract_keypoints(self, poses):
        # Convert DaisyKit pose format to OpenPose-style keypoints
        keypoints = []
        for pose in poses:
            keypoints.append(pose.keypoints.flatten())  # Assuming DaisyKit returns keypoints in a structured format
        return np.array(keypoints)

# Initialize CheatDetection
cheat_detector = CheatDetection()

# Load an image or capture from a webcam
img = cv2.imread(r"C:\Users\buing\Downloads\anticheating\dataset\cheating\WIN_20240823_15_09_36_Pro.jpg")

# Generate Pose
output_img = cheat_detector.GeneratePose(img)

# Detect cheating
result_img, is_cheating = cheat_detector.DetectCheat(ShowPose=True, img=output_img)

# Display the result
cv2.imshow('Cheat Detection', result_img)
print(f"Cheating detected: {is_cheating}")
cv2.waitKey(0)
cv2.destroyAllWindows()
    