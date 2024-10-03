import cv2
import json
from daisykit.utils import get_asset_file, to_py_type
from daisykit import HumanPoseMoveNetFlow
import os
from typing import Dict, List, Tuple
import numpy as np
from utils import *
import pandas as pd


class posedetector:
    """Daisykit Human Pose Detector - Detect multiple human poses from an image."""

    def __init__(self, config, many_pose: bool) -> None:
        """Initialize the posedetector."""
        self.human_pose_flow = HumanPoseMoveNetFlow(json.dumps(config))
        self.many_pose = many_pose

    def detect(self, image: np.ndarray, threshold=0.2) -> Person:
        """Runs model inference on the image.
        args: numpy array of image
        output: 
        if set as 1 pose per frame/image -> object of a pose(Person) or None
         else(many poses): list of pose/person object  """
        # Convert image to RGB color space before processing
        if image is None:
            return None
        
        image_height, image_width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect pose
        poses = self.human_pose_flow.Process(image_rgb)
        # Convert pose (list of objects) to list of dicts
        poses_dict = to_py_type(poses)
        keypoints = []
        for pose in poses_dict:
            # Extract keypoints for each pose
            kp = [[p["x"], p["y"], 1 if p["confidence"] >= threshold else 0] for p in pose["keypoints"]]
            keypoints.append(kp)

        keypoints_array = np.array(keypoints)
        #Debug part
        if self.many_pose==False:
            if keypoints_array.shape[0] == 1: # squeezed from (1,17,3) to (17,3)
                keypoints_array = np.squeeze(keypoints_array, 0)
                return person_from_keypoints_with_scores(keypoints_array, image_height, image_width)
            else:
                # comment out because this stops the loop in preprocessor
                # raise ValueError("Keypoints array corresponding to one pose can not be squeezed because of its shape ", keypoints_array.shape)
                return None # pass this to preprocessor to skip
            
        # hihi 
        else: #if there are n poses with keypoints array be(n,17,3)
            person_list=[]
            if keypoints_array.shape[0] > 0: #at least 1 pose
                for i in range(keypoints_array.shape[0]):
                    each_pose_keypoints=keypoints_array[i]
                    each_person= person_from_keypoints_with_scores(each_pose_keypoints, image_height, image_width)
                    person_list.append(each_person)
                return person_list
            else:
                # raise ValueError("can not detect any pose")
                return None # pass this to preprocessor to skip
            
        
    # def detect(self, input_image: np.ndarray) -> np.ndarray:
    #     """Run detection on an input image."""
    #     image_height, image_width, _ = input_image.shape
    #     poses, keypoint_with_scores = self._run_detector(input_image)
    #     return person_from_keypoints_with_scores(keypoint_with_scores, image_height, image_width)

# #Test
# # detector_1 = posedetector(config, False)
# detector_many = posedetector(config, True)

# image_path = r"/home/huyn/anticheating/dataset/WIN_20240823_15_22_32_Pro.jpg"
# img = cv2.imread(image_path)
# # # keypoints = detector_1.detect(img)
# persons = detector_many.detect(img)
# for i in range(len(persons)):
#     print("person", i+1 ,":",persons)

# #test dataframe of manyposes in a frame
# for person in persons:
#     keypoints_array = np.array([[kp.coordinate.x, kp.coordinate.y, kp.score] for kp in person.keypoints],
#                         dtype=np.float32)
#     # Normalize keypoints or process as needed
#     keypoints_dataframe = pd.DataFrame(keypoints_array.flatten()).T
#     keypoints_dataframe.columns= [f'{bp.name}_{coord}' for bp in BodyPart for coord in ['x', 'y', 'score']]
#     print(keypoints_dataframe.head)
    