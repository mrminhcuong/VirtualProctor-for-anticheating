import cv2
import json
from daisykit.utils import get_asset_file, to_py_type
from daisykit import HumanPoseMoveNetFlow
import os
from typing import Dict, List
import numpy as np
from data import BodyPart, Person, person_from_poses, config


class DaisykitPoseDetector():
    """Daisykit Human Pose Detector
    Detect multiple human poses from image 
    """
    _MIN_CROP_KEYPOINT_SCORE = 0.2
    _TORSO_EXPANSION_RATIO = 1.9
    _BODY_EXPANSION_RATIO = 1.2

    def __init__(self) -> None:
        """Initialize the DaisykitPoseDetector """
        self.human_pose_flow = HumanPoseMoveNetFlow(json.dumps(config))
        self._crop_region = None

    def init_crop_region(self, image_height: int, image_width: int) -> dict:
        """Defines the default crop region."""
        if image_width > image_height:
            x_min = 0.0
            box_width = 1.0
            y_min = (image_height / 2 - image_width / 2) / image_height
            box_height = image_width / image_height
        else:
            y_min = 0.0
            box_height = 1.0
            x_min = (image_width / 2 - image_height / 2) / image_width
            box_width = image_height / image_width

        return {
            'y_min': y_min,
            'x_min': x_min,
            'y_max': y_min + box_height,
            'x_max': x_min + box_width,
            'height': box_height,
            'width': box_width
        }

    def _torso_visible(self, keypoints: np.ndarray) -> bool:
        """Checks whether there are enough torso keypoints."""
        left_hip_score = keypoints[BodyPart.LEFT_HIP.value, 2]
        right_hip_score = keypoints[BodyPart.RIGHT_HIP.value, 2]
        left_shoulder_score = keypoints[BodyPart.LEFT_SHOULDER.value, 2]
        right_shoulder_score = keypoints[BodyPart.RIGHT_SHOULDER.value, 2]

        left_hip_visible = left_hip_score > DaisykitPoseDetector._MIN_CROP_KEYPOINT_SCORE
        right_hip_visible = right_hip_score > DaisykitPoseDetector._MIN_CROP_KEYPOINT_SCORE
        left_shoulder_visible = left_shoulder_score > DaisykitPoseDetector._MIN_CROP_KEYPOINT_SCORE
        right_shoulder_visible = right_shoulder_score > DaisykitPoseDetector._MIN_CROP_KEYPOINT_SCORE

        return ((left_hip_visible or right_hip_visible) and
                (left_shoulder_visible or right_shoulder_visible))

    def _determine_torso_and_body_range(self, keypoints: np.ndarray,
                                        target_keypoints: dict,
                                        center_y: float,
                                        center_x: float) -> list:
        """Calculates the maximum distance from keypoints to the center."""
        torso_joints = [
            BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER, BodyPart.LEFT_HIP,
            BodyPart.RIGHT_HIP
        ]
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for idx in range(len(BodyPart)):
            if keypoints[BodyPart(idx).value, 2] < DaisykitPoseDetector._MIN_CROP_KEYPOINT_SCORE:
                continue
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [
            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange
        ]

    def _determine_crop_region(self, keypoints: np.ndarray, image_height: int,
                               image_width: int) -> dict:
        """Determines the region to crop the image for further processing."""
        target_keypoints = {}
        for idx in range(len(BodyPart)):
            target_keypoints[BodyPart(idx)] = [
                keypoints[idx, 0] * image_height, keypoints[idx, 1] * image_width
            ]

        if self._torso_visible(keypoints):
            center_y = (target_keypoints[BodyPart.LEFT_HIP][0] +
                        target_keypoints[BodyPart.RIGHT_HIP][0]) / 2
            center_x = (target_keypoints[BodyPart.LEFT_HIP][1] +
                        target_keypoints[BodyPart.RIGHT_HIP][1]) / 2

            (max_torso_yrange, max_torso_xrange, max_body_yrange,
             max_body_xrange) = self._determine_torso_and_body_range(
                 keypoints, target_keypoints, center_y, center_x)

            crop_length_half = np.amax([
                max_torso_xrange * DaisykitPoseDetector._TORSO_EXPANSION_RATIO,
                max_torso_yrange * DaisykitPoseDetector._TORSO_EXPANSION_RATIO,
                max_body_yrange * DaisykitPoseDetector._BODY_EXPANSION_RATIO,
                max_body_xrange * DaisykitPoseDetector._BODY_EXPANSION_RATIO
            ])

            distances_to_border = np.array(
                [center_x, image_width - center_x, center_y, image_height - center_y])
            crop_length_half = np.amin(
                [crop_length_half, np.amax(distances_to_border)])

            if crop_length_half > max(image_width, image_height) / 2:
                return self.init_crop_region(image_height, image_width)
            else:
                crop_length = crop_length_half * 2
            crop_corner = [center_y - crop_length_half, center_x - crop_length_half]
            return {
                'y_min': crop_corner[0] / image_height,
                'x_min': crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': (crop_corner[0] + crop_length) / image_height -
                          crop_corner[0] / image_height,
                'width': (crop_corner[1] + crop_length) / image_width -
                         crop_corner[1] / image_width
            }
        else:
            return self.init_crop_region(image_height, image_width)

    def _crop_and_resize(self, image: np.ndarray, crop_region: dict,
                         crop_size: (int, int)) -> np.ndarray:
        """Crops and resizes the image to prepare for further processing."""
        y_min, x_min, y_max, x_max = [
            crop_region['y_min'], crop_region['x_min'], crop_region['y_max'],
            crop_region['x_max']
        ]

        crop_top = int(0 if y_min < 0 else y_min * image.shape[0])
        crop_bottom = int(image.shape[0] if y_max >= 1 else y_max * image.shape[0])
        crop_left = int(0 if x_min < 0 else x_min * image.shape[1])
        crop_right = int(image.shape[1] if x_max >= 1 else x_max * image.shape[1])

        padding_top = int(0 - y_min * image.shape[0] if y_min < 0 else 0)
        padding_bottom = int((y_max - 1) * image.shape[0] if y_max >= 1 else 0)
        padding_left = int(0 - x_min * image.shape[1] if x_min < 0 else 0)
        padding_right = int((x_max - 1) * image.shape[1] if x_max >= 1 else 0)

        output_image = image[crop_top:crop_bottom, crop_left:crop_right]
        output_image = cv2.copyMakeBorder(output_image, padding_top, padding_bottom,
                                          padding_left, padding_right,
                                          cv2.BORDER_CONSTANT)
        output_image = cv2.resize(output_image, (crop_size[0], crop_size[1]))

        return output_image
    
    def _run_detector(
      self, image: np.ndarray, crop_region: Dict[(str, float)],
      crop_size: (int, int)) -> np.ndarray:
        """Runs model inference on the cropped region.

        The function runs the model inference on the cropped region and updates the model output to the original image coordinate system.
        Args:
        image: The input image(numpy array).
        crop_region: The region of interest to run inference on.
        crop_size: The size of the crop region.

        Returns:
        An array of shape [17, 3] representing the keypoint absolute coordinates
        and scores.
        """

        input_image = self._crop_and_resize(image, crop_region, crop_size=crop_size)
        # input_image = input_image.astype(dtype=np.uint8)

        # Convert image to RGB color space before pushing into Daisykit
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        poses = self.human_pose_flow.Process(img)

        poses = np.squeeze(poses)

        # Update the coordinates.
        for idx in range(len(BodyPart)):
            poses[idx, 0] = crop_region[
                'y_min'] + crop_region['height'] * poses[idx, 0]
            poses[idx, 1] = crop_region[
                'x_min'] + crop_region['width'] * poses[idx, 1]

        return poses
  
    

def detect(self,
             input_image: np.ndarray,
             reset_crop_region: bool = False) -> Person:
    """Run detection on an input image.

    Args:
      input_image: A [height, width, 3] RGB image. Note that height and width
        can be anything since the image will be immediately resized according to
        the needs of the model within this function.
      reset_crop_region: Whether to use the crop region inferred from the
        previous detection result to improve accuracy. Set to True if this is a
        frame from a video. Set to False if this is a static image. Default
        value is True.

    Returns:
      An array of shape [17, 3] representing the keypoint coordinates and
      scores.
    """
    image_height, image_width, _ = input_image.shape
    if (self._crop_region is None) or reset_crop_region:
      # Set crop region for the first frame.
      self._crop_region = self.init_crop_region(image_height, image_width)

    # Detect pose using the crop region inferred from the detection result in
    # the previous frame
    keypoint_with_scores = self._run_detector(
        input_image,
        self._crop_region,
        crop_size=(self._input_height, self._input_width))
    # Calculate the crop region for the next frame
    self._crop_region = self._determine_crop_region(keypoint_with_scores,
                                                    image_height, image_width)

    # Convert the keypoints with scores to a Person data type

    return keypoint_with_scores
    #return person_from_poses(keypoint_with_scores, image_height, image_width)


