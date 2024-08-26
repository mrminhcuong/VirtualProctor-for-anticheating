import cv2
import json
import daisykit
from daisykit.utils import get_asset_file, to_py_type
from daisykit import HumanPoseMoveNetFlow
from data import config, BodyPart
from draftPoseDetection import draftposedetector
from DaisykitHumanposeDetector import DaisykitPoseDetector

#test
detector1 = DaisykitPoseDetector(config)
detector2 = draftposedetector(config)
image_path = r"C:\Users\buing\Downloads\anticheating\dataset\train\cheating\cheating (418).jpg"
img = cv2.imread(image_path)#original image is numpy array with dimension of (720, 1280, 3) 
print(img.shape) 

# org, resize = detector2.resize_pad(img)
# print("after padding but not resize", org.shape)  #(1279, 1280, 3)
# print("resize image", resize.shape) #(256, 256, 3)
# detector2.visualize2(resize, debug=True ) 


### video capture
human_pose_flow = HumanPoseMoveNetFlow(json.dumps(config))

###
print("Running....") 
# Open video stream from webcam
vid = cv2.VideoCapture(0)
 
while(True):
 
    # Capture the video frame
    ret, frame = vid.read()
 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    poses = human_pose_flow.Process(frame)
    human_pose_flow.DrawResult(frame, poses)
 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
 
    # Convert poses to Python list of dict
    poses = to_py_type(poses)
 
    # Display the result frame
    cv2.imshow('frame', frame)
 
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

