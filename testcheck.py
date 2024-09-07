import cv2
import json
import daisykit
from daisykit.utils import get_asset_file, to_py_type
from daisykit import HumanPoseMoveNetFlow
from utils import config

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

