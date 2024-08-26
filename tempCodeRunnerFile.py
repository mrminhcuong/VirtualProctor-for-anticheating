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
#depends on the image original size
# detector1.visualize(img)

# TEST OUTPUT OF 2 FUNCTIONS RELATED TO PADDING+RESIZING
res, pad = detector1._pad_and_resize(img) 
print("after resizing", res.shape) #(144, 256, 3)
print("after resize and padding", pad.shape)  #after resize and padding (256, 256, 3)
detector1.visualize(pad) 