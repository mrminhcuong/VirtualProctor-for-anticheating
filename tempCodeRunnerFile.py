import cv2
import json
from daisykit.utils import get_asset_file, to_py_type
from daisykit import HumanPoseMoveNetFlow
import os
from typing import Dict, List
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
# Load an image from file
image_path = r"C:\Users\buing\Downloads\anticheating\dataset\sitting\sitting (287).jpg"

# #use cv2
imgcv2 = cv2.imread(image_path)#output is numpy array with dimension of 
print("cv2 image type is:", imgcv2.dtype, imgcv2.size, imgcv2.shape)