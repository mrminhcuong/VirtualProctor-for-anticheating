import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import pandas as pd 
import os
from posedetector import posedetector
import wget
import csv
import tqdm 
from utils import BodyPart, Person, config
import cv2

detector_1 = posedetector(config, False)
detector_many = posedetector(config, True)

class Preprocessor(object):
#     this class preprocess pose samples, it predicts keypoints on the images 
#     and save those keypoints in a dataframe for the later use in the classification task 

        def __init__(self, images_in_folder):
            self._images_in_folder = images_in_folder
            self._message = []
            #get list of pose classes
            self._pose_class_names = sorted([n for n in os.listdir(images_in_folder)])
            self.data = pd.DataFrame()
        
        def process(self, detection_threshold=0.1):
            """used for process images in the given dataset folder
            can only process picture of 1 person, pose
            get the dataframe in which each row is the keypoints of only 1 pose
            """
            for class_index, pose_class_name in enumerate(self._pose_class_names):
                images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
                image_names = sorted([n for n in os.listdir(images_in_folder)])
                valid_image_count = 0
                # Detect pose landmarks in each image
                for image_name in tqdm.tqdm(image_names):
                    image_path = os.path.join(images_in_folder, image_name)
                    try:
                        image = cv2.imread(image_path)
                    except:
                        self._message.append('Skipped' + image_path + ' Invalid image')
                        continue                                           
                    person = detector_1.detect(image)   
                    if person is None:
                        self._message.append(f'Skipped {image_path} - No valid pose detected')
                        continue                                               
                    valid_image_count += 1                        
                    
                    pose_landmarks = np.array(
                            [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                            for keypoint in person.keypoints],
                                dtype=np.float32)

                    data_row = [image_name] + pose_landmarks.tolist() + [class_index, pose_class_name]
                    self.data = pd.concat([self.data, pd.DataFrame([data_row])], ignore_index=True)

            self.data.columns = ['filename'] + [f'{bp.name}_{coord}' for bp in BodyPart for coord in ['x', 'y', 'score']] + ['class_no', 'class_name']
            print(self._message)

        def class_names(self):
            return self.pose_class_names
        

images_in_folder = os.path.join('teststand', 'test')
train_preprocessor = Preprocessor(
    images_in_folder,
)
dataframe=train_preprocessor.process()   
print("new dataframe")
dataframe.head