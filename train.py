import csv
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils import BodyPart 
import os
import numpy
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from preprocess import Preprocessor

def make_df(folder, subfolder):
    images_in_folder=os.path.join(folder, subfolder)
    train_preprocessor = Preprocessor(
    images_in_folder,)
    dataframe =train_preprocessor.process()   
    return dataframe  

def load_data(df):
    df.drop(['filename'],axis=1, inplace=True)
    classes = df.pop('class_name').unique()
    y = df.pop('class_no')
    X = df.astype('float64')
    y = keras.utils.to_categorical(y)
    return X, y, classes


def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

    It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    # hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
    #                              BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                      BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    # torso_size = tf.linalg.norm(shoulders_center - hips_center)
    # Pose center
    pose_center= shoulders_center
    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center = tf.broadcast_to(pose_center,
                                    [tf.size(landmarks) // (17*2), 17, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center, 0, axis=0,
                name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))
    pose_size= max_dist
    # Normalize scale
    # pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size



def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
  """
  # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart. LEFT_SHOULDER, 
                                 BodyPart.RIGHT_SHOULDER)

    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center, 
                                [tf.size(landmarks) // (17*2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks

import numpy as np

def calculate_angle(pointA, pointB, pointC):
    """Calculates the angle ABC (in degrees) where B is the vertex."""
    # Convert tensors to numpy arrays if necessary
    A = pointA.numpy() if isinstance(pointA, tf.Tensor) else pointA
    B = pointB.numpy() if isinstance(pointB, tf.Tensor) else pointB
    C = pointC.numpy() if isinstance(pointC, tf.Tensor) else pointC

    #print("a,b,c are", A,B,C, type(A), type(B), A.shape, C.shape)
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    
    # Vectors BA and BC
    BA = A - B
    BC = C - B

    # Calculate the cosine of the angle using the dot product and magnitudes
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    # Clip values to avoid numerical errors outside [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    # Convert to angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)
    return embedding 


def preprocess_data(X_train):
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 51)))
        processed_X_train.append(tf.reshape(embedding, (34)))
    return tf.convert_to_tensor(processed_X_train) 


def landmarks_to_embedding_angles(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)
    
    # Extract specific points for angle calculations
    left_shoulder = landmarks[:, BodyPart.LEFT_SHOULDER.value, :]
    right_shoulder = landmarks[:, BodyPart.RIGHT_SHOULDER.value, :]
    left_elbow = landmarks[:, BodyPart.LEFT_ELBOW.value, :]
    right_elbow = landmarks[:, BodyPart.RIGHT_ELBOW.value, :]
    left_wrist = landmarks[:, BodyPart.LEFT_WRIST.value, :]
    right_wrist = landmarks[:, BodyPart.RIGHT_WRIST.value, :]
    nose = landmarks[:, BodyPart.NOSE.value, :]

    # Calculate angles
    angles = [
        calculate_angle(left_shoulder, left_elbow, left_wrist),
        calculate_angle(right_shoulder, right_elbow, right_wrist),
        calculate_angle(nose, right_shoulder, right_wrist),
        calculate_angle(nose, left_shoulder, left_wrist),
        calculate_angle( right_elbow, nose, left_elbow),
        calculate_angle( right_wrist, nose, left_wrist),
    ]

    # Add angles to embedding
    angles_tensor = tf.convert_to_tensor(angles, dtype=tf.float32)
    angles_tensor = tf.reshape(angles_tensor, (1, -1))
    embedding = tf.concat([embedding, angles_tensor], axis=1)
    return embedding 


def preprocess_data_with_angles(X_train):
    processed_X_train = []
    for i in range(X_train.shape[0]):
        embedding = landmarks_to_embedding_angles(tf.reshape(tf.convert_to_tensor(X_train.iloc[i]), (1, 51)))
        processed_X_train.append(tf.reshape(embedding, (46)))
    return tf.convert_to_tensor(processed_X_train) 