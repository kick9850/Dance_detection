# Import the much needed stuff for training
import pandas as pd
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import csv
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.utils import to_categorical

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

# Function to Extract Feature from images or Frame
def extract_feature(input_image):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    image = cv.imread(input_image)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        while True:
            results = pose.process(cv.flip(cv.cvtColor(image, cv.COLOR_BGR2RGB), 1))
            image_height, image_width, _ = image.shape
            # Print handedness (left v.s. right hand).
            # Caution : Uncomment these print command will resulting long log of mediapipe log
            #print(f'Handedness of {input_image}:')
            #print(results.multi_handedness)

            # Draw hand landmarks of each hand.
            # Caution : Uncomment these print command will resulting long log of mediapipe log
            #print(f'Hand landmarks of {input_image}:')
            if not results.pose_world_landmarks:
                # Here we will set whole landmarks into zero as no handpose detected
                # in a picture wanted to extract.
                noseX = 0
                noseY = 0
                noseZ = 0

                left_eye_innerX = 0
                left_eye_innerY = 0
                left_eye_innerZ = 0

                left_eyeX = 0
                left_eyeY = 0
                left_eyeZ = 0

                left_eye_outerX = 0
                left_eye_outerY = 0
                left_eye_outerZ = 0

                right_eye_innerX = 0
                right_eye_innerY = 0
                right_eye_innerZ = 0

                right_eyeX = 0
                right_eyeY = 0
                right_eyeZ = 0

                right_eye_outerX = 0
                right_eye_outerY = 0
                right_eye_outerZ = 0

                left_earX = 0
                left_earY = 0
                left_earZ = 0

                right_earX = 0
                right_earY = 0
                right_earZ = 0

                mouth_leftX = 0
                mouth_leftY = 0
                mouth_leftZ = 0

                mouth_rightX = 0
                mouth_rightY = 0
                mouth_rightZ = 0

                left_shoulderX = 0
                left_shoulderY = 0
                left_shoulderZ = 0

                right_shoulderX = 0
                right_shoulderY = 0
                right_shoulderZ = 0

                left_elbowX = 0
                left_elbowY = 0
                left_elbowZ = 0

                right_elbowX = 0
                right_elbowY = 0
                right_elbowZ = 0

                left_wristX = 0
                left_wristY = 0
                left_wristZ = 0

                right_wristX = 0
                right_wristY = 0
                right_wristZ = 0

                left_pinkyX = 0
                left_pinkyY = 0
                left_pinkyZ = 0

                right_pinkyX = 0
                right_pinkyY = 0
                right_pinkyZ = 0

                left_indexX = 0
                left_indexY = 0
                left_indexZ = 0

                right_indexX = 0
                right_indexY = 0
                right_indexZ = 0

                left_thumbX = 0
                left_thumbY = 0
                left_thumbZ = 0

                right_thumbX = 0
                right_thumbY = 0
                right_thumbZ = 0

                left_hipX = 0
                left_hipY = 0
                left_hipZ = 0

                right_hipX = 0
                right_hipY = 0
                right_hipZ = 0

                left_kneeX = 0
                left_kneeY = 0
                left_kneeZ = 0

                right_kneeX = 0
                right_kneeY = 0
                right_kneeZ = 0

                left_ankleX = 0
                left_ankleY = 0
                left_ankleZ = 0

                right_ankleX = 0
                right_ankleY = 0
                right_ankleZ = 0

                left_heelX = 0
                left_heelY = 0
                left_heelZ = 0

                right_heelX = 0
                right_heelY = 0
                right_heelZ  = 0

                left_foot_indexX = 0
                left_foot_indexY = 0
                left_foot_indexZ = 0

                right_foot_indexX = 0
                right_foot_indexY = 0
                right_foot_indexZ = 0

                # Set image to Zero
                annotated_image = 0

                return (noseX,	noseY, noseZ,
                        left_eye_innerX, left_eye_innerY, left_eye_innerZ,
                        left_eyeX,	left_eyeY,	left_eyeZ,
                        left_eye_outerX, left_eye_outerY, left_eye_outerZ,
                        right_eye_innerX, right_eye_innerY, right_eye_innerZ,
                        right_eyeX, right_eyeY, right_eyeZ,
                        right_eye_outerX, right_eye_outerY, right_eye_outerZ,
                        left_earX, left_earY, left_earZ,
                        right_earX, right_earY, right_earZ,
                        mouth_leftX, mouth_leftY, mouth_leftZ,
                        mouth_rightX, mouth_rightY, mouth_rightZ,
                        left_shoulderX, left_shoulderY, left_shoulderZ,
                        right_shoulderX, right_shoulderY, right_shoulderZ,
                        left_elbowX, left_elbowY, left_elbowZ,
                        right_elbowX, right_elbowY, right_elbowZ,
                        left_wristX, left_wristY, left_wristZ,
                        right_wristX, right_wristY, right_wristZ,
                        left_pinkyX, left_pinkyY, left_pinkyZ,
                        right_pinkyX, right_pinkyY, right_pinkyZ,
                        left_indexX, left_indexY, left_indexZ,
                        right_indexX, right_indexY, right_indexZ,
                        left_thumbX, left_thumbY, left_thumbZ,
                        right_thumbX, right_thumbY, right_thumbZ,
                        left_hipX,	left_hipY,	left_hipZ,
                        right_hipX,	right_hipY,	right_hipZ,
                        left_kneeX,	left_kneeY,	left_kneeZ,
                        right_kneeX, right_kneeY, right_kneeZ,
                        left_ankleX, left_ankleY, left_ankleZ,
                        right_ankleX, right_ankleY, right_ankleZ,
                        left_heelX, left_heelY, left_heelZ,
                        right_heelX, right_heelY, right_heelZ,
                        left_foot_indexX, left_foot_indexY, left_foot_indexZ,
                        right_foot_indexX, right_foot_indexY, right_foot_indexZ,
                        annotated_image)

            annotated_image = cv.flip(image.copy(), 1)
            for pose_landmarks in results.pose_world_landmarks:
                noseX = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x
                noseY = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
                noseZ = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z

                left_eye_innerX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].x
                left_eye_innerY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].y
                left_eye_innerZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].z

                left_eyeX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x
                left_eyeY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y
                left_eyeZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].z

                left_eye_outerX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].x
                left_eye_outerY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].y
                left_eye_outerZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].z

                right_eye_innerX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].x
                right_eye_innerY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].y
                right_eye_innerZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].z

                right_eyeX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x
                right_eyeY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y
                right_eyeZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].z

                right_eye_outerX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].x
                right_eye_outerY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].y
                right_eye_outerZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].z

                left_earX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x
                left_earY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y
                left_earZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].z

                right_earX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x
                right_earY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y
                right_earZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].z

                mouth_leftX = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].x
                mouth_leftY = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].y
                mouth_leftZ = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].z

                mouth_rightX = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x
                mouth_rightY = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y
                mouth_rightZ = pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].z

                left_shoulderX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
                left_shoulderY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                left_shoulderZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z

                right_shoulderX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
                right_shoulderY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                right_shoulderZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z

                left_elbowX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
                left_elbowY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
                left_elbowZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z

                right_elbowX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
                right_elbowY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
                right_elbowZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z

                left_wristX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
                left_wristY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
                left_wristZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z

                right_wristX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
                right_wristY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
                right_wristZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z

                left_pinkyX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x
                left_pinkyY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y
                left_pinkyZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].z

                right_pinkyX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x
                right_pinkyY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y
                right_pinkyZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].z

                left_indexX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x
                left_indexY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y
                left_indexZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].z

                right_indexX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x
                right_indexY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y
                right_indexZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].z

                left_thumbX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].x
                left_thumbY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y
                left_thumbZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].z

                right_thumbX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].x
                right_thumbY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y
                right_thumbZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].z

                left_hipX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x
                left_hipY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
                left_hipZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z

                right_hipX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x
                right_hipY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
                right_hipZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z

                left_kneeX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
                left_kneeY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
                left_kneeZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z

                right_kneeX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x
                right_kneeY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
                right_kneeZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z

                left_ankleX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
                left_ankleY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
                left_ankleZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z

                right_ankleX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x
                right_ankleY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
                right_ankleZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z

                left_heelX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x
                left_heelY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y
                left_heelZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].z

                right_heelX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x
                right_heelY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y
                right_heelZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].z

                left_foot_indexX = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x
                left_foot_indexY = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
                left_foot_indexZ = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z

                right_foot_indexX = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x
                right_foot_indexY = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
                right_foot_indexZ = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z

                # Draw the Skeleton
                mp_drawing.draw_landmarks(annotated_image, pose_landmarks, mp_pose.POSE_CONNECTIONS)

            return (noseX, noseY, noseZ,
                    left_eye_innerX, left_eye_innerY, left_eye_innerZ,
                    left_eyeX, left_eyeY, left_eyeZ,
                    left_eye_outerX, left_eye_outerY, left_eye_outerZ,
                    right_eye_innerX, right_eye_innerY, right_eye_innerZ,
                    right_eyeX, right_eyeY, right_eyeZ,
                    right_eye_outerX, right_eye_outerY, right_eye_outerZ,
                    left_earX, left_earY, left_earZ,
                    right_earX, right_earY, right_earZ,
                    mouth_leftX, mouth_leftY, mouth_leftZ,
                    mouth_rightX, mouth_rightY, mouth_rightZ,
                    left_shoulderX, left_shoulderY, left_shoulderZ,
                    right_shoulderX, right_shoulderY, right_shoulderZ,
                    left_elbowX, left_elbowY, left_elbowZ,
                    right_elbowX, right_elbowY, right_elbowZ,
                    left_wristX, left_wristY, left_wristZ,
                    right_wristX, right_wristY, right_wristZ,
                    left_pinkyX, left_pinkyY, left_pinkyZ,
                    right_pinkyX, right_pinkyY, right_pinkyZ,
                    left_indexX, left_indexY, left_indexZ,
                    right_indexX, right_indexY, right_indexZ,
                    left_thumbX, left_thumbY, left_thumbZ,
                    right_thumbX, right_thumbY, right_thumbZ,
                    left_hipX, left_hipY, left_hipZ,
                    right_hipX, right_hipY, right_hipZ,
                    left_kneeX, left_kneeY, left_kneeZ,
                    right_kneeX, right_kneeY, right_kneeZ,
                    left_ankleX, left_ankleY, left_ankleZ,
                    right_ankleX, right_ankleY, right_ankleZ,
                    left_heelX, left_heelY, left_heelZ,
                    right_heelX, right_heelY, right_heelZ,
                    left_foot_indexX, left_foot_indexY, left_foot_indexZ,
                    right_foot_indexX, right_foot_indexY, right_foot_indexZ,
                    annotated_image)
#Function to create CSV file or add dataset to the existed CSV file
def toCSV(filecsv, class_type,
          noseX, noseY, noseZ,
          left_eye_innerX, left_eye_innerY, left_eye_innerZ,
          left_eyeX, left_eyeY, left_eyeZ,
          left_eye_outerX, left_eye_outerY, left_eye_outerZ,
          right_eye_innerX, right_eye_innerY, right_eye_innerZ,
          right_eyeX, right_eyeY, right_eyeZ,
          right_eye_outerX, right_eye_outerY, right_eye_outerZ,
          left_earX, left_earY, left_earZ,
          right_earX, right_earY, right_earZ,
          mouth_leftX, mouth_leftY, mouth_leftZ,
          mouth_rightX, mouth_rightY, mouth_rightZ,
          left_shoulderX, left_shoulderY, left_shoulderZ,
          right_shoulderX, right_shoulderY, right_shoulderZ,
          left_elbowX, left_elbowY, left_elbowZ,
          right_elbowX, right_elbowY, right_elbowZ,
          left_wristX, left_wristY, left_wristZ,
          right_wristX, right_wristY, right_wristZ,
          left_pinkyX, left_pinkyY, left_pinkyZ,
          right_pinkyX, right_pinkyY, right_pinkyZ,
          left_indexX, left_indexY, left_indexZ,
          right_indexX, right_indexY, right_indexZ,
          left_thumbX, left_thumbY, left_thumbZ,
          right_thumbX, right_thumbY, right_thumbZ,
          left_hipX, left_hipY, left_hipZ,
          right_hipX, right_hipY, right_hipZ,
          left_kneeX, left_kneeY, left_kneeZ,
          right_kneeX, right_kneeY, right_kneeZ,
          left_ankleX, left_ankleY, left_ankleZ,
          right_ankleX, right_ankleY, right_ankleZ,
          left_heelX, left_heelY, left_heelZ,
          right_heelX, right_heelY, right_heelZ,
          left_foot_indexX, left_foot_indexY, left_foot_indexZ,
          right_foot_indexX, right_foot_indexY, right_foot_indexZ):
    if os.path.isfile(filecsv):
        # print ("File exist thus shall write append to the file")
        with open(filecsv, 'a+', newline='') as file:
            # Create a writer object from csv module
            writer = csv.writer(file)
            writer.writerow([class_type,
                             noseX, noseY, noseZ,
                             left_eye_innerX, left_eye_innerY, left_eye_innerZ,
                             left_eyeX, left_eyeY, left_eyeZ,
                             left_eye_outerX, left_eye_outerY, left_eye_outerZ,
                             right_eye_innerX, right_eye_innerY, right_eye_innerZ,
                             right_eyeX, right_eyeY, right_eyeZ,
                             right_eye_outerX, right_eye_outerY, right_eye_outerZ,
                             left_earX, left_earY, left_earZ,
                             right_earX, right_earY, right_earZ,
                             mouth_leftX, mouth_leftY, mouth_leftZ,
                             mouth_rightX, mouth_rightY, mouth_rightZ,
                             left_shoulderX, left_shoulderY, left_shoulderZ,
                             right_shoulderX, right_shoulderY, right_shoulderZ,
                             left_elbowX, left_elbowY, left_elbowZ,
                             right_elbowX, right_elbowY, right_elbowZ,
                             left_wristX, left_wristY, left_wristZ,
                             right_wristX, right_wristY, right_wristZ,
                             left_pinkyX, left_pinkyY, left_pinkyZ,
                             right_pinkyX, right_pinkyY, right_pinkyZ,
                             left_indexX, left_indexY, left_indexZ,
                             right_indexX, right_indexY, right_indexZ,
                             left_thumbX, left_thumbY, left_thumbZ,
                             right_thumbX, right_thumbY, right_thumbZ,
                             left_hipX, left_hipY, left_hipZ,
                             right_hipX, right_hipY, right_hipZ,
                             left_kneeX, left_kneeY, left_kneeZ,
                             right_kneeX, right_kneeY, right_kneeZ,
                             left_ankleX, left_ankleY, left_ankleZ,
                             right_ankleX, right_ankleY, right_ankleZ,
                             left_heelX, left_heelY, left_heelZ,
                             right_heelX, right_heelY, right_heelZ,
                             left_foot_indexX, left_foot_indexY, left_foot_indexZ,
                             right_foot_indexX, right_foot_indexY, right_foot_indexZ])
    else:
        # print ("File not exist thus shall create new file as", filecsv)
        with open(filecsv, 'w', newline='') as file:
            # Create a writer object from csv module
            writer = csv.writer(file)
            writer.writerow(["class_type",
                             'noseX', 'noseY', 'noseZ',
                             'left_eye_innerX', 'left_eye_innerY', 'left_eye_innerZ',
                             'left_eyeX', 'left_eyeY', 'left_eyeZ',
                             'left_eye_outerX', 'left_eye_outerY', 'left_eye_outerZ',
                             'right_eye_innerX', 'right_eye_innerY', 'right_eye_innerZ',
                             'right_eyeX', 'right_eyeY', 'right_eyeZ',
                             'right_eye_outerX', 'right_eye_outerY', 'right_eye_outerZ',
                             'left_earX', 'left_earY', 'left_earZ',
                             'right_earX', 'right_earY', 'right_earZ',
                             'mouth_leftX', 'mouth_leftY', 'mouth_leftZ',
                             'mouth_rightX', 'mouth_rightY', 'mouth_rightZ',
                             'left_shoulderX', 'left_shoulderY', 'left_shoulderZ',
                             'right_shoulderX', 'right_shoulderY', 'right_shoulderZ',
                             'left_elbowX', 'left_elbowY', 'left_elbowZ',
                             'right_elbowX', 'right_elbowY', 'right_elbowZ',
                             'left_wristX', 'left_wristY', 'left_wristZ',
                             'right_wristX', 'right_wristY', 'right_wristZ',
                             'left_pinkyX', 'left_pinkyY', 'left_pinkyZ',
                             'right_pinkyX', 'right_pinkyY', 'right_pinkyZ',
                             'left_indexX', 'left_indexY', 'left_indexZ',
                             'right_indexX', 'right_indexY', 'right_indexZ',
                             'left_thumbX', 'left_thumbY', 'left_thumbZ',
                             'right_thumbX', 'right_thumbY', 'right_thumbZ',
                             'left_hipX', 'left_hipY', 'left_hipZ',
                             'right_hipX', 'right_hipY', 'right_hipZ',
                             'left_kneeX', 'left_kneeY', 'left_kneeZ',
                             'right_kneeX', 'right_kneeY', 'right_kneeZ',
                             'left_ankleX', 'left_ankleY', 'left_ankleZ',
                             'right_ankleX', 'right_ankleY', 'right_ankleZ',
                             'left_heelX', 'left_heelY', 'left_heelZ',
                             'right_heelX', 'right_heelY', 'right_heelZ',
                             'left_foot_indexX', 'left_foot_indexY', 'left_foot_indexZ',
                             'right_foot_indexX', 'right_foot_indexY', 'right_foot_indexZ'])
            writer.writerow([class_type,
                             noseX, noseY, noseZ,
                             left_eye_innerX, left_eye_innerY, left_eye_innerZ,
                             left_eyeX, left_eyeY, left_eyeZ,
                             left_eye_outerX, left_eye_outerY, left_eye_outerZ,
                             right_eye_innerX, right_eye_innerY, right_eye_innerZ,
                             right_eyeX, right_eyeY, right_eyeZ,
                             right_eye_outerX, right_eye_outerY, right_eye_outerZ,
                             left_earX, left_earY, left_earZ,
                             right_earX, right_earY, right_earZ,
                             mouth_leftX, mouth_leftY, mouth_leftZ,
                             mouth_rightX, mouth_rightY, mouth_rightZ,
                             left_shoulderX, left_shoulderY, left_shoulderZ,
                             right_shoulderX, right_shoulderY, right_shoulderZ,
                             left_elbowX, left_elbowY, left_elbowZ,
                             right_elbowX, right_elbowY, right_elbowZ,
                             left_wristX, left_wristY, left_wristZ,
                             right_wristX, right_wristY, right_wristZ,
                             left_pinkyX, left_pinkyY, left_pinkyZ,
                             right_pinkyX, right_pinkyY, right_pinkyZ,
                             left_indexX, left_indexY, left_indexZ,
                             right_indexX, right_indexY, right_indexZ,
                             left_thumbX, left_thumbY, left_thumbZ,
                             right_thumbX, right_thumbY, right_thumbZ,
                             left_hipX, left_hipY, left_hipZ,
                             right_hipX, right_hipY, right_hipZ,
                             left_kneeX, left_kneeY, left_kneeZ,
                             right_kneeX, right_kneeY, right_kneeZ,
                             left_ankleX, left_ankleY, left_ankleZ,
                             right_ankleX, right_ankleY, right_ankleZ,
                             left_heelX, left_heelY, left_heelZ,
                             right_heelX, right_heelY, right_heelZ,
                             left_foot_indexX, left_foot_indexY, left_foot_indexZ,
                             right_foot_indexX, right_foot_indexY, right_foot_indexZ])


# Extract Feature for Training
# We will using SIBI datasets version V02
paths = "img_cp/old/"
csv_path = "Dance_training.csv"

if os.path.exists(csv_path):
    print("CSV File does exist, going delete before start extraction and replace it with new")
    os.remove(csv_path)
else:
    print("The CSV file does not exist", csv_path, ",Going Create after Extraction")

for dirlist in os.listdir(paths):
    for root, directories, filenames in os.walk(os.path.join(paths, dirlist)):
        print("Inside Folder", dirlist, "Consist :", len(filenames), "Imageset")
        for filename in filenames:
            if filename.endswith(".jpg") or filename.endswith(".JPG"):
                # print(os.path.join(root, filename), True)
                (noseX, noseY, noseZ,
                 left_eye_innerX, left_eye_innerY, left_eye_innerZ,
                 left_eyeX, left_eyeY, left_eyeZ,
                 left_eye_outerX, left_eye_outerY, left_eye_outerZ,
                 right_eye_innerX, right_eye_innerY, right_eye_innerZ,
                 right_eyeX, right_eyeY, right_eyeZ,
                 right_eye_outerX, right_eye_outerY, right_eye_outerZ,
                 left_earX, left_earY, left_earZ,
                 right_earX, right_earY, right_earZ,
                 mouth_leftX, mouth_leftY, mouth_leftZ,
                 mouth_rightX, mouth_rightY, mouth_rightZ,
                 left_shoulderX, left_shoulderY, left_shoulderZ,
                 right_shoulderX, right_shoulderY, right_shoulderZ,
                 left_elbowX, left_elbowY, left_elbowZ,
                 right_elbowX, right_elbowY, right_elbowZ,
                 left_wristX, left_wristY, left_wristZ,
                 right_wristX, right_wristY, right_wristZ,
                 left_pinkyX, left_pinkyY, left_pinkyZ,
                 right_pinkyX, right_pinkyY, right_pinkyZ,
                 left_indexX, left_indexY, left_indexZ,
                 right_indexX, right_indexY, right_indexZ,
                 left_thumbX, left_thumbY, left_thumbZ,
                 right_thumbX, right_thumbY, right_thumbZ,
                 left_hipX, left_hipY, left_hipZ,
                 right_hipX, right_hipY, right_hipZ,
                 left_kneeX, left_kneeY, left_kneeZ,
                 right_kneeX, right_kneeY, right_kneeZ,
                 left_ankleX, left_ankleY, left_ankleZ,
                 right_ankleX, right_ankleY, right_ankleZ,
                 left_heelX, left_heelY, left_heelZ,
                 right_heelX, right_heelY, right_heelZ,
                 left_foot_indexX, left_foot_indexY, left_foot_indexZ,
                 right_foot_indexX, right_foot_indexY, right_foot_indexZ,
                 annotated_image) = extract_feature(os.path.join(root, filename))

                if ((not noseX == 0) and (not noseY == 0)):
                    toCSV(csv_path, dirlist,
                          noseX, noseY, noseZ,
                          left_eye_innerX, left_eye_innerY, left_eye_innerZ,
                          left_eyeX, left_eyeY, left_eyeZ,
                          left_eye_outerX, left_eye_outerY, left_eye_outerZ,
                          right_eye_innerX, right_eye_innerY, right_eye_innerZ,
                          right_eyeX, right_eyeY, right_eyeZ,
                          right_eye_outerX, right_eye_outerY, right_eye_outerZ,
                          left_earX, left_earY, left_earZ,
                          right_earX, right_earY, right_earZ,
                          mouth_leftX, mouth_leftY, mouth_leftZ,
                          mouth_rightX, mouth_rightY, mouth_rightZ,
                          left_shoulderX, left_shoulderY, left_shoulderZ,
                          right_shoulderX, right_shoulderY, right_shoulderZ,
                          left_elbowX, left_elbowY, left_elbowZ,
                          right_elbowX, right_elbowY, right_elbowZ,
                          left_wristX, left_wristY, left_wristZ,
                          right_wristX, right_wristY, right_wristZ,
                          left_pinkyX, left_pinkyY, left_pinkyZ,
                          right_pinkyX, right_pinkyY, right_pinkyZ,
                          left_indexX, left_indexY, left_indexZ,
                          right_indexX, right_indexY, right_indexZ,
                          left_thumbX, left_thumbY, left_thumbZ,
                          right_thumbX, right_thumbY, right_thumbZ,
                          left_hipX, left_hipY, left_hipZ,
                          right_hipX, right_hipY, right_hipZ,
                          left_kneeX, left_kneeY, left_kneeZ,
                          right_kneeX, right_kneeY, right_kneeZ,
                          left_ankleX, left_ankleY, left_ankleZ,
                          right_ankleX, right_ankleY, right_ankleZ,
                          left_heelX, left_heelY, left_heelZ,
                          right_heelX, right_heelY, right_heelZ,
                          left_foot_indexX, left_foot_indexY, left_foot_indexZ,
                          right_foot_indexX, right_foot_indexY, right_foot_indexZ,)

                else:
                    print(os.path.join(root, filename), "Hand does not have landmarks")

print("===================Feature Extraction for TRAINING is Completed===================")