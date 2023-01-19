import csv
import cv2
import numpy as np
import os
import sys
import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import time, os
import glob
import sys

video_Dir = 'train_video/body/연습_팔사위5_kor_몸전체_동작부분2.mp4'
output_frame_folder = 'img_cp/new/arm_dance5_2'
index=len(video_Dir)
images_out_folder = 'result_video/'
csv_out_path = 'fitness_poses_csvs_out/fitness_poses_csvs_out.csv'

for video in glob.glob(video_Dir):
    print("video:"+video)
    cap = cv2.VideoCapture(video)
    created_time = int(time.time())
    start_time = time.time()
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not os.path.exists(output_frame_folder):
        os.makedirs(output_frame_folder)
    for framenum in range(0, vid_length):
        print(framenum)
        cap.set(cv2.CAP_PROP_FRAME_COUNT, framenum)
        ret, frame = cap.read()
        if ret is False:
            break
        # Image Processing
        cv2.imwrite(output_frame_folder + '/' + str(framenum).zfill(5) + '.jpg', frame)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:  # Escape (ESC)
            break
cap.release()
cv2.destroyAllWindows()

