import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import load_model
#클래스 값
classes = {
    'arm_dance1_1': 0,
    'arm_dance1_2': 1,
    'arm_dance1_3': 2,
    'arm_dance2_1': 3,
    'arm_dance2_2': 4,
    'arm_dance3_1': 5,
    'arm_dance5_1': 6,
    'arm_dance5_2': 7,
    'breath1_1': 8,
    'breath1_2': 9,
    'breath2_1': 10,
    'breath2_2': 11,
    'foot_dance1_1': 12,
    'foot_dance1_2': 13,
    'foot_dance2_1': 14,
    'foot_dance2_2': 15,
    'foot_dance3_1': 16,
    'foot_dance3_2': 17,
    'foot_dance4_1': 18,
    'foot_dance4_2': 19,
    'foot_dance4_3': 20,
    'foot_dance5_1': 21,
    'foot_dance5_2': 22,
}

#기본값
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
video_path = 'video/body_old/연습_팔사위2_kor_몸전체.mp4'
img_path = 'img_cp/new/foot_dance1_1/00000.jpg'

#모델
csv = pd.read_csv('fitness_poses_csvs_out/Dance_training.csv')
model = load_model('model_SIBI.h5')

#카메라
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
temp = []
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
            break

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        landmarks = results.pose_landmarks.landmark
        for j in landmarks:
            temp = temp + [j.x, j.y, j.z, j.visibility]
        y = model.predict([temp])
        print(y)
        assan = classes[y]
        cv2.putText(image, assan,
                    org=(50,50), fontFace=cv2.FONT_ITALIC, fontScale=1,color=(255,0,0), thickness=2)
        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()