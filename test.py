import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model

video_path = './new_video'
file_list = os.listdir(video_path)
print('Class_list : ',file_list)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

suit = file_list
seq_length = 30

model = load_model('models/model2_2.1.0.h5')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

seq = []
action_seq = []

cap = cv2.VideoCapture("./test_video/test_video3.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('video_Result/video_Result4.mp4', fourcc, 30.0, (int(width), int(height)))

while cap.isOpened():
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = holistic.process(imgRGB)
        if result.pose_landmarks is not None:

            res_pose = result.pose_landmarks
            joint_pose = np.zeros((33, 4))
            for j, lm in enumerate(res_pose.landmark):
                joint_pose[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint_pose[[11, 12, 12, 14, 16, 16, 16, 24, 12, 11, 11, 13, 15, 15, 15, 23], :]
            v2 = joint_pose[[12, 14, 24, 16, 18, 20, 22, 26, 11, 13, 23, 15, 17, 19, 21, 25], :]
            # 0-왼어깨-오른어꺠 1-오른어깨-팔꿈치 2 팔꿈치-손목 3-5 손목-각 손가락
            # 6-오른어깨-왼어꺠 7-왼어깨-팔꿈치 8 팔꿈치-손목 9-11 손목-각 손가락
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                                             v[[0, 1, 2, 2, 2, 6, 7, 8, 8, 8], :],
                                             v[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11],
                                             :]))  # [15,]

            angle = np.degrees(angle)
            new_point = joint_pose.flatten()

            d_p = np.concatenate([new_point, angle])

            seq.append(d_p)
            if len(seq) > 30:
                mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                y_pred = model.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                action = suit[i_pred]
                action_seq.append(action)

                if conf > 0.8:
                    this_action = action
                else :
                    this_action = 'NOT GOOD POSE'

                for i in range(len(suit)):
                    if y_pred[i] == y_pred[i_pred] :
                        cv2.putText(img, f'{suit[i]} : %d%%' % (y_pred[i] * 100),
                                    org=(int(10), int(30) + i * 31),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)
                    else :
                        cv2.putText(img, f'{suit[i]} : %d%%' %(y_pred[i]*100),
                                    org=(int(10), int(30)+i*30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)

        cv2.imshow('Dance', img)
        out.write(img)
        if cv2.waitKey(1) & 0xFF == 27:  # esc 키를 누르면 닫음
            break
    else:
        break