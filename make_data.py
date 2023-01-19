import cv2
import mediapipe as mp
import numpy as np
import os

video_path = './new_video'
file_list = os.listdir(video_path)
print('Class_list : ',file_list)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

suit = file_list
seq_length = 30

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

for idx, fol in enumerate(suit):
    data = []
    pse = []
    path = file_list[idx]
    file_list2 = os.listdir('new_video/%s' % path)
    print('Video_list : ', file_list2)
    for num in range(len(file_list2)):
        print('In Cam : ', "./new_video/%s/%s" % (file_list[idx], file_list2[num]))
        cap = cv2.VideoCapture("./new_video/%s/%s" % (file_list[idx], file_list2[num]))
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
                    angle_pose = np.arccos(np.einsum('nt,nt->n',
                                                     v[[0, 1, 2, 2, 2, 6, 7, 8, 8, 8], :],
                                                     v[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11],
                                                     :]))  # [15,]

                    angle_pose = np.degrees(angle_pose)

                    angle_label_pose = np.array([angle_pose], dtype=np.float32)
                    angle_label_pose = np.append(angle_label_pose, idx)
                    new_point = joint_pose.flatten()

                    d_p = np.concatenate([new_point, angle_label_pose])

                    data.append(d_p)

                mp_drawing.draw_landmarks(img, result.face_landmarks, landmark_drawing_spec=None, )
                mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                cv2.imshow('HandTracking', img)

                if cv2.waitKey(1) & 0xFF == 27:  # esc 키를 누르면 닫음
                    break
            else:
                break

    data = np.array(data)
    print(len(data.shape))
    print(fol, data.shape)
    np.save(os.path.join('dataset/raw', f'raw_p{fol}_{seq_length}'), data)

    # Create sequence data
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(fol, full_seq_data.shape)
    np.save(os.path.join('dataset/seq', f'seq_p{fol}_{seq_length}'), full_seq_data)

