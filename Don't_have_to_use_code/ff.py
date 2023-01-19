import cv2
import mediapipe as mp
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

max_num_hands = 1
suit = ['개', '경찰', '계단', '달', '방망이', '벌', '병원', '붕대', '선생님', '아기', '아파트', '어지러움', '엘리베이터', '유리', '음식물', '자동차', '장난감', '체온계', '친구', '화장실']

seq_length = 30

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

for idx, fol in enumerate(suit):
    data = []
    pse = []
    for num in range(1,31):
        cap = cv2.VideoCapture("./video/a/202209-25-0.mp4")
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

                    v1 = joint_pose[[11, 12, 14, 16, 16, 16, 12, 11, 13, 15, 15, 15], :]
                    v2 = joint_pose[[12, 14, 16, 18, 20, 22, 11, 13, 15, 17, 19, 21], :]
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
                    for i in range(131, 88, -1):
                        new_point = np.delete(new_point, i, axis=0)

                    #for i in range(131 , 90, -1):
                        #new_point = np.delete(new_point, i, axis=0)

                    #print(len(new_point))

                    d_p = np.concatenate([new_point, angle_label_pose])

                    #print(len(d_p))


                    '''for i in range(144 , 103, -1):
                        d_p = np.delete(d_p, i, axis=0)'''


                    data.append(d_p)

                #data.append(point_label_p_end)

                mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(img, result.face_landmarks, landmark_drawing_spec=None, )
                mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())



                cv2.imshow('HandTracking', img)

                if cv2.waitKey(1) & 0xFF == 27:  # esc 키를 누르면 닫음
                    break

            else:
                break

    #print(data_p.shape)

    #data = np.concatenate((data_r, data_l, data_p))
    data = np.array(data)
    print(len(data.shape))
    print(fol, data.shape)
    np.save(os.path.join('dataset3', f'raw_p{fol}_{30}'), data)

    # Create sequence data
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(fol, full_seq_data.shape)
    np.save(os.path.join('dataset3', f'seq_p{fol}_{30}'), full_seq_data)