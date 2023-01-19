import cv2
import mediapipe as mp
import numpy as np

import os
video_path = './video'
file_list = os.listdir(video_path)
print('Class_list : ',file_list)

max_num_hands = 1
suit = file_list
actions = file_list

seq_length = 30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)
for fol in range(len(suit)):
    data = []
    for num in range(1,10):
        cap = cv2.VideoCapture("지문자/%s/%s_%d.mp4" % (suit[fol], suit[fol], num))
        mpHands = mp.solutions.hands
        my_hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils
        while cap.isOpened():
            ret, img = cap.read()

            if ret:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(imgRGB)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]

                        v = v2 - v1
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                        angle = np.arccos(np.einsum('nt,nt->n',
                                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                    :]))  # [15,]

                        angle = np.degrees(angle)  # Convert radian to degree


                        for po in range(1,21):
                            if ((po%4)==0):
                                point_label = np.array([res.landmark[po].x, res.landmark[po].y, res.landmark[po].z])

                                point_label_end = np.append(point_label, po)


                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, fol)

                        d = np.concatenate([joint.flatten(), point_label_end, angle_label])

                        data.append(d)




                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('HandTracking', img)

                if cv2.waitKey(1) & 0xFF == 27:  # esc 키를 누르면 닫음
                    break

            else:
                break

    data = np.array(data)
    print(len(data.shape))
    print(suit[fol], data.shape)
    np.save(os.path.join('dataset', f'raw_p{suit[fol]}_{9}'), data)

    # Create sequence data
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(suit[fol], full_seq_data.shape)
    np.save(os.path.join('dataset', f'seq_p{suit[fol]}_{9}'), full_seq_data)