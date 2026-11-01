import cv2
import mediapipe as mp
import numpy as np

from common import landmarks_to_features

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

history = []


def label(img, text, row, colour=(0, 255, 0)):
    cv2.putText(img, text, (10, 30 + row * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)                 # flip FIRST, keeps handedness right
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        raw = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        apparent = np.linalg.norm(raw[9] - raw[0])    # hand size as seen by the camera
        wrist_x = raw[0][0]                           # where the hand is in frame

        feats = landmarks_to_features(lm)
        norm = float(np.linalg.norm(feats))
        history.append(norm)
        if len(history) > 150:
            history.pop(0)
        spread = (max(history) - min(history)) / (np.mean(history) + 1e-9) * 100

        label(frame, f"apparent hand size : {apparent:.3f}", 0, (0, 200, 255))
        label(frame, f"wrist x in frame   : {wrist_x:.2f}", 1, (0, 200, 255))
        label(frame, f"||features||       : {norm:.2f}", 2)
        label(frame, f"drift over {len(history):3d} frames: {spread:.1f}%", 3)
    else:
        label(frame, "no hand", 0, (0, 0, 255))

    cv2.imshow("normalisation check  -  q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()