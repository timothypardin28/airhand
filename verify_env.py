import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

print(f"python      {sys.version.split()[0]}")
print(f"numpy       {np.__version__}")
print(f"opencv      {cv2.__version__}")
print(f"mediapipe   {mp.__version__}")
print(f"torch       {torch.__version__}")
# False is correct here — the demo runs on CPU on both laptops
print(f"cuda available: {torch.cuda.is_available()}")

# proves there is nothing to download from Google
tflite = os.path.join(os.path.dirname(mp.__file__),
                      "modules", "hand_landmark", "hand_landmark_full.tflite")
print(f"bundled hand model: {os.path.exists(tflite)}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # DSHOW avoids a 3s black first frame
if not cap.isOpened():
    sys.exit("FAIL: cannot open webcam 0 — close Teams / Zoom / browser tabs, then retry")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=1,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

frames = found = n_landmarks = 0
t0 = time.time()
while frames < 60:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)             # flip FIRST — keeps handedness correct
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        found += 1
        n_landmarks = len(res.multi_hand_landmarks[0].landmark)
    frames += 1
elapsed = time.time() - t0
cap.release()
hands.close()

print(f"camera      {w}x{h}")
print(f"fps         {frames / elapsed:.1f}   (over {frames} frames)")
print(f"hand seen   {found}/{frames} frames, {n_landmarks} landmarks")

if found == 0:
    print("NOTE: no hand detected. Hold your hand in frame and run it again.")
else:
    print("OK — this machine is ready.")