import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from common import GESTURES, landmarks_to_features

# ---- edit these three before each recording session ----------------------
PERSON = "A"          # A = Pardin, B = Cornelius, C = Andrew
SESSION = "s01"       # s01, s02, ... one per sitting. NEVER reuse an id.
CAMERA = 0            # 0 = built-in, 1 = external webcam

TARGET = 150          # frames to capture per burst
COUNTDOWN = 3.0       # seconds to get into position

PATH = os.path.join("data", "raw", f"{SESSION}.csv")
os.makedirs(os.path.dirname(PATH), exist_ok=True)

# write the header only when the file is new, so you can stop and resume
if not os.path.exists(PATH):
    with open(PATH, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["person", "session", "hand", "label"] + [f"f{i}" for i in range(63)])

# count what is already in the file, so the HUD shows real totals on resume
counts = {g: 0 for g in GESTURES}
with open(PATH, newline="") as fh:
    for row in csv.DictReader(fh):
        if row["label"] in counts:
            counts[row["label"]] += 1

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)

selected = 0          # index into GESTURES
state = "idle"        # idle -> countdown -> recording
t_start = 0.0
captured = 0
fh = open(PATH, "a", newline="")
writer = csv.writer(fh)


def label(img, text, row, colour=(0, 255, 0), scale=0.6):
    cv2.putText(img, text, (10, 28 + row * 26),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    lm = None
    hand_label = "?"
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        if res.multi_handedness:
            hand_label = res.multi_handedness[0].classification[0].label

    # state machine
    if state == "countdown":
        left = COUNTDOWN - (time.time() - t_start)
        if left <= 0:
            state, captured = "recording", 0
        else:
            label(frame, f"{left:.1f}", 4, (0, 165, 255), 1.6)

    elif state == "recording":
        if lm is not None:
            writer.writerow([PERSON, SESSION, hand_label, GESTURES[selected]]
                            + [f"{v:.6f}" for v in landmarks_to_features(lm)])
            counts[GESTURES[selected]] += 1
            captured += 1
        if captured >= TARGET:
            state = "idle"
            fh.flush()
        label(frame, f"RECORDING {captured}/{TARGET}", 4, (0, 0, 255), 0.9)

    # HUD
    label(frame, f"{PERSON} / {SESSION} / {hand_label} hand", 0, (0, 200, 255))
    label(frame, f"selected: {GESTURES[selected]}  [{selected + 1}]", 1)
    label(frame, "  ".join(f"{i+1}:{g[:5]}={counts[g]}"
                           for i, g in enumerate(GESTURES)), 2, (200, 200, 200), 0.5)
    label(frame, "1-6 pick   SPACE record   q quit", 3, (160, 160, 160), 0.5)

    cv2.imshow("collect  -  q to quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if ord("1") <= key <= ord("6") and state == "idle":
        selected = key - ord("1")
    if key == ord(" ") and state == "idle":
        state, t_start = "countdown", time.time()

fh.close()
cap.release()
cv2.destroyAllWindows()
print("\nfinal counts:", counts)
print("saved to", PATH)