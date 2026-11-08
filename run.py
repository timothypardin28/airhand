import json
import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import pyautogui
import torch

from common import ACTIONS, GESTURES, HandNet, landmarks_to_features

CONF_MIN = 0.85       # ignore frames the model is unsure about
VOTE_N = 7            # frames in the sliding window
NEED_VOTES = 5        # agreeing frames required to fire
COOLDOWN = 1.0        # seconds before another action can fire
CAMERA = 0

pyautogui.FAILSAFE = True          # slam the mouse into a corner to abort

labels = GESTURES
if os.path.exists(os.path.join("models", "labels.json")):
    labels = json.load(open(os.path.join("models", "labels.json")))

model = HandNet(len(labels))
model.load_state_dict(torch.load(os.path.join("models", "model.pt")))
model.eval()
torch.set_num_threads(1)

hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=1,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("cannot open webcam - close Teams/Zoom/browser tabs")

recent = deque(maxlen=VOTE_N)
armed = False
last_fire = 0.0
last_action = "-"
fired_total = 0
t_armed_since = None
armed_seconds = 0.0
frame_times = deque(maxlen=60)


def put(img, text, row, colour=(0, 255, 0), scale=0.6):
    cv2.putText(img, text, (10, 28 + row * 26),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


print("running. a = arm/disarm, r = reset counters, q = quit")

while True:
    t_frame = time.time()
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)          # flip FIRST, keeps handedness correct
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    name, conf = None, 0.0
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
        x = torch.tensor(landmarks_to_features(lm)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        c, idx = probs.max(0)
        conf = float(c)
        if conf >= CONF_MIN:
            name = labels[int(idx)]
    recent.append(name)

    valid = [v for v in recent if v is not None]
    vote, votes = (Counter(valid).most_common(1)[0] if valid else (None, 0))

    now = time.time()
    if armed and vote and votes >= NEED_VOTES and now - last_fire > COOLDOWN:
        action = ACTIONS.get(vote)
        if action:
            pyautogui.press(action)
            last_fire = now
            last_action = f"{vote} -> {action}"
            fired_total += 1
            print(f"  fired {vote:<12s} -> {action:<12s} conf {conf:.2f}  "
                  f"votes {votes}/{VOTE_N}  {(now - t_frame) * 1000:.0f} ms")

    frame_times.append(time.time() - t_frame)
    fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
    shown = armed_seconds + (now - t_armed_since if armed and t_armed_since else 0)
    rate = fired_total / (shown / 60) if shown > 5 else 0.0

    put(frame, "ARMED" if armed else "DISARMED", 0,
        (0, 0, 255) if armed else (160, 160, 160), 0.9)
    put(frame, f"{name or '-':<12s} conf {conf:.2f}", 1)
    put(frame, f"vote {vote or '-':<12s} {votes}/{VOTE_N}", 2, (0, 200, 255))
    put(frame, f"last: {last_action}", 3, (200, 200, 200), 0.5)
    put(frame, f"fired {fired_total}   armed {shown:.0f}s   "
               f"{rate:.2f}/min   {fps:.0f} fps", 4, (200, 200, 200), 0.5)

    cv2.imshow("airhand  -  a=arm  r=reset  q=quit", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("a"):
        if armed:
            armed_seconds += time.time() - t_armed_since
            t_armed_since = None
        else:
            t_armed_since = time.time()
        armed = not armed
        recent.clear()
        print("ARMED" if armed else "DISARMED")
    if k == ord("r"):
        fired_total, armed_seconds = 0, 0.0
        t_armed_since = time.time() if armed else None

if armed and t_armed_since:
    armed_seconds += time.time() - t_armed_since
cap.release()
cv2.destroyAllWindows()
print(f"\nfired {fired_total} in {armed_seconds:.1f}s armed")
if armed_seconds > 5:
    print(f"rate: {fired_total / (armed_seconds / 60):.2f} per minute")