import json
import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import torch

from common import ACTIONS, GESTURES, HandNet, landmarks_to_features

# ---- tuning knobs: this table goes on slide 7 ---------------------------
CONF_MIN = 0.90       # minimum softmax confidence to accept a frame
VOTE_N = 9            # frames in the sliding window
NEED_VOTES = 7        # agreeing frames required to fire
MOTION_MAX = 0.05     # max feature jitter across the window.
                      # held pose ~0.01, hand moving while talking ~0.3
COOLDOWN = 1.2        # seconds before another action can fire
CAMERA = 0            # 0 = external, 1 = built-in (yours is reversed)
SLIDES_ONLY = False   # start in full mode; press V to toggle
VOLUME_GESTURES = {"thumbs_up", "thumbs_down"}
# --------------------------------------------------------------------------

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
feat_hist = deque(maxlen=VOTE_N)
armed = False
last_fire = 0.0
last_action = "-"
fired_total = 0
blocked_by_motion = 0
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
        feats = landmarks_to_features(lm)
        feat_hist.append(feats)
        x = torch.tensor(feats).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        c, idx = probs.max(0)
        conf = float(c)
        if conf >= CONF_MIN:
            name = labels[int(idx)]
    else:
        feat_hist.clear()               # no hand -> idle by rule
    recent.append(name)

    # how much the hand shape has wobbled across the window
    if len(feat_hist) == VOTE_N:
        motion = float(np.mean(np.std(np.array(feat_hist), axis=0)))
    else:
        motion = 9.99
    stable = motion < MOTION_MAX

    valid = [v for v in recent if v is not None]
    vote, votes = (Counter(valid).most_common(1)[0] if valid else (None, 0))

    now = time.time()
    agreed = bool(vote) and votes >= NEED_VOTES and now - last_fire > COOLDOWN
    if armed and agreed and not stable:
        blocked_by_motion += 1
    if armed and agreed and stable:
        action = ACTIONS.get(vote)
        if SLIDES_ONLY and vote in VOLUME_GESTURES:
            action = None               # disabled in slides-only mode
        if action:
            pyautogui.press(action)
            last_fire = now
            last_action = f"{vote} -> {action}"
            fired_total += 1
            print(f"  fired {vote:<12s} -> {action:<12s} conf {conf:.2f}  "
                  f"votes {votes}/{VOTE_N}  motion {motion:.3f}  "
                  f"{(now - t_frame) * 1000:.0f} ms")

    frame_times.append(time.time() - t_frame)
    fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
    shown = armed_seconds + (now - t_armed_since if armed and t_armed_since else 0)
    rate = fired_total / (shown / 60) if shown > 5 else 0.0

    put(frame, "ARMED" if armed else "DISARMED", 0,
        (0, 0, 255) if armed else (160, 160, 160), 0.9)
    put(frame, f"{name or '-':<12s} conf {conf:.2f}", 1)
    put(frame, f"vote {vote or '-':<12s} {votes}/{VOTE_N}", 2, (0, 200, 255))
    put(frame, f"motion {motion:5.3f}  {'HELD' if stable else 'MOVING'}", 3,
        (0, 255, 0) if stable else (0, 100, 255))
    put(frame, "SLIDES ONLY" if SLIDES_ONLY else "FULL (slides + volume)",
        6, (255, 200, 0) if SLIDES_ONLY else (160, 160, 160), 0.5)
    put(frame, f"last: {last_action}", 4, (200, 200, 200), 0.5)
    put(frame, f"fired {fired_total}  blocked {blocked_by_motion}  "
               f"armed {shown:.0f}s  {rate:.2f}/min  {fps:.0f} fps",
        5, (200, 200, 200), 0.5)

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
        feat_hist.clear()
        print("ARMED" if armed else "DISARMED")
    if k == ord("r"):
        fired_total, blocked_by_motion, armed_seconds = 0, 0, 0.0
        t_armed_since = time.time() if armed else None
        print("counters reset")
    if k == ord("v"):
        SLIDES_ONLY = not SLIDES_ONLY
        print("SLIDES ONLY" if SLIDES_ONLY else "FULL MODE")

if armed and t_armed_since:
    armed_seconds += time.time() - t_armed_since
cap.release()
cv2.destroyAllWindows()

print(f"\nfired            : {fired_total}")
print(f"blocked by motion: {blocked_by_motion}")
print(f"armed time       : {armed_seconds:.1f} s")
if armed_seconds > 5:
    print(f"rate             : {fired_total / (armed_seconds / 60):.2f} per minute")