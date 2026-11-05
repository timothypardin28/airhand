# airhand

Write Chinese characters (汉字) in the air with your finger and have a webcam
recognise them.

Final project for **AI in Practice** (Beijing Institute of Technology). The
course brief is in [`Final_Project_Brief.pdf`](Final_Project_Brief.pdf). In short:
the system has to run live, be compared against a simple baseline, and be
presented in a 15-minute talk with a demo. There is no code submission.

---

## Current status (read this first)

The repo currently holds a **static hand-gesture pipeline**, not air-writing
yet. It recognises 6 hand *poses* from a single frame (`fist`, `point`,
`peace`, `thumbs_up`, `thumbs_down`, `idle`) and maps them to keyboard/media
actions. This works as the foundation (camera, hand tracking, feature
normalisation, data recording, a small PyTorch model), but hanzi recognition
needs these extra pieces:

| Done | Component |
|:---:|---|
| ✅ | Environment pinned and smoke-tested (`verify_env.py`) |
| ✅ | Hand landmark detection with MediaPipe (21 points per hand) |
| ✅ | Position- and size-invariant pose features (`common.landmarks_to_features`) |
| ✅ | Live check that the normalisation works (`test_live.py`) |
| ✅ | Recorder for labelled pose frames (`collect.py`) |
| ⬜ | Recorded data. `data/raw/s01.csv` has the header only, no rows yet |
| ⬜ | Training script → `models/` |
| ⬜ | Live inference / demo script |
| ⬜ | **Air-writing:** pen-up/pen-down detection, fingertip trajectory capture |
| ⬜ | **Hanzi classifier** on the trajectory (sequence or rendered-image model) |
| ⬜ | Baseline comparison and evaluation → `results/` |

Keep this table up to date as work lands.

---

## How it works

### Planned pipeline (hanzi)

```
webcam frame
   │  OpenCV: read + mirror (cv2.flip)
   ▼
MediaPipe Hands ──► 21 landmarks (x, y, z), normalised to the image
   │
   ├─► pose of the hand ──► "pen down" (e.g. index finger only) / "pen up"
   │                         └─ reuses the existing gesture classifier
   │
   └─► index fingertip = landmark 8 ──► trajectory of points while pen is down
                                          │
                                          ▼
                       character classifier (sequence model on the stroke points,
                       or CNN on the trajectory drawn as an image)
                                          │
                                          ▼
                                   predicted 汉字 + confidence
```

The current gesture classifier fits naturally as the **pen-up / pen-down
switch**: one pose means "I'm writing", another means "stop / submit".

### What exists today (gestures)

1. **Capture**: OpenCV grabs a frame and mirrors it. The flip happens *before*
   MediaPipe so left/right handedness comes out correct.
2. **Landmarks**: MediaPipe Hands returns 21 3-D points
   ([landmark map](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)).
   Landmark 0 is the wrist, 8 the index fingertip, 9 the middle-finger knuckle.
3. **Features** (`common.landmarks_to_features`): subtract the wrist (removes
   *where* the hand is), divide by the wrist→knuckle-9 distance (removes *how
   big* it looks). Output is 63 floats (21 × xyz) that describe hand shape only.
4. **Model** (`common.HandNet`): MLP `63 → 128 → 64 → N`, about 16.8k
   parameters for 6 classes. It's small enough to run on a laptop CPU in real time.
5. **Action** (`common.ACTIONS`): gesture → key name (e.g. `point` → `right`
   for next slide).

> ⚠️ For air-writing, *do not* use the wrist-normalised features for the
> trajectory. That normalisation removes exactly the motion we need. Track the
> raw image coordinates of landmark 8 (then normalise the *whole trajectory*
> for position/scale once the character is finished).

---

## Repository layout

```
airhand/
├── common.py              # shared code: GESTURES, ACTIONS, landmarks_to_features(), HandNet
├── collect.py             # webcam recorder → appends labelled feature rows to data/raw/<session>.csv
├── test_live.py           # live sanity check: is the feature vector stable as the hand moves/scales?
├── verify_env.py          # day-0 smoke test: versions, webcam, FPS, bundled hand model
├── requirements.txt       # pinned deps (Windows, CPU-only torch)
├── Final_Project_Brief.pdf
├── data/
│   └── raw/               # one CSV per recording session (s01.csv, s02.csv, …)
├── models/                # trained weights go here   (empty)
├── results/               # metrics, confusion matrices, plots (empty)
└── demo/                  # demo script / screen recordings (empty)
```

Empty folders are kept in git with a `.gitkeep` file.

### File details

**`common.py`** is imported by every other script. If you change the feature
function or the class list, old CSVs and trained models become incompatible,
so re-record or retrain.

**`collect.py`** is the recorder. Edit the three constants at the top before each sitting:

```python
PERSON  = "A"    # who is recording
SESSION = "s01"  # one id per sitting — never reuse one
CAMERA  = 0      # 0 = built-in webcam, 1 = external
```

Controls: `1`–`6` choose a gesture, `SPACE` starts a 3 s countdown and then
records 150 frames, `q` quits. The file is appended to, so you can stop and resume;
the on-screen counts are reloaded from the CSV.

CSV format: `person, session, hand, label, f0 … f62`.

Keep sessions separate so you can split train/test **by session or person**,
not by random frames. Neighbouring frames are nearly identical, so a random split
inflates accuracy.

**`test_live.py`** shows apparent hand size, wrist position, the feature-vector
norm and its drift over the last 150 frames. Move your hand closer/farther and
around the frame while holding one pose. The drift % should stay small, which
shows the normalisation works.

**`verify_env.py`** prints library versions, opens the webcam, runs MediaPipe
on 60 frames and reports FPS and detection rate. Run it on every team laptop.

---

## Setup

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python verify_env.py
```

`requirements.txt` targets **Windows**. It pins `torch==...+cpu` and includes
Windows-only packages (`pycaw`, `comtypes`, `PyGetWindow`).

**macOS notes**
- Install a local `mac_requirements.txt` instead (same pins without the
  Windows-only packages and with plain `torch`). It is git-ignored, so each Mac
  user keeps their own.
- The scripts open the camera with `cv2.CAP_DSHOW`, a Windows DirectShow
  backend. On a Mac, if the camera doesn't open, change it to
  `cv2.VideoCapture(0)` (or `cv2.CAP_AVFOUNDATION`). A cross-platform fix in
  the code would be better.
- macOS will ask for camera permission for your terminal/IDE the first time.

---

## Typical workflow

```bash
python verify_env.py     # once per machine
python test_live.py      # check features are stable
python collect.py        # record data (edit PERSON / SESSION first)
# next: train → models/, evaluate → results/, live demo → demo/
```

---

## Conventions

- Run scripts from the repo root (paths like `data/raw/` are relative).
- Never reuse a session id. Never edit recorded CSVs by hand.
- Report real numbers only. The brief treats edited results as misconduct.
- Cite MediaPipe, any datasets (e.g. CASIA-OLHWDB for online handwriting),
  and AI-assistant use in the slides, as the brief requires.

## Team

| Member | Role |
|---|---|
| Timothy Pardin | scaffold, environment, recorder |
| Cornelius | data recording |
| Andrew Zhuo | |

*(fill in roles; teamwork is 15% of the grade)*
