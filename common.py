import numpy as np
import torch
import torch.nn as nn

GESTURES = ["fist", "point", "peace", "thumbs_up", "thumbs_down", "idle"]

ACTIONS = {
    "fist":        "playpause",
    "point":       "right",        # next slide
    "peace":       "left",         # previous slide
    "thumbs_up":   "volumeup",
    "thumbs_down": "volumedown",
    "idle":        None,           # recognised, but deliberately does nothing
}


def landmarks_to_features(hand_landmarks):
    """21 landmarks -> 63 translation- and scale-invariant floats.

    Raw landmark coordinates encode WHERE the hand is in the frame and HOW BIG
    it appears. Neither of those is the gesture, so we remove both:
      - subtracting the wrist makes the features independent of position
      - dividing by the wrist-to-knuckle length makes them independent of size
    What survives is hand shape alone.
    """
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark],
                   dtype=np.float32)
    pts = pts - pts[0]                    # landmark 0 is the wrist -> origin
    scale = np.linalg.norm(pts[9])        # landmark 9 is the middle knuckle
    scale = scale or 1.0                  # a collapsed detection gives 0
    pts = pts / scale
    return pts.reshape(-1)                # (21, 3) -> (63,)

def get_device():
    """Return the best available device for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"

class HandNet(nn.Module):
    """63 -> 128 -> 64 -> N. Three linear layers, two ReLUs. 16,838 params at N=6."""

    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(63, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)