import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from common import GESTURES

FEATURE_COLS = [f"f{i}" for i in range(63)]


def load_sessions():
    """Read every data/raw/*.csv into {session_name: (X, y, persons)}."""
    import glob
    out = {}
    for path in sorted(glob.glob(os.path.join("data", "raw", "*.csv"))):
        rows = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=0)
        if rows.ndim == 1 or len(rows) < 2:
            continue
        header = list(rows[0])
        body = rows[1:]
        col = {name: i for i, name in enumerate(header)}
        X = body[:, [col[c] for c in FEATURE_COLS]].astype(np.float32)
        y = np.array([GESTURES.index(l) for l in body[:, col["label"]]],
                     dtype=np.int64)
        name = os.path.splitext(os.path.basename(path))[0]
        out[name] = (X, y, body[:, col["person"]])
        print(f"  {name:6s}  {len(X):5d} rows")
    return out


def macro_f1(y_true, y_pred, n):
    f1s = []
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


os.makedirs("results", exist_ok=True)

TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
THUMB_TIP = 4


def finger_rule(x):
    """No learning at all: count extended fingers, then map counts to gestures.

    A finger is 'extended' when its tip is further from the wrist than its PIP
    joint. Works for fist / point / peace. Fails on thumbs_up vs thumbs_down,
    because that difference is thumb ORIENTATION, not finger count - which is
    exactly the gap that justifies the MLP.
    """
    p = x.reshape(21, 3)
    n_ext = sum(1 for f in TIPS
                if np.linalg.norm(p[TIPS[f]]) > np.linalg.norm(p[PIPS[f]]))
    thumb_y = p[THUMB_TIP][1]          # y grows downward in image coords
    if n_ext == 0:
        if thumb_y < -0.6:
            return GESTURES.index("thumbs_up")
        if thumb_y > 0.6:
            return GESTURES.index("thumbs_down")
        return GESTURES.index("fist")
    if n_ext == 1:
        return GESTURES.index("point")
    if n_ext == 2:
        return GESTURES.index("peace")
    return GESTURES.index("idle")


print("loading sessions")
sessions = load_sessions()
names = sorted(sessions)
if len(names) < 2:
    raise SystemExit("need at least 2 sessions - record s02 before running this")

test_name = names[-1]
X_te, y_te, _ = sessions[test_name]
X_tr = np.concatenate([sessions[n][0] for n in names[:-1]])
y_tr = np.concatenate([sessions[n][1] for n in names[:-1]])
print(f"\nheld out {test_name}: train {len(X_tr)}, test {len(X_te)}\n")

rows = []

pred = np.array([finger_rule(x) for x in X_te])
rows.append(("Finger-count rule (no ML)", float((pred == y_te).mean()),
             macro_f1(y_te, pred, len(GESTURES)), 0.0))

t0 = time.time()
lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
t_lr = time.time() - t0
pred = lr.predict(X_te)
rows.append(("Logistic regression", float((pred == y_te).mean()),
             macro_f1(y_te, pred, len(GESTURES)), t_lr))

t0 = time.time()
knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
t_knn = time.time() - t0
pred = knn.predict(X_te)
rows.append(("kNN (k=5)", float((pred == y_te).mean()),
             macro_f1(y_te, pred, len(GESTURES)), t_knn))

mlp = None
try:
    m = json.load(open(os.path.join("results", "metrics.json")))
    if m.get("session_split"):
        s = m["session_split"]
        rows.append(("MLP 63-128-64-6", s["acc"], s["macro_f1"],
                     s["train_seconds"]))
        mlp = s["acc"]
except FileNotFoundError:
    print("(run train.py first to include the MLP row)\n")

print(f"{'Model':<28}{'Acc':>8}{'MacroF1':>10}{'Train s':>10}{'n_test':>8}")
print("-" * 64)
for name, acc, f1, secs in rows:
    print(f"{name:<28}{acc:>8.4f}{f1:>10.4f}{secs:>10.2f}{len(y_te):>8}")

json.dump({"held_out": test_name, "n_test": int(len(y_te)),
           "rows": [{"model": n, "acc": a, "macro_f1": f, "train_seconds": s}
                    for n, a, f, s in rows]},
          open(os.path.join("results", "baselines.json"), "w"), indent=2)
print("\nsaved results/baselines.json")

if mlp is not None:
    best_simple = max(r[1] for r in rows[:-1])
    print(f"\nMLP {mlp:.1%} vs best simpler method {best_simple:.1%} "
          f"-> {mlp - best_simple:+.1%}")