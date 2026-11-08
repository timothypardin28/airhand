import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import GESTURES, HandNet

torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)

FEATURE_COLS = [f"f{i}" for i in range(63)]
EPOCHS = 150
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4      # mild pull toward smaller weights
ROT_DEG = 15.0           # random 3D rotation applied to each training batch
NOISE = 0.02

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_sessions():
    out = {}
    for path in sorted(glob.glob(os.path.join("data", "raw", "*.csv"))):
        rows = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=0)
        if rows.ndim == 1 or len(rows) < 2:
            continue
        header, body = list(rows[0]), rows[1:]
        col = {name: i for i, name in enumerate(header)}
        X = body[:, [col[c] for c in FEATURE_COLS]].astype(np.float32)
        y = np.array([GESTURES.index(l) for l in body[:, col["label"]]],
                     dtype=np.int64)
        name = os.path.splitext(os.path.basename(path))[0]
        out[name] = (X, y, body[:, col["person"]])
        print(f"  {name:6s}  {len(X):5d} rows  persons={sorted(set(out[name][2]))}")
    return out


def random_rotations(n, max_deg):
    """n random small 3D rotation matrices, as an (n, 3, 3) tensor.

    The features are 21 wrist-centred, scale-normalised points, so rotating
    them is physically meaningful: it is the same hand seen from a slightly
    different angle. That is exactly the variation a new person or a new
    camera position introduces, so it should help the held-out number rather
    than the training one.
    """
    r = torch.deg2rad((torch.rand(n, 3) * 2 - 1) * max_deg)
    cx, cy, cz = torch.cos(r).T
    sx, sy, sz = torch.sin(r).T
    one, zero = torch.ones(n), torch.zeros(n)
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], 1).view(n, 3, 3)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], 1).view(n, 3, 3)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], 1).view(n, 3, 3)
    return Rx @ Ry @ Rz


def augment(xb, max_deg=ROT_DEG, noise=NOISE):
    """Rotate each sample in the batch, then add light noise."""
    n = xb.shape[0]
    pts = xb.view(n, 21, 3)
    pts = torch.bmm(pts, random_rotations(n, max_deg).transpose(1, 2))
    out = pts.reshape(n, 63)
    return out + noise * torch.randn_like(out)


def class_counts(y):
    return {GESTURES[i]: int((y == i).sum()) for i in range(len(GESTURES))}


def fit(X_tr, y_tr, X_te, y_te, epochs=EPOCHS, verbose=False):
    model = HandNet(len(GESTURES))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss()
    Xtr, ytr = torch.tensor(X_tr), torch.tensor(y_tr)
    Xte, yte = torch.tensor(X_te), torch.tensor(y_te)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            xb = augment(Xtr[idx])
            opt.zero_grad()
            loss = lossf(model(xb), ytr[idx])
            loss.backward()
            opt.step()
        if verbose and (ep + 1) % 40 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte).argmax(1) == yte).float().mean().item()
            print(f"    epoch {ep+1:3d}  loss {loss.item():.4f}  test acc {acc:.3f}")

    secs = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1)
        acc = (pred == yte).float().mean().item()
    return model, acc, secs, pred.numpy()


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


def confusion(y_true, y_pred, n):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


print("loading sessions")
sessions = load_sessions()
names = sorted(sessions)
if len(names) < 2:
    raise SystemExit("need at least 2 sessions")

test_name = names[-1]
X_te, y_te, _ = sessions[test_name]
X_tr = np.concatenate([sessions[n][0] for n in names[:-1]])
y_tr = np.concatenate([sessions[n][1] for n in names[:-1]])

print(f"\nholding out {test_name}: train {len(X_tr)}, test {len(X_te)}")
print(f"augmentation: +/-{ROT_DEG} deg rotation, noise {NOISE}, "
      f"weight decay {WEIGHT_DECAY}, {EPOCHS} epochs\n")

model, acc, secs, pred = fit(X_tr, y_tr, X_te, y_te, verbose=True)
f1 = macro_f1(y_te, pred, len(GESTURES))
print(f"\n      accuracy {acc:.4f}   macro-F1 {f1:.4f}   (n_test={len(y_te)})")
print(f"      trained in {secs:.1f}s")

cm = confusion(y_te, pred, len(GESTURES))
short = [g[:5] for g in GESTURES]
print("\n    true \\ pred   " + "".join(f"{n:>7s}" for n in short))
for i, row in enumerate(cm):
    print(f"    {short[i]:12s}" + "".join(f"{v:7d}" for v in row))

json.dump({"held_out": test_name, "acc": acc, "macro_f1": f1,
           "n_test": int(len(y_te)), "train_seconds": round(secs, 2),
           "augmentation": {"rot_deg": ROT_DEG, "noise": NOISE,
                            "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS},
           "confusion": cm.tolist(), "labels": GESTURES},
          open(os.path.join("results", "metrics_augmented.json"), "w"), indent=2)

torch.save(model.state_dict(), os.path.join("models", "model_aug.pt"))
print("\nsaved models/model_aug.pt, results/metrics_augmented.json")
print(f"\nCompare with the 85.0% from train.py. Difference: "
      f"{acc - 0.85:+.1%}")