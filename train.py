import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import GESTURES, HandNet

torch.set_num_threads(1)          # 16,838 params: threads cost more than they save
torch.manual_seed(0)
np.random.seed(0)

FEATURE_COLS = [f"f{i}" for i in range(63)]
EPOCHS = 60
BATCH = 64
LR = 1e-3

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_sessions():
    """Read every data/raw/*.csv into {session_name: (X, y, persons)}."""
    out = {}
    for path in sorted(glob.glob(os.path.join("data", "raw", "*.csv"))):
        rows = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=0)
        if rows.ndim == 1 or len(rows) < 2:
            print(f"  skipping {path} - no data rows")
            continue
        header = list(rows[0])
        body = rows[1:]
        col = {name: i for i, name in enumerate(header)}
        feat_idx = [col[c] for c in FEATURE_COLS]

        X = body[:, feat_idx].astype(np.float32)
        labels = body[:, col["label"]]
        persons = body[:, col["person"]]
        y = np.array([GESTURES.index(l) for l in labels], dtype=np.int64)

        name = os.path.splitext(os.path.basename(path))[0]
        out[name] = (X, y, persons)
        print(f"  {name:6s}  {len(X):5d} rows  persons={sorted(set(persons))}")
    return out


def class_counts(y):
    return {GESTURES[i]: int((y == i).sum()) for i in range(len(GESTURES))}


def fit(X_tr, y_tr, X_te, y_te, epochs=EPOCHS, verbose=False):
    """Train a fresh HandNet. Returns (model, accuracy, seconds, predictions)."""
    model = HandNet(len(GESTURES))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    Xtr = torch.tensor(X_tr)
    ytr = torch.tensor(y_tr)
    Xte = torch.tensor(X_te)
    yte = torch.tensor(y_te)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            xb = Xtr[idx] + 0.01 * torch.randn_like(Xtr[idx])   # light noise
            opt.zero_grad()
            loss = lossf(model(xb), ytr[idx])
            loss.backward()
            opt.step()

        if verbose and (ep + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte).argmax(1) == yte).float().mean().item()
            print(f"    epoch {ep+1:3d}  loss {loss.item():.4f}  test acc {acc:.3f}")

    secs = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1)          # argmax -> class index, not value
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


def print_confusion(m):
    names = [g[:5] for g in GESTURES]
    print("\n    true \\ pred   " + "".join(f"{n:>7s}" for n in names))
    for i, row in enumerate(m):
        print(f"    {names[i]:12s}" + "".join(f"{v:7d}" for v in row))


# ---------------------------------------------------------------- main
print("loading sessions")
sessions = load_sessions()
if not sessions:
    raise SystemExit("no data found - run collect.py first")

X_all = np.concatenate([v[0] for v in sessions.values()])
y_all = np.concatenate([v[1] for v in sessions.values()])
print(f"\ntotal {len(X_all)} rows, class counts {class_counts(y_all)}")

results = {"n_total": int(len(X_all)), "class_counts": class_counts(y_all),
           "sessions": {k: int(len(v[0])) for k, v in sessions.items()}}

# ---- split 1: random rows (the dishonest one, reported for contrast) -----
print("\n[1/2] random row split (leaks between adjacent frames)")
perm = np.random.permutation(len(X_all))
cut = int(0.8 * len(X_all))
tr, te = perm[:cut], perm[cut:]
_, acc_rand, secs_rand, _ = fit(X_all[tr], y_all[tr], X_all[te], y_all[te])
print(f"      accuracy {acc_rand:.4f}  (n_test={len(te)})")
results["random_split"] = {"acc": acc_rand, "n_test": int(len(te)),
                           "train_seconds": round(secs_rand, 2)}

# ---- split 2: hold out a whole session (the honest one) ------------------
if len(sessions) >= 2:
    names = sorted(sessions)
    test_name = names[-1]
    X_te, y_te, _ = sessions[test_name]
    X_tr = np.concatenate([sessions[n][0] for n in names[:-1]])
    y_tr = np.concatenate([sessions[n][1] for n in names[:-1]])

    print(f"\n[2/2] session split - holding out {test_name}")
    print(f"      train {len(X_tr)} rows, test {len(X_te)} rows")
    print(f"      test class counts {class_counts(y_te)}")
    model, acc_sess, secs_sess, pred = fit(X_tr, y_tr, X_te, y_te, verbose=True)
    f1 = macro_f1(y_te, pred, len(GESTURES))
    print(f"\n      accuracy {acc_sess:.4f}   macro-F1 {f1:.4f}   "
          f"(n_test={len(y_te)})")
    cm = confusion(y_te, pred, len(GESTURES))
    print_confusion(cm)

    results["session_split"] = {
        "held_out": test_name, "acc": acc_sess, "macro_f1": f1,
        "n_train": int(len(X_tr)), "n_test": int(len(y_te)),
        "train_seconds": round(secs_sess, 2),
        "confusion": cm.tolist(), "labels": GESTURES,
    }
    print(f"\n      >> honest number: {acc_sess:.1%}  "
          f"vs {acc_rand:.1%} from the random split")
    print(f"      >> the {acc_rand - acc_sess:+.1%} gap IS the leakage")
else:
    print("\n[2/2] session split SKIPPED - only one session recorded.")
    print("      Record a second session; without it there is no honest number.")
    model = fit(X_all, y_all, X_all, y_all)[0]
    results["session_split"] = None

torch.save(model.state_dict(), os.path.join("models", "model.pt"))
json.dump(GESTURES, open(os.path.join("models", "labels.json"), "w"))
json.dump(results, open(os.path.join("results", "metrics.json"), "w"), indent=2)
print("\nsaved models/model.pt, models/labels.json, results/metrics.json")