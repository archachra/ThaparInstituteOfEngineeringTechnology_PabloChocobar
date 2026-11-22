"""Train SignNet on cached landmarks.
  python -m echogesture.prep && python -m echogesture.train --epochs 80
  python -m echogesture.train --synthetic            # smoke test, no dataset needed
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from . import features as F
from .dataset import load_cached
from .model import SignNet
from .synthetic import make_dataset


def build_xy(lms, ys, rng, aug, occlude=0.0):
    X = [F.clip_features(F.augment_landmarks(l, rng, occlude) if aug else l) for l in lms]
    return torch.tensor(np.stack(X)), torch.tensor(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--cache", default="data/landmarks"); ap.add_argument("--out", default="models/signnet.pt")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--width", type=int, default=192); ap.add_argument("--drop", type=float, default=0.3); ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--mix", type=float, default=0.0); ap.add_argument("--occ", type=float, default=0.15); ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0); torch.manual_seed(0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    if a.synthetic:
        classes, (tl, ty), (vl, vy) = make_dataset(8, 24, 8, rng)
    else:
        man = json.load(open(a.manifest))
        classes = [w.upper() for w in man["vocab"]]
        def split(name):
            its = [(i["id"], i["label"]) for i in man["items"] if i["split"] == name]
            return load_cached(a.cache, its)
        (tl, ty), (vl, vy) = split("train"), split("val")
        print(f"train {len(tl)} clips, val {len(vl)} clips, {len(classes)} classes", flush=True)
    model = SignNet(len(classes), a.width, a.drop)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=a.epochs * (len(tl) // 32 + 1))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xv, Yv = build_xy(vl, vy, rng, False)
    best = 0.0
    for ep in range(a.epochs):
        model.train()
        X, Y = build_xy(tl, ty, rng, True, occlude=a.occ)
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            b = perm[i:i + 32]
            if len(b) < 2: continue
            xb, yb = X[b], Y[b]
            opt.zero_grad()
            if a.mix > 0:  # mixup
                lam = float(np.random.beta(a.mix, a.mix)); j = torch.randperm(len(b))
                out = model(lam * xb + (1 - lam) * xb[j])
                loss = lam * lossf(out, yb) + (1 - lam) * lossf(out, yb[j])
            else:
                loss = lossf(model(xb), yb)
            loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == Yv).float().mean().item()
        if acc >= best:
            best = acc; torch.save(model.state_dict(), a.out)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:3d}  val top-1 {acc*100:5.1f}%  (best {best*100:.1f}%)", flush=True)
    json.dump(classes, open(os.path.splitext(a.out)[0] + "_vocab.json", "w"))
    print(f"saved {a.out}; classes={len(classes)}; best val top-1={best*100:.1f}%")


if __name__ == "__main__":
    main()
