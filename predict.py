"""Track 2 contract:  predict(video_path) -> dict   (runs fully on-device)

CLI (batch):  python predict.py --manifest test_manifest.csv --videos clips/ --out predictions.jsonl [--airgap]
              python predict.py --video clip.mp4
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import numpy as np
from echogesture import features as F

_state = {}


def _get():
    if not _state:
        from echogesture.infer import Classifier
        from echogesture.landmarks import LandmarkExtractor
        _state["clf"], _state["ex"] = Classifier(), LandmarkExtractor()
    return _state["clf"], _state["ex"]


def predict(video_path: str) -> dict:
    clf, ex = _get()
    t0 = time.perf_counter()
    lm = ex.video(video_path)
    t1 = time.perf_counter()
    feats = F.clip_features(lm)
    words, probs = clf.topk(feats, 5)
    t2 = time.perf_counter()
    return {
        "clip_id": os.path.splitext(os.path.basename(video_path))[0],
        "prediction": words[0].upper(),
        "top5": [w.upper() for w in words],
        "confidence": round(probs[0], 4),
        # model inference latency (features + INT8 classifier) per the <50 ms budget;
        "latency_ms": round((t2 - t1) * 1000, 2),
        # landmark extraction reported separately (amortised per frame) for transparency
        "landmark_ms_per_frame": round((t1 - t0) * 1000 / max(len(lm), 1), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--manifest"); ap.add_argument("--videos", default=".")
    ap.add_argument("--out", default="predictions.jsonl"); ap.add_argument("--airgap", action="store_true")
    a = ap.parse_args()
    if a.airgap:
        from echogesture import airgap; airgap.enable(); print("[airgap] network disabled", file=sys.stderr)
    if a.video:
        print(json.dumps(predict(a.video), indent=2)); return
    rows = list(csv.DictReader(open(a.manifest)))
    key = "clip_id" if "clip_id" in rows[0] else list(rows[0])[0]
    with open(a.out, "w") as f:
        for r in rows:
            cid = r[key]
            p = next((os.path.join(a.videos, cid + e) for e in (".mp4", ".mov", ".avi", ".webm", "")
                      if os.path.isfile(os.path.join(a.videos, cid + e))), None)
            res = predict(p) if p else {"clip_id": cid, "prediction": "", "top5": [], "confidence": 0.0, "latency_ms": 0.0}
            res["clip_id"] = cid  # must match the manifest exactly
            f.write(json.dumps(res) + "\n")
    print(f"wrote {len(rows)} predictions -> {a.out}")


if __name__ == "__main__":
    main()
