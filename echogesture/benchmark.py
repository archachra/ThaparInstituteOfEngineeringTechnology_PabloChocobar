"""Measure Track-2 metrics and write results.csv (target vs measured, pass computed).
  python -m echogesture.benchmark --device "MacBook (Apple Silicon)"
Without a manifest, accuracy rows are left EMPTY (never fabricated); size/latency are still measured."""
from __future__ import annotations
import argparse, csv, os, platform, time
import numpy as np
from . import features as F
from .infer import Classifier

TARGETS = [("top1_accuracy", ">=", 70, "%"), ("latency_p95", "<=", 50, "ms"), ("model_size", "<=", 25, "MB"),
           ("accuracy_drop_robustness", "<=", 5, "pt"), ("avg_power", "<=", 500, "mW"),
           ("thermal_drift_10min", "<=", 20, "%")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=platform.platform()); ap.add_argument("--runtime", default="ONNX Runtime INT8 (CPU, 1 thread)")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--cache", default="data/landmarks"); ap.add_argument("--out", default="results.csv")
    ap.add_argument("--runs", type=int, default=500); ap.add_argument("--minutes", type=float, default=0.0)
    a = ap.parse_args()
    clf = Classifier(threads=1)
    x = np.random.default_rng(0).normal(size=(32, F.FEAT_DIM)).astype(np.float32)
    for _ in range(30): clf.logits(x)
    def lat_run(n):
        ts = []
        for _ in range(n):
            t = time.perf_counter(); clf.logits(x); ts.append((time.perf_counter() - t) * 1000)
        return np.array(ts)
    ts = lat_run(a.runs)
    m = {"latency_p95": round(float(np.percentile(ts, 95)), 3),
         "model_size": round(os.path.getsize(os.path.join(os.path.dirname(__file__), "..", "models", "signnet_int8.onnx")) / 1e6, 3)}
    if a.minutes > 0:  # thermal drift: p50 latency in first vs last minute of sustained load
        end, first, last, t0 = time.time() + a.minutes * 60, [], [], time.time()
        while time.time() < end:
            t = time.perf_counter(); clf.logits(x); d = (time.perf_counter() - t) * 1000
            (first if time.time() - t0 < 60 else last).append(d) if (time.time() - t0 < 60 or end - time.time() < 60) else None
        m["thermal_drift_10min"] = round(100 * (np.median(last) - np.median(first)) / np.median(first), 2)
    if os.path.exists(a.manifest):
        import json
        from .dataset import load_cached
        man = json.load(open(a.manifest)); vocab = json.load(open("models/signnet_vocab.json"))
        items = [(i["id"], i["label"]) for i in man["items"] if i["split"] == "test"]
        lms, ys = load_cached(a.cache, items)
        rng = np.random.default_rng(0)
        def acc(occ):
            ok = 0
            for l, y in zip(lms, ys):
                l2 = F.augment_landmarks(l, rng, occ) if occ else l
                ok += vocab[int(np.argmax(clf.logits(F.clip_features(l2))))] == man["vocab"][y].upper()
            return 100 * ok / max(len(ys), 1)
        base, occ = acc(0), acc(0.3)
        print(f"test clips: {len(ys)}")
        m["top1_accuracy"] = round(base, 2); m["accuracy_drop_robustness"] = round(base - occ, 2)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "target", "measured", "unit", "device", "runtime", "pass"])
        for name, op, tgt, unit in TARGETS:
            v = m.get(name)
            ok = "" if v is None else str((v >= tgt) if op == ">=" else (v <= tgt))
            w.writerow([name, f"{op}{tgt}", "" if v is None else v, unit, a.device, a.runtime, ok])
    print(open(a.out).read())


if __name__ == "__main__":
    main()
