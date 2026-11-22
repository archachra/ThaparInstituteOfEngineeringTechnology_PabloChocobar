"""WLASL loader. Expects the standard layout:
  <root>/nslt_{100,300,1000,2000}.json   (subset + class index per video id)
  <root>/videos/<id>.mp4
and models/wlasl_class_list.txt ("idx<TAB>gloss"). Landmarks are cached to <cache>/<id>.npy.
"""
from __future__ import annotations
import json, os
import numpy as np


def load_classes(path: str) -> list[str]:
    out = {}
    for line in open(path):
        i, g = line.rstrip("\n").split("\t")
        out[int(i)] = g.upper()
    return [out[i] for i in range(len(out))]


def load_split(root: str, nslt: int, subset: str) -> list[tuple[str, int]]:
    meta = json.load(open(os.path.join(root, f"nslt_{nslt}.json")))
    return [(vid, m["action"][0]) for vid, m in meta.items() if m["subset"] == subset]


def cache_landmarks(extractor, root: str, items, cache: str, log_every: int = 50):
    os.makedirs(cache, exist_ok=True)
    for n, (vid, _) in enumerate(items):
        out = os.path.join(cache, f"{vid}.npy")
        src = os.path.join(root, "videos", f"{vid}.mp4")
        if os.path.exists(out) or not os.path.exists(src):
            continue
        np.save(out, extractor.video(src))
        if n % log_every == 0:
            print(f"  cached {n}/{len(items)}", flush=True)


def load_cached(cache: str, items):
    xs, ys = [], []
    for vid, y in items:
        p = os.path.join(cache, f"{vid}.npy")
        if os.path.exists(p):
            xs.append(np.load(p)); ys.append(y)
    return xs, ys
