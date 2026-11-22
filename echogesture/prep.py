"""Build manifest + landmark cache from the public WLASL100 video shards (HF: aipieces/WLASL100).
  python -m echogesture.prep --raw data/raw --cache data/landmarks --workers 6
Landmarks are extracted once; training then never touches video again."""
from __future__ import annotations
import argparse, json, os
from multiprocessing import Pool
import numpy as np

_ex = None


def _work(job):
    global _ex
    from .landmarks import LandmarkExtractor
    _ex = _ex or LandmarkExtractor()
    vid, path, bbox, out = job
    if not os.path.exists(out):
        np.save(out, _ex.video(path, bbox=bbox))
    return vid


def build_manifest(raw: str):
    labels = {}
    for l in open(os.path.join(raw, "folder2label_str.txt")):
        i, g = l.split()
        labels[g] = int(i)
    meta = json.load(open(os.path.join(raw, "WLASL_v0.3.json")))
    items = []
    for e in meta:
        if e["gloss"] not in labels:
            continue
        for ins in e["instances"]:
            p = os.path.join(raw, "videos", str(labels[e["gloss"]]), f"{ins['video_id']}.mp4")
            if os.path.exists(p):
                items.append({"id": ins["video_id"], "label": labels[e["gloss"]], "split": ins["split"],
                              "path": p, "bbox": ins["bbox"]})
    vocab = [g.upper() for g, _ in sorted(labels.items(), key=lambda kv: kv[1])]
    return items, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--cache", default="data/landmarks")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    items, vocab = build_manifest(a.raw)
    json.dump({"items": items, "vocab": vocab}, open("data/manifest.json", "w"))
    print(f"{len(items)} clips, {len(vocab)} classes", {s: sum(i['split'] == s for i in items) for s in ('train', 'val', 'test')}, flush=True)
    jobs = [(i["id"], i["path"], i["bbox"], os.path.join(a.cache, f"{i['id']}.npy")) for i in items]
    with Pool(a.workers) as p:
        for n, _ in enumerate(p.imap_unordered(_work, jobs, chunksize=4)):
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
