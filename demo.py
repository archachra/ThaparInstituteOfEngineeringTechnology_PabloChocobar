"""Live conversation HUD (Ghost Mode).   python demo.py [--airgap] [--speak] [--video clip.mp4]
q quits.  Camera frames never leave RAM and are never displayed or saved."""
import argparse, collections, subprocess, sys, shutil, time
import cv2, numpy as np
from echogesture import features as F, hud
from echogesture.infer import Classifier
from echogesture.landmarks import LandmarkExtractor, N_PTS

ap = argparse.ArgumentParser()
ap.add_argument("--airgap", action="store_true"); ap.add_argument("--speak", action="store_true")
ap.add_argument("--video"); ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--window", type=int, default=45); ap.add_argument("--min-conf", type=float, default=0.5)
a = ap.parse_args()
if a.airgap:
    from echogesture import airgap; airgap.enable()
    from echogesture.airgap import blocked_attempts
else:
    blocked_attempts = lambda: 0
clf, ex = Classifier(), LandmarkExtractor()
cap = cv2.VideoCapture(a.video if a.video else a.cam)
buf = collections.deque(maxlen=a.window)
word, conf, top5, lat, last_spoken, last_t = "", 0.0, [], 0.0, "", time.time()
fps = 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    lm = ex.frame(frame, int(time.time() * 1000)); del frame        # raw frame dropped immediately
    buf.append(lm)
    if len(buf) >= 16 and (lm[0, 3] + lm[21, 3]) > 0 and len(buf) % 5 == 0:
        t = time.perf_counter()
        words, probs = clf.topk(F.clip_features(np.stack(buf)))
        lat = (time.perf_counter() - t) * 1000
        if probs[0] >= a.min_conf:
            word, conf, top5 = words[0].upper(), probs[0], [w.upper() for w in words]
            if a.speak and word != last_spoken and shutil.which("say"):
                subprocess.Popen(["say", word.lower()]); last_spoken = word
    now = time.time(); fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3); last_t = now
    cv2.imshow("EchoGesture", hud.draw(lm, word, conf, top5, lat, fps, blocked=blocked_attempts()))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
