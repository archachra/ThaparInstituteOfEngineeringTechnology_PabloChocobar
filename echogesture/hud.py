"""Ghost-Mode HUD: Samsung One UI x Apple Dynamic Island styling.
The camera image is NEVER drawn or stored -- only an abstract glowing skeleton. Frames are
dropped right after landmark extraction (privacy by construction)."""
from __future__ import annotations
import time
import cv2, numpy as np

BG = (22, 18, 18); GLASS = (48, 42, 40); TEXT = (245, 245, 247); DIM = (160, 150, 150)
ACCENT = (255, 132, 10)   # BGR -> iOS-style blue (10,132,255)
GREEN = (89, 199, 52)
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
              (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
POSE_EDGES = [(1,2),(1,3),(3,5),(2,4),(4,6),(0,1),(0,2)]


def _pill(img, x, y, w, h, color, alpha=0.85):
    ov = img.copy()
    r = h // 2
    cv2.rectangle(ov, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.circle(ov, (x + r, y + r), r, color, -1); cv2.circle(ov, (x + w - r, y + r), r, color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _text(img, s, x, y, scale=0.7, color=TEXT, thick=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw(lm: np.ndarray, word: str, conf: float, top5, lat_ms: float, fps: float, size=(960, 600),
         trail=None, blocked=0, listening=True):
    W, H = size
    img = np.full((H, W, 3), BG, np.uint8)
    # soft vignette
    yy, xx = np.ogrid[:H, :W]
    v = 1 - 0.5 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img = (img * np.clip(v, 0.3, 1)[..., None]).astype(np.uint8)
    glow = np.zeros_like(img)
    P = lambda p: (int(p[0] * W), int(p[1] * H))
    for base, edges in ((0, HAND_EDGES), (21, HAND_EDGES), (42, POSE_EDGES)):
        pts = lm[base:base + (21 if base < 42 else 7)]
        for a, b in edges:
            if pts[a, 3] > 0 and pts[b, 3] > 0:
                cv2.line(glow, P(pts[a]), P(pts[b]), ACCENT, 7, cv2.LINE_AA)
        for p in pts:
            if p[3] > 0:
                cv2.circle(glow, P(p), 6, ACCENT, -1, cv2.LINE_AA)
    img = cv2.addWeighted(img, 1, cv2.GaussianBlur(glow, (0, 0), 14), 1.6, 0)
    sharp = np.zeros_like(img)
    for base, edges in ((0, HAND_EDGES), (21, HAND_EDGES), (42, POSE_EDGES)):
        pts = lm[base:base + (21 if base < 42 else 7)]
        for a, b in edges:
            if pts[a, 3] > 0 and pts[b, 3] > 0:
                cv2.line(sharp, P(pts[a]), P(pts[b]), TEXT, 2, cv2.LINE_AA)
    m = sharp.any(2); img[m] = sharp[m]

    # Dynamic-Island caption
    cw = int(min(W - 80, 260 + 34 * len(word)))
    cx = (W - cw) // 2
    _pill(img, cx, 28, cw, 64, (0, 0, 0), 0.92)
    if word:
        _text(img, word, cx + 30, 70, 1.0, TEXT, 2)
        cv2.ellipse(img, (cx + cw - 38, 60), (14, 14), -90, 0, int(360 * conf), GREEN if conf > .6 else ACCENT, 4, cv2.LINE_AA)
        _text(img, f"{int(conf*100)}", cx + cw - 50, 65, 0.4, DIM)
    else:
        _text(img, "Sign something...", cx + 30, 68, 0.8, DIM)
    # top-5 chips
    x = 28
    for i, w in enumerate(top5[1:5]):
        tw = cv2.getTextSize(w, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 28
        _pill(img, x, 108, tw, 30, GLASS, 0.9); _text(img, w, x + 14, 129, 0.5, DIM); x += tw + 8
    # bottom status glass
    _pill(img, 28, H - 70, W - 56, 44, GLASS, 0.9)
    cv2.circle(img, (56, H - 48), 6, GREEN, -1, cv2.LINE_AA)
    _text(img, f"ON-DEVICE  |  0 B uploaded  |  video not stored  |  net blocked: {blocked}", 72, H - 42, 0.45, TEXT)
    _text(img, f"{lat_ms:.1f} ms  {fps:.0f} fps", W - 190, H - 42, 0.45, DIM)
    return img
