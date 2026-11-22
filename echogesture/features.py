"""Pose-invariant feature engineering (pure numpy, shared by train + inference).

Invariances: translation + scale (shoulder-width units, mid-shoulder origin), hand shape is
wrist-relative and scaled by hand size, so signer distance / position / camera angle matter less.
Missing hands/pose (occlusion) are zero-filled with explicit presence flags.
"""
from __future__ import annotations
import numpy as np
from . import T_FRAMES

PER_FRAME = 2 * (63 + 3) + 21 + 3  # hands(shape+wrist pos) + pose + 3 presence flags = 156
FEAT_DIM = PER_FRAME * 2           # + temporal deltas


def _hand_shape(h: np.ndarray) -> np.ndarray:
    rel = h[:, :3] - h[0, :3]
    size = np.linalg.norm(h[9, :2] - h[0, :2]) + 1e-6  # wrist -> middle MCP
    return (rel / size).reshape(-1)


def frame_features(f: np.ndarray) -> np.ndarray:
    pose, sh_l, sh_r = f[42:], f[43], f[44]
    has_pose = pose[:, 3].min() > 0
    if has_pose:
        origin = (sh_l[:3] + sh_r[:3]) / 2
        scale = np.linalg.norm(sh_l[:2] - sh_r[:2]) + 1e-6
    else:  # fall back to hand-only normalisation
        origin, scale = np.array([0.5, 0.5, 0.0], np.float32), 0.3
    parts, flags = [], []
    for base in (0, 21):
        h = f[base:base + 21]
        if h[0, 3] > 0:
            parts.append(_hand_shape(h))
            parts.append(((h[0, :3] - origin) / scale))
            flags.append(1.0)
        else:
            parts.append(np.zeros(66, np.float32)); flags.append(0.0)
    pose_f = ((pose[:, :3] - origin) / scale).reshape(-1) if has_pose else np.zeros(21, np.float32)
    flags.append(1.0 if has_pose else 0.0)
    return np.concatenate(parts + [pose_f, np.array(flags, np.float32)]).astype(np.float32)


def resample(x: np.ndarray, t: int = T_FRAMES) -> np.ndarray:
    idx = np.linspace(0, len(x) - 1, t).round().astype(int)
    return x[idx]


def clip_features(lm: np.ndarray, t: int = T_FRAMES) -> np.ndarray:
    """(T_raw, N_PTS, 4) -> (t, FEAT_DIM). Trims leading/trailing frames with no hands."""
    active = np.where((lm[:, 0, 3] + lm[:, 21, 3]) > 0)[0]
    if len(active) >= 4:
        lm = lm[active[0]:active[-1] + 1]
    f = np.stack([frame_features(x) for x in lm])
    f = resample(f, t)
    d = np.vstack([np.zeros((1, f.shape[1]), np.float32), np.diff(f, axis=0)])
    return np.concatenate([f, d], axis=1).astype(np.float32)


def augment_landmarks(lm: np.ndarray, rng: np.random.Generator, occlude: float = 0.0) -> np.ndarray:
    """Training augmentation on raw landmarks: rotation, scale, shift, jitter, drop-outs."""
    lm = lm.copy()
    ang = rng.normal(0, 0.12)
    c, s = np.cos(ang), np.sin(ang)
    xy = lm[..., :2] - 0.5
    lm[..., 0] = xy[..., 0] * c - xy[..., 1] * s + 0.5
    lm[..., 1] = xy[..., 0] * s + xy[..., 1] * c + 0.5
    lm[..., :3] *= rng.normal(1, 0.08)
    lm[..., :2] += rng.normal(0, 0.03, size=2)
    lm[..., :3] += rng.normal(0, 0.004, size=lm[..., :3].shape)
    drop = rng.random(len(lm)) < 0.08           # random frame dropouts
    lm[drop] = lm[np.clip(np.where(drop)[0] - 1, 0, None)]
    if occlude > 0:                              # random per-frame hand occlusion
        for base in (0, 21):
            m = rng.random(len(lm)) < occlude
            lm[m, base:base + 21] = 0
    return lm
