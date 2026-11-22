"""Synthetic landmark clips so the whole pipeline (train->INT8->predict) can be smoke-tested
without any dataset. NOT used for any reported accuracy number."""
import numpy as np
from .landmarks import N_PTS


def _clip(c: int, rng, t=40):
    lm = np.zeros((t, N_PTS, 4), np.float32)
    ph = np.linspace(0, 1, t)[:, None]
    base = np.random.default_rng(c).normal(0, 0.05, (21, 3)); base2 = np.random.default_rng(c + 99).normal(0, 0.05, (21, 3))
    for k, b in ((0, base), (21, base2)):
        path = np.stack([0.5 + 0.15 * np.sin(2 * np.pi * (ph[:, 0] * (1 + c % 3)) + c),
                         0.5 + 0.15 * np.cos(2 * np.pi * ph[:, 0] * (1 + c % 2) + c), 0 * ph[:, 0]], 1)
        lm[:, k:k + 21, :3] = path[:, None, :] + b[None]
        lm[:, k:k + 21, 3] = 1
    lm[:, 42:, :3] = [[0.5, 0.3, 0], [0.4, 0.45, 0], [0.6, 0.45, 0], [0.35, 0.6, 0], [0.65, 0.6, 0], [0.3, 0.75, 0], [0.7, 0.75, 0]]
    lm[:, 42:, 3] = 1
    return lm + rng.normal(0, 0.003, lm.shape).astype(np.float32) * (np.arange(4) < 3)


def make_dataset(n_cls, n_tr, n_va, rng):
    classes = [f"WORD{i}" for i in range(n_cls)]
    mk = lambda n: ([_clip(c, rng) for c in range(n_cls) for _ in range(n)], [c for c in range(n_cls) for _ in range(n)])
    return classes, mk(n_tr), mk(n_va)
