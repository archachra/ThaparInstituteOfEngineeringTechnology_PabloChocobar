import numpy as np, subprocess, sys, os
from echogesture import features as F
from echogesture.synthetic import make_dataset


def test_features_shape_and_invariance():
    rng = np.random.default_rng(0)
    _, (l, _), _ = make_dataset(2, 1, 1, rng)
    a = F.clip_features(l[0])
    assert a.shape == (32, F.FEAT_DIM)
    shifted = l[0].copy(); shifted[..., :2] += 0.1; shifted[..., :3] *= 1.0
    b = F.clip_features(shifted)
    assert np.abs(a[:, :F.PER_FRAME] - b[:, :F.PER_FRAME]).max() < 1e-3  # translation invariant


def test_occlusion_zero_fills():
    rng = np.random.default_rng(0)
    _, (l, _), _ = make_dataset(2, 1, 1, rng)
    o = l[0].copy(); o[:, :42] = 0
    assert np.isfinite(F.clip_features(o)).all()


def test_airgap_blocks():
    from echogesture import airgap
    import socket, pytest
    airgap.enable()
    with pytest.raises(airgap.AirgapViolation):
        socket.create_connection(("example.com", 80))
