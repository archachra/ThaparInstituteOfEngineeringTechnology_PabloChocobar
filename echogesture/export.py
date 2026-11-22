"""PyTorch -> ONNX (fp32) -> static INT8 (QDQ) with calibration. Enforces the <20 MB budget.
  python -m echogesture.export --ckpt models/signnet.pt --calib data/landmarks
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np, torch
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from . import T_FRAMES, features as F
from .model import SignNet


class _Reader(CalibrationDataReader):
    def __init__(self, arrs):
        self.it = iter([{"features": a[None]} for a in arrs])
    def get_next(self):
        return next(self.it, None)


def calib_set(cache: str | None, n: int = 200):
    rng = np.random.default_rng(1)
    if cache and glob.glob(f"{cache}/*.npy"):
        files = sorted(glob.glob(f"{cache}/*.npy"))[:n]
        return [F.clip_features(np.load(f)) for f in files]
    from .synthetic import make_dataset
    _, (l, _), _ = make_dataset(8, 12, 1, rng)
    return [F.clip_features(x) for x in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/signnet.pt"); ap.add_argument("--calib", default="data/landmarks")
    ap.add_argument("--out-dir", default="models")
    a = ap.parse_args()
    classes = json.load(open(os.path.splitext(a.ckpt)[0] + "_vocab.json"))
    m = SignNet(len(classes)); m.load_state_dict(torch.load(a.ckpt)); m.eval()
    fp32 = os.path.join(a.out_dir, "signnet_fp32.onnx"); int8 = os.path.join(a.out_dir, "signnet_int8.onnx")
    torch.onnx.export(m, torch.zeros(1, T_FRAMES, F.FEAT_DIM), fp32, input_names=["features"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    quantize_static(fp32, int8, _Reader(calib_set(a.calib)), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
    mb = lambda p: os.path.getsize(p) / 1e6
    print(f"fp32 {mb(fp32):.2f} MB -> INT8 {mb(int8):.2f} MB  (budget 20 MB: {'OK' if mb(int8) < 20 else 'FAIL'})")


if __name__ == "__main__":
    main()
