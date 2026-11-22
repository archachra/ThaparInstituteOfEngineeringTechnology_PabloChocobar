from __future__ import annotations
import json, os, time
import numpy as np
import onnxruntime as ort
from . import features as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Classifier:
    def __init__(self, onnx_path=None, vocab_path=None, threads: int = 1):
        onnx_path = onnx_path or os.path.join(ROOT, "models", "signnet_int8.onnx")
        vocab_path = vocab_path or os.path.join(ROOT, "models", "signnet_vocab.json")
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.vocab = json.load(open(vocab_path))

    def logits(self, feats: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {"features": feats[None].astype(np.float32)})[0][0]

    def topk(self, feats: np.ndarray, k: int = 5):
        lg = self.logits(feats)
        p = np.exp(lg - lg.max()); p /= p.sum()
        idx = np.argsort(-p)[:k]
        return [self.vocab[i] for i in idx], [float(p[i]) for i in idx]
