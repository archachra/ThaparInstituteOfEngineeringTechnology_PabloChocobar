"""On-device landmark extraction with MediaPipe Tasks (hands + upper-body pose).

Output per frame is a (N_PTS, 4) array: x, y, z, visible-flag, in image-normalised coords.
Layout: [0:21] left hand, [21:42] right hand, [42:49] pose (nose, shoulders, elbows, wrists).
Nothing leaves the process: no network calls happen at inference time.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # nose, L/R shoulder, L/R elbow, L/R wrist
N_PTS = 21 + 21 + len(POSE_IDX)


class LandmarkExtractor:
    def __init__(self, max_side: int = 480):
        self.max_side = max_side
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=os.path.join(MODELS_DIR, "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.3, min_tracking_confidence=0.3))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=os.path.join(MODELS_DIR, "pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO))
        self._ts = 0
        self._off = 0  # keeps MediaPipe VIDEO-mode timestamps monotonic across clips

    def frame(self, bgr: np.ndarray, ts_ms: int | None = None) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self.max_side / max(h, w)
        if s < 1:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if ts_ms is None:
            self._ts += 33
            ts_ms = self._ts
        out = np.zeros((N_PTS, 4), np.float32)
        hr = self.hands.detect_for_video(img, ts_ms)
        for lms, hd in zip(hr.hand_landmarks, hr.handedness):
            base = 0 if hd[0].category_name == "Left" else 21
            out[base:base + 21] = [[p.x, p.y, p.z, 1.0] for p in lms]
        pr = self.pose.detect_for_video(img, ts_ms)
        if pr.pose_landmarks:
            lm = pr.pose_landmarks[0]
            out[42:] = [[lm[i].x, lm[i].y, lm[i].z, 1.0] for i in POSE_IDX]
        return out

    def video(self, path: str, max_frames: int = 300, bbox=None) -> np.ndarray:
        """Extract (T, N_PTS, 4) for a clip. bbox=(x1,y1,x2,y2) optionally crops to the signer.
        Returns zeros((1,N_PTS,4)) if unreadable."""
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames, i = [], 0
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = bgr[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
                if crop.size and crop.shape[0] > 32 and crop.shape[1] > 32:
                    bgr = crop
            frames.append(self.frame(bgr, self._off + int(i * 1000 / fps)))
            i += 1
        self._off += int(i * 1000 / fps) + 1000
        cap.release()
        return np.stack(frames) if frames else np.zeros((1, N_PTS, 4), np.float32)

    def close(self):
        self.hands.close(); self.pose.close()
