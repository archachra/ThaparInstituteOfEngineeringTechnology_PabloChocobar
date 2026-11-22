# EchoGesture: on-device sign-language word prediction (XR Hackathon 2026, Track 2)

A camera sees a signer; the phone/laptop itself says the word. **No cloud, no server, no stored video.**

```
camera -> MediaPipe hands+pose (49 landmarks/frame, frame discarded) -> pose-invariant features
       -> INT8 SignNet (ONNX Runtime, 1 CPU thread) -> word + top-5 -> caption / speech
```

**Team Pablo Chocobar**, Thapar Institute of Engineering & Technology: (Parth Puri, Arnav Chachra, Hasrat Aulakh)

## 📌 Submission Deliverables
- 📊 **Presentation / Pitch Deck:** [`PPT_and_Video/ThaparInstituteOfEngineeringTechnology_PabloChocobar.pptx`](PPT_and_Video/ThaparInstituteOfEngineeringTechnology_PabloChocobar.pptx)
- 🎥 **Demo Video:** [`PPT_and_Video/EchoGesture_Demo.mp4`](PPT_and_Video/EchoGesture_Demo.mp4)

## Results (measured, `results.csv`)
Test split of WLASL100 (258 held-out clips, 100 words), Apple M4 MacBook CPU, ONNX Runtime INT8, 1 thread.

| metric | target | measured | pass |
|---|---|---|---|
| top-1 accuracy | >= 70 % | **55.0 %** | no |
| latency p95 (model inference) | <= 50 ms | **0.15 ms** | yes |
| model size (INT8) | <= 25 MB | **0.46 MB** | yes |
| accuracy drop, 30 % hand occlusion | <= 5 pt | **3.1 pt** | yes |
| avg power | <= 500 mW | not measured (needs a phone) | - |
| 10-min thermal drift | <= 20 % | not measured | - |

Honest notes: accuracy is below target (small data: ~14 clips per word); landmark extraction costs ~20 ms/frame
on CPU and is reported separately as `landmark_ms_per_frame` in `predict()` output; power/thermal need a device.

## Unique: Ghost Mode + Airgap Proof
- **Ghost Mode** (`demo.py`): the HUD never shows or saves the camera image, only a glowing skeleton of the signer.
- **Airgap Proof** (`--airgap`): every network call raises an error. If it still translates, nothing used the cloud.

## Setup (Python 3.12)
```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
Models are included: `models/signnet_int8.onnx`, `models/signnet_vocab.json`, `models/*.task` (MediaPipe, Google, Apache-2.0).

## Run
```bash
python predict.py --video clip.mp4                                   # one clip -> JSON
python predict.py --manifest test_manifest.csv --videos clips/ --out predictions.jsonl --airgap
python demo.py --airgap --speak                                      # live webcam HUD (q to quit)
docker build -t echogesture . && docker run -v $PWD/clips:/app/clips echogesture
```
`predict(video_path) -> {"clip_id","prediction","top5","confidence","latency_ms"}` (+ `landmark_ms_per_frame`).

## Reproduce training
Data: public WLASL100 videos (HF `aipieces/WLASL100`, derived from WLASL, Li et al. WACV 2020).
```bash
# put shards in data/raw/ (videos/, WLASL_v0.3.json, folder2label_str.txt), then:
python -m echogesture.prep --workers 6                 # landmarks cache + manifest
python -m echogesture.train --epochs 80 --width 192 --drop 0.5 --wd 0.05 --mix 0.4 --occ 0.25
python -m echogesture.export --calib data/landmarks    # fp32 ONNX -> static INT8 (QDQ), size check
python -m echogesture.benchmark --device "Your device" # writes results.csv
python -m pytest -q tests
```

## Layout
`echogesture/` landmarks, features, model, train, export, infer, benchmark, hud, airgap · `predict.py` · `demo.py`
· `PPT_and_Video/ThaparInstituteOfEngineeringTechnology_PabloChocobar.pptx` · `PPT_and_Video/EchoGesture_Demo.mp4` (demo video, rebuild: `python scripts/build_video.py`) · `AI_DISCLOSURE.md` · `results.csv` · `predictions.jsonl` (6-clip sample)

## Limitations
Isolated words only (not sentences); 100-word vocabulary; trained on public WLASL, not the organisers' test set;
no power/thermal measurement yet. See `AI_DISCLOSURE.md` for AI tools and data used.
