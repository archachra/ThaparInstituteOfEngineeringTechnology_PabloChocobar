# AI Disclosure

## AI tools used in building this project
- **Claude Code (Anthropic, model Claude Sonnet 5.5)** was used as a coding assistant to write most of the
  source in this repository (landmark extraction, feature pipeline, model, training, INT8 export,
  `predict()`, benchmark script, Ghost-Mode HUD, tests, Dockerfile) and to draft the README and the
  submission deck content. All code was run and tested by the team; results in `results.csv` come from
  scripts in this repo, not from the assistant's claims.

## AI models / pretrained assets inside the product
- **MediaPipe Hand Landmarker and Pose Landmarker (Google, Apache-2.0)**: pretrained landmark models, used
  unmodified, run fully on-device (`models/*.task`).
- **Microsoft Edge neural TTS (en-IN-NeerjaExpressiveNeural)**: generated the demo-video narration (script text only was sent).
- **SignNet**: our own small temporal CNN, trained from scratch on landmarks (no pretrained weights).

## Data
- **WLASL** (Li et al., WACV 2020), via the public Hugging Face mirror `aipieces/WLASL100`: 100-word subset,
  used for training/validation/testing with the dataset's own splits. Please cite WLASL if reusing.
- No cloud AI service is called at inference time. The `--airgap` flag proves this by blocking all sockets.

## Human contribution
- Team: problem selection, project direction (privacy-first "Ghost Mode" and "Airgap Proof" concept),
  review of outputs, and the presentation/demo.
- Prior work reused: an earlier Sign-Link prototype (rule-based gesture mapper, FastAPI landmark service)
  from a previous hackathon; only the WLASL class list file is carried into this repo.
