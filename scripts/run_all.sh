#!/bin/bash
# Train -> INT8 export -> benchmark -> deck. Run from EchoGesture/ with the venv active.
set -e
python -m echogesture.train --epochs 80
python -m echogesture.export --calib data/landmarks
python -m echogesture.benchmark --device "${DEVICE:-$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)}"
python scripts/build_deck.py "../Submission_Final_forGithub.pptx" results.csv docs/ThaparInstituteOfEngineeringTechnology_PabloChocobar.pptx
