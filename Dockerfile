FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir numpy opencv-python-headless mediapipe onnxruntime
COPY . .
# Batch inference, network disabled inside the process:
ENTRYPOINT ["python", "predict.py", "--airgap"]
CMD ["--manifest", "test_manifest.csv", "--videos", "clips", "--out", "predictions.jsonl"]
