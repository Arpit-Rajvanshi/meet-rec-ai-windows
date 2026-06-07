# =========================================
# Meet Recorder - Full Pipeline GPU Image
#   Whisper + Pyannote + SpeechBrain (ECAPA-TDNN)
#   R2 upload/download, transcript.txt output
# =========================================

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# ----- System deps -----
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# ----- Python deps -----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "filelock>=3.12.2" && \
    pip install --no-cache-dir -r requirements.txt

# ----- Application code -----
# Core pipeline modules
COPY transcription.py        .
COPY diarize.py              .
COPY media_io.py             .
COPY speaker_mapper_voice.py .
COPY transcript_writer.py    .
COPY r2_client.py            .
COPY runpod_handler.py       .

# Baked-in biometric enrollment DB
# (Temporary: will be loaded from intranet/R2 in a future iteration)
COPY seed_data.json          .

# ----- Runtime env -----
ENV PYTHONUNBUFFERED=1
# Persistent volume on RunPod for model caches
ENV HF_HOME=/runpod-volume/huggingface
ENV TORCH_HOME=/runpod-volume/torch
# SpeechBrain cache (also lands on persistent volume)
ENV SPEECHBRAIN_CACHE=/runpod-volume/speechbrain/spkrec-ecapa-voxceleb
# Default seed DB location inside the image
ENV SEED_DB_PATH=/app/seed_data.json

# ----- Start -----
CMD ["python", "-u", "runpod_handler.py"]
