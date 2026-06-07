# =============================================
# meet-recorder  —  GPU pipeline image
# Base already contains: torch 2.2.2, torchaudio 2.2.2, CUDA 12.1
# We add: faster-whisper, pyannote, speechbrain, boto3
# =============================================

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# --- System deps (one layer, clean up in same RUN) ---
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# --- Upgrade pip once ---
RUN pip install --no-cache-dir --upgrade pip

# --- Layer 1: heavy compiled deps first (best cache reuse) ---
# Install pyannote deps manually to avoid pulling metrics/database/lightning
RUN pip install --no-cache-dir \
    "asteroid-filterbanks==0.4.0" \
    "einops>=0.6.0" \
    "pyannote.core==5.0.0" \
    "speechbrain==1.0.2" \
    "faster-whisper==1.2.1"

# --- Layer 2: pyannote.audio (pulls lightning etc as deps — unavoidable) ---
RUN pip install --no-cache-dir "pyannote.audio==3.4.0"

# --- Layer 3: everything else (light, changes often) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Clean pip cache ---
RUN pip cache purge || true

# --- Application code ---
COPY transcription.py        .
COPY diarize.py              .
COPY media_io.py             .
COPY speaker_mapper_voice.py .
COPY transcript_writer.py    .
COPY r2_client.py            .
COPY runpod_handler.py       .
COPY seed_data.json          .

# --- Runtime env ---
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TORCH_HOME=/runpod-volume/torch
ENV SPEECHBRAIN_CACHE=/runpod-volume/speechbrain/spkrec-ecapa-voxceleb
ENV SEED_DB_PATH=/app/seed_data.json

CMD ["python", "-u", "runpod_handler.py"]
