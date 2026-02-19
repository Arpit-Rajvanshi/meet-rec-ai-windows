# ==========================================
# RunPod Handler
# Optimized for Size (Runtime Base)
# ==========================================

# 1. Use the Runtime image with Torch 2.2.2 pre-installed
# This matches your requirements.txt exactly, so no re-downloading!
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# 2. System Dependencies
# DEBIAN_FRONTEND prevents the time zone prompt that freezes builds
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Python Dependencies
COPY requirements.txt .

# Upgrade pip and install requirements
# --no-cache-dir reduces image size by not saving downloaded wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "filelock>=3.12.2" && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy Code
COPY transcription.py .
COPY diarize.py .
COPY runpod_handler.py .

# 5. Environment Config
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TORCH_HOME=/runpod-volume/torch

# 6. Start
CMD ["python", "-u", "runpod_handler.py"]