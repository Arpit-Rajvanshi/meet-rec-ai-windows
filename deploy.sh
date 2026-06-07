#!/bin/bash

# ==========================================
# Meet Recorder - Full Pipeline RunPod Deploy
# Whisper + Pyannote + SpeechBrain in one image
# ==========================================

set -e

echo "========================================"
echo "Meet Recorder Deployment"
echo "Full Pipeline: Whisper + Pyannote + SpeechBrain"
echo "========================================"
echo ""

# ==========================================
# CONFIGURATION
# ==========================================

DOCKER_USERNAME="${DOCKER_USERNAME:-mannoyed}"
IMAGE_NAME="${IMAGE_NAME:-meet-recorder}"
IMAGE_TAG="${IMAGE_TAG:-v3.0-full-pipeline}"

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# ==========================================
# VALIDATE
# ==========================================

echo "Validating prerequisites..."
echo ""

if ! command -v docker &> /dev/null; then
    echo "Docker not found"
    exit 1
fi
echo "Docker found"

REQUIRED_FILES=(
    "Dockerfile"
    "requirements.txt"
    "runpod_handler.py"
    "transcription.py"
    "diarize.py"
    "media_io.py"
    "speaker_mapper_voice.py"
    "transcript_writer.py"
    "r2_client.py"
    "seed_data.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Required file not found: $file"
        exit 1
    fi
    echo "Found: $file"
done

echo ""

# ==========================================
# BUILD
# ==========================================

echo "Building Docker image..."
echo "Image: $FULL_IMAGE_NAME"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo "Building..."
echo ""

docker build \
    --platform linux/amd64 \
    -f Dockerfile \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -t ${FULL_IMAGE_NAME} \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
else
    echo ""
    echo "Build failed!"
    exit 1
fi

echo ""

# ==========================================
# PUSH
# ==========================================

echo "Pushing to Docker Hub..."
echo ""

read -p "Push now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Pushing..."
    echo ""

    docker push ${FULL_IMAGE_NAME}

    if [ $? -eq 0 ]; then
        echo ""
        echo "Push successful!"
        echo ""
        echo "Image: $FULL_IMAGE_NAME"
    else
        echo ""
        echo "Push failed!"
        exit 1
    fi
fi

echo ""

# ==========================================
# INSTRUCTIONS
# ==========================================

cat <<EOF
========================================
DEPLOYMENT READY
========================================

Docker image: $FULL_IMAGE_NAME

NEXT STEPS:

1. Go to: https://www.runpod.io/console/serverless

2. Create or update the endpoint:
   - Image:             $FULL_IMAGE_NAME
   - GPU:               RTX 3090 / RTX 4090
   - Container Disk:    30 GB
   - Volume:            30 GB at /runpod-volume
   - Min Workers:       0
   - Max Workers:       3
   - Idle Timeout:      5 s
   - Execution Timeout: 3600 s    (1 hour - covers 2hr meetings)

3. Required env vars on the RunPod endpoint:
   - HF_TOKEN                 (Hugging Face, for Pyannote 3.1)
   - R2_ACCESS_KEY_ID         (Cloudflare R2)
   - R2_SECRET_ACCESS_KEY     (Cloudflare R2)
   - R2_BUCKET                (default: meet-recorder)
   - R2_ENDPOINT_URL          (https://<account-id>.r2.cloudflarestorage.com)

   Optional:
   - R2_AUDIO_PREFIX          (default: audios)
   - R2_TRANSCRIPT_PREFIX     (default: transcripts)
   - R2_PRESIGN_EXPIRY        (default: 604800 seconds / 7 days)
   - SEED_DB_PATH             (default: /app/seed_data.json)

4. Invoke the endpoint with one of:
     {"input": {"video_url":   "https://..."}}
     {"input": {"audio_url":   "https://..."}}     # WAV preferred
     {"input": {"audio_base64": "..."}}            # legacy, <10 MB only

   Optional fields:
     meeting_id, min_speakers, max_speakers, num_speakers,
     upload_audio_to_r2 (default true), upload_transcript_to_r2 (default true),
     return_transcript_inline (default true)

5. Response includes:
   - transcript_text   (inline transcript.txt)
   - transcript_url    (pre-signed R2 URL)
   - audio_url         (pre-signed R2 URL)
   - speaker_mapping   (SPEAKER_XX -> real name)
   - transcript_segments
   - metadata          (durations, counts, timings)

========================================
EOF
