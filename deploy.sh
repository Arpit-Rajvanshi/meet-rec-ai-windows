#!/bin/bash

# ==========================================
# RunPod Deployment
# Only Transcription + Diarization
# ==========================================

set -e

echo "========================================"
echo "RunPod Deployment"
echo "GPU: Transcription + Diarization Only"
echo "========================================"
echo ""

# ==========================================
# CONFIGURATION
# ==========================================

DOCKER_USERNAME="ayushk1503"
IMAGE_NAME="meeting-transcription"
IMAGE_TAG="v2.1"

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# ==========================================
# VALIDATE
# ==========================================

echo "Validating prerequisites..."
echo ""

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found"
    exit 1
fi
echo "✅ Docker found"

# Check required files
REQUIRED_FILES=(
    "Dockerfile"
    "requirements.txt"
    "runpod_handler.py"
    "transcription.py"
    "diarize.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file not found: $file"
        exit 1
    fi
    echo "✅ Found: $file"
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

echo "🔨 Building..."
echo ""

docker build \
    --platform linux/amd64 \
    -f Dockerfile \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -t ${FULL_IMAGE_NAME} \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
else
    echo ""
    echo "❌ Build failed!"
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
    echo "📤 Pushing..."
    echo ""
    
    docker push ${FULL_IMAGE_NAME}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Push successful!"
        echo ""
        echo "Image: $FULL_IMAGE_NAME"
    else
        echo ""
        echo "❌ Push failed!"
        exit 1
    fi
fi

echo ""

# ==========================================
# INSTRUCTIONS
# ==========================================

echo "========================================"
echo "✅ DEPLOYMENT READY"
echo "========================================"
echo ""
echo "Docker image: $FULL_IMAGE_NAME"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Go to: https://www.runpod.io/console/serverless"
echo ""
echo "2. Create endpoint:"
echo "   - Name: transcription-diarization"
echo "   - Image: $FULL_IMAGE_NAME"
echo "   - GPU: RTX 3090 / RTX 4090"
echo "   - Container Disk: 30GB  (smaller!)"
echo "   - Volume: 30GB at /runpod-volume"
echo ""
echo "3. Environment variables:"
echo "   - HF_TOKEN=your-huggingface-token"
echo ""
echo "4. Scaling:"
echo "   - Min Workers: 0"
echo "   - Max Workers: 3"
echo "   - Idle Timeout: 5s"
echo "   - Execution Timeout: 600s"
echo ""
echo "5. On your server:"
echo "   - Set RUNPOD_API_KEY in .env"
echo "   - Set RUNPOD_ENDPOINT_ID in .env"
echo "   - Run: python orchestrator_modular.py video.mp4"
echo ""
echo "========================================"
echo ""

echo "Benefits of this setup:"
echo "  ✅ Smaller Docker image (~5GB vs ~10GB)"
echo "  ✅ Faster cold starts (~30s vs ~60s)"
echo "  ✅ Change OCR/analysis without rebuild"
echo "  ✅ Lower GPU costs (only 3-4 min per video)"
echo ""