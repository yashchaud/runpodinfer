#!/bin/bash

# Build and push Docker image to Docker Hub
# Usage: ./build_and_push.sh <dockerhub-username> [tag]

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <dockerhub-username> [tag]"
    echo "Example: $0 myusername latest"
    exit 1
fi

DOCKERHUB_USERNAME=$1
TAG=${2:-latest}
IMAGE_NAME="vllm-inference-api"
FULL_IMAGE_NAME="$DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG"

echo "================================================"
echo "Building and Pushing vLLM Inference API"
echo "================================================"
echo "Image: $FULL_IMAGE_NAME"
echo ""

# Build the image
echo "[1/3] Building Docker image..."
docker build -t $FULL_IMAGE_NAME .

# Also tag as latest if a specific tag was provided
if [ "$TAG" != "latest" ]; then
    echo "[2/3] Tagging as latest..."
    docker tag $FULL_IMAGE_NAME $DOCKERHUB_USERNAME/$IMAGE_NAME:latest
else
    echo "[2/3] Skipping additional tagging (already latest)"
fi

# Push to Docker Hub
echo "[3/3] Pushing to Docker Hub..."
docker push $FULL_IMAGE_NAME

if [ "$TAG" != "latest" ]; then
    docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:latest
fi

echo ""
echo "================================================"
echo "Build and Push Complete!"
echo "================================================"
echo "Image: $FULL_IMAGE_NAME"
echo ""
echo "To use in RunPod:"
echo "1. Create new template"
echo "2. Container Image: $FULL_IMAGE_NAME"
echo "3. Container Disk: 50GB+"
echo "4. Expose HTTP Port: 8000"
echo "5. Add environment variables:"
echo "   MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct"
echo "   (see README.md for all options)"
echo ""
