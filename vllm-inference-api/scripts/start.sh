#!/bin/bash

set -e

echo "================================================"
echo "vLLM Inference API - Startup"
echo "================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to log with timestamp
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Check CUDA availability
log "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    log "Found $GPU_COUNT GPU(s)"
else
    log_error "nvidia-smi not found! GPU acceleration not available."
    exit 1
fi

# Validate configuration
log "Validating configuration..."
python3 -c "
from app.config import config
issues = config.validate()
if issues:
    print('Configuration warnings:')
    for issue in issues:
        print(f'  - {issue}')
else:
    print('Configuration valid')
"

# Display configuration
log "Current configuration:"
python3 -c "
from app.config import config
import json
print(json.dumps(config.to_dict(), indent=2))
"

# Check if model needs to be downloaded
log "Checking model availability..."
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-VL-7B-Instruct}"
CACHE_DIR="${HF_HOME:-/root/.cache/huggingface}"

log "Model: $MODEL_NAME"
log "Cache directory: $CACHE_DIR"

# Pre-download model if not cached (helps with timeout issues)
if [ ! -z "$HF_TOKEN" ]; then
    log "HuggingFace token provided, logging in..."
    echo "$HF_TOKEN" | huggingface-cli login --token
fi

log "Ensuring model is downloaded..."
python3 -c "
import sys
from huggingface_hub import snapshot_download
import os

model_name = os.getenv('MODEL_NAME', 'Qwen/Qwen2-VL-7B-Instruct')
token = os.getenv('HF_TOKEN', None)

try:
    print(f'Downloading/verifying model: {model_name}')
    path = snapshot_download(
        repo_id=model_name,
        token=token,
        resume_download=True,
        local_files_only=False
    )
    print(f'Model ready at: {path}')
except Exception as e:
    print(f'Error downloading model: {str(e)}', file=sys.stderr)
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    log_error "Failed to download model"
    exit 1
fi

log "Model download/verification complete"

# Set environment variables for vLLM
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_IMAGE_FETCH_TIMEOUT="${VLLM_IMAGE_FETCH_TIMEOUT:-30}"
export VLLM_VIDEO_FETCH_TIMEOUT="${VLLM_VIDEO_FETCH_TIMEOUT:-60}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-60}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

log "Environment variables set:"
log "  VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
log "  VLLM_IMAGE_FETCH_TIMEOUT=$VLLM_IMAGE_FETCH_TIMEOUT"
log "  VLLM_VIDEO_FETCH_TIMEOUT=$VLLM_VIDEO_FETCH_TIMEOUT"
log "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# Start vLLM server in background
log "Starting vLLM server..."
python3 -c "
from app.config import config
import subprocess
import sys

cmd = config.get_vllm_command()
print('Command:', ' '.join(cmd))
print('Starting vLLM server...')
sys.stdout.flush()

# Start vLLM in background
try:
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env={**subprocess.os.environ, **config.get_env_vars()}
    )

    # Write PID to file for potential shutdown
    with open('/tmp/vllm.pid', 'w') as f:
        f.write(str(process.pid))

    print(f'vLLM server started with PID: {process.pid}')
except Exception as e:
    print(f'Failed to start vLLM server: {str(e)}', file=sys.stderr)
    sys.exit(1)
" &

VLLM_PID=$!
log "vLLM server process started (PID: $VLLM_PID)"

# Give vLLM a moment to start
sleep 5

# Start FastAPI wrapper
log "Starting FastAPI wrapper..."
log "API will be available at http://0.0.0.0:${PORT:-8000}"

# Start FastAPI in foreground
python3 -m uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --log-level "${LOG_LEVEL:-info}"

# If FastAPI exits, cleanup
log_warn "FastAPI exited, cleaning up..."
if [ -f /tmp/vllm.pid ]; then
    VLLM_PID=$(cat /tmp/vllm.pid)
    log "Stopping vLLM server (PID: $VLLM_PID)..."
    kill $VLLM_PID 2>/dev/null || true
    rm -f /tmp/vllm.pid
fi

log "Shutdown complete"
