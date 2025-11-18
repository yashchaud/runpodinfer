# Multi-stage build for vLLM inference API
# Optimized for RunPod deployment with CUDA 12.4.1

# Builder stage
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Runtime stage
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create app directory
WORKDIR /app

# Copy application code
COPY app/ /app/app/
COPY scripts/ /app/scripts/

# Make startup script executable
RUN chmod +x /app/scripts/start.sh

# Set default environment variables
ENV MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct \
    MAX_MODEL_LEN=32768 \
    GPU_MEMORY_UTIL=0.95 \
    TENSOR_PARALLEL_SIZE=1 \
    DATA_PARALLEL_SIZE=1 \
    MAX_NUM_SEQS=12 \
    MAX_BATCHED_TOKENS=10240 \
    IMAGE_LIMIT=5 \
    VIDEO_LIMIT=1 \
    AUDIO_LIMIT=0 \
    HOST=0.0.0.0 \
    PORT=8000 \
    VLLM_PORT=8001 \
    LOG_LEVEL=INFO \
    ENABLE_CHUNKED_PREFILL=true \
    TRUST_REMOTE_CODE=true \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    VLLM_IMAGE_FETCH_TIMEOUT=30 \
    VLLM_VIDEO_FETCH_TIMEOUT=60 \
    VLLM_ENGINE_ITERATION_TIMEOUT_S=60 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/root/.cache/huggingface

# Create cache directory
RUN mkdir -p /root/.cache/huggingface

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Run startup script
CMD ["/app/scripts/start.sh"]
