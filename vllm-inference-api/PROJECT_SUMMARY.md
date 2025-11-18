# vLLM Inference API - Project Summary

## What We Built

A production-ready Docker image for **vLLM multimodal inference** with an OpenAI-compatible API, optimized for RunPod deployment. The key feature is **one-command model swapping** via environment variables.

## Key Features ✅

1. **Easy Model Swapping** - Change `MODEL_NAME` environment variable to switch between any vLLM-supported model
2. **Multimodal Support** - Text, images (URL/base64), videos (URL/base64), all in one unified API
3. **OpenAI Compatible** - Works with existing OpenAI client libraries and tools
4. **Production Optimized** - Continuous batching, chunked prefill, optimal memory management
5. **Streaming Support** - Real-time token streaming via Server-Sent Events
6. **Comprehensive Logging** - Structured JSON logs with request tracking, TTFT, tokens/sec
7. **Health & Metrics** - RunPod-compatible health checks and Prometheus metrics
8. **Auto Model Download** - Models downloaded on startup, no pre-baking needed
9. **Parallel Request Handling** - Efficient request queuing and batching built into vLLM

## Project Structure

```
vllm-inference-api/
├── Dockerfile                    # Multi-stage Docker build (CUDA 12.4.1)
├── docker-compose.yml            # Local dev environment with optional monitoring
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── README.md                     # Comprehensive documentation
├── QUICKSTART.md                 # 5-minute quick start guide
├── PROJECT_SUMMARY.md            # This file
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI wrapper (health, metrics, proxy)
│   ├── config.py                # Configuration management (env vars)
│   └── utils/
│       ├── __init__.py
│       └── logging.py           # Structured JSON logging
│
├── scripts/
│   └── start.sh                 # Startup script (model download, server launch)
│
├── monitoring/
│   └── prometheus.yml           # Prometheus configuration
│
├── test_api.py                  # Test suite for all features
├── client_example.py            # Example Python client library
└── build_and_push.sh            # Docker Hub build/push script
```

## Technology Stack

- **Inference Engine**: vLLM >= 0.6.1 (continuous batching, PagedAttention)
- **API Framework**: FastAPI + Uvicorn
- **Base Image**: NVIDIA CUDA 12.4.1 Runtime (Ubuntu 22.04)
- **Model Support**: 90+ architectures (Qwen, LLaVA, Phi, Pixtral, etc.)
- **Monitoring**: Prometheus + Grafana (optional)

## How It Works

1. **Startup** (`start.sh`):
   - Validates CUDA/GPU availability
   - Downloads model from HuggingFace (if not cached)
   - Starts vLLM server with optimized configuration
   - Starts FastAPI wrapper for health/metrics endpoints

2. **Request Flow**:
   - Client sends request to FastAPI wrapper (port 8000)
   - FastAPI proxies to vLLM OpenAI API (internal port 8001)
   - vLLM processes with continuous batching
   - Response streamed back to client (if streaming enabled)
   - Structured logs emitted with timing metrics

3. **Model Swapping**:
   - Change `MODEL_NAME` environment variable
   - Restart container
   - New model auto-downloads and loads
   - No code changes required

## Quick Commands

### Local Development
```bash
# Start the service
docker-compose up --build

# Test the API
python test_api.py http://localhost:8000
```

### Production Deployment
```bash
# Build and push to Docker Hub
./build_and_push.sh yourusername latest

# Use in RunPod template with environment variables
```

### Model Swapping Examples
```bash
# Qwen2-VL 7B (default, vision + video)
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct

# LLaVA 1.5 (vision)
MODEL_NAME=llava-hf/llava-1.5-7b-hf

# Phi-3.5 Vision
MODEL_NAME=microsoft/Phi-3.5-vision-instruct

# Quantized (4-bit, uses less memory)
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct-AWQ
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check for orchestrators |
| `/info` | GET | Server info and configuration |
| `/metrics` | GET | Prometheus metrics |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completion |

## Configuration Highlights

**Default Configuration** (Qwen2-VL 7B on single GPU):
```bash
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct
MAX_MODEL_LEN=32768
GPU_MEMORY_UTIL=0.95
TENSOR_PARALLEL_SIZE=1
MAX_NUM_SEQS=12
MAX_BATCHED_TOKENS=10240
IMAGE_LIMIT=5
VIDEO_LIMIT=1
```

**Multi-GPU Configuration** (72B model on 8 GPUs):
```bash
MODEL_NAME=Qwen/Qwen2.5-VL-72B-Instruct
TENSOR_PARALLEL_SIZE=8
MAX_MODEL_LEN=65536
GPU_MEMORY_UTIL=0.95
```

## Performance Optimizations Included

1. **Continuous Batching** - Dynamic request batching (vLLM built-in)
2. **PagedAttention** - Efficient KV cache management
3. **Chunked Prefill** - Better time-to-first-token
4. **Flash Attention** - Optimized attention computation
5. **Memory Management** - `expandable_segments:True` to reduce fragmentation
6. **Data Parallelism** - For vision encoder (better than tensor parallelism)

## Monitoring & Logging

**Structured JSON Logs** - Every request logged with:
- Request ID
- Duration (total, TTFT)
- Tokens per second
- Input/output token counts
- Modalities used (text, image, video)

**Prometheus Metrics**:
- Request rate and latency (P50, P95, P99)
- Throughput (tokens/sec)
- GPU memory utilization
- Queue depth

## Example Usage

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="https://your-pod-8000.proxy.runpod.net/v1"
)

# Text completion
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Image analysis
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://s3.amazonaws.com/bucket/image.jpg"}}
        ]
    }]
)

# Streaming
stream = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```

## Testing

Run the test suite:
```bash
python test_api.py http://localhost:8000
```

Tests include:
1. Health check
2. Server info
3. Text completion
4. Streaming response
5. Image analysis (URL)
6. Image analysis (base64)

## Deployment Checklist

- [ ] Build Docker image
- [ ] Push to Docker Hub
- [ ] Create RunPod template
- [ ] Set environment variables (MODEL_NAME, etc.)
- [ ] Deploy pod with appropriate GPU
- [ ] Wait for health check (3-5 minutes)
- [ ] Test with client library
- [ ] Monitor logs and metrics

## Files Reference

| File | Purpose |
|------|---------|
| `README.md` | Complete documentation |
| `QUICKSTART.md` | 5-minute deployment guide |
| `PROJECT_SUMMARY.md` | This summary |
| `.env.example` | All configuration options |
| `test_api.py` | API test suite |
| `client_example.py` | Example client library |
| `build_and_push.sh` | Docker Hub deployment |

## Next Steps

1. **Test Locally**: `docker-compose up --build`
2. **Deploy to RunPod**: Follow QUICKSTART.md
3. **Customize Configuration**: See .env.example
4. **Swap Models**: Change MODEL_NAME and restart
5. **Monitor Performance**: Check logs and metrics

## Support & Resources

- **vLLM Docs**: https://docs.vllm.ai
- **RunPod Docs**: https://docs.runpod.io
- **OpenAI API**: https://platform.openai.com/docs/api-reference
- **This Project**: See README.md for full details

---

**Built with vLLM, FastAPI, and ❤️ for easy LLM deployment**
