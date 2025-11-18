# vLLM Inference API

Production-ready Docker image for vLLM multimodal inference with OpenAI-compatible API. Optimized for RunPod deployment with easy model swapping via environment variables.

## Features

- **One-Command Model Swap** - Change models by updating a single environment variable
- **OpenAI-Compatible API** - Works with existing OpenAI clients and tools
- **Multimodal Support** - Text, images, videos, and audio in a unified API
- **Streaming Responses** - Real-time token streaming with Server-Sent Events
- **Production-Ready** - Continuous batching, chunked prefill, optimized memory usage
- **Comprehensive Logging** - Structured JSON logs with request tracking and performance metrics
- **Health Checks** - RunPod-compatible health endpoints for orchestration
- **Prometheus Metrics** - Built-in metrics for monitoring throughput and latency
- **Automatic Model Download** - Models downloaded on first startup (no pre-baking required)
- **S3 URL Support** - Direct support for S3 image/video URLs and base64 encoding

## Quick Start

### Local Development with Docker Compose

1. **Clone and navigate to the directory:**
   ```bash
   cd vllm-inference-api
   ```

2. **Set HuggingFace token (optional, for private models):**
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```

3. **Start the service:**
   ```bash
   docker-compose up --build
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Health: http://localhost:8000/health
   - Info: http://localhost:8000/info
   - Metrics: http://localhost:8000/metrics

### RunPod Deployment

1. **Build and push to Docker Hub:**
   ```bash
   docker build -t your-dockerhub-username/vllm-inference-api:latest .
   docker push your-dockerhub-username/vllm-inference-api:latest
   ```

2. **Create RunPod Template:**
   - Go to RunPod → Templates → New Template
   - Container Image: `your-dockerhub-username/vllm-inference-api:latest`
   - Container Disk: 50GB+ (for model caching)
   - Expose HTTP Port: 8000
   - Environment Variables (see configuration section below)

3. **Deploy Pod:**
   - Select your template
   - Choose GPU type (A100, H100, etc.)
   - Start pod
   - Wait for health check to pass (~3-5 minutes for model download)

## Configuration

All configuration is done via environment variables. The most important one:

### Essential Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2-VL-7B-Instruct` | **HuggingFace model ID - CHANGE THIS TO SWAP MODELS** |
| `HF_TOKEN` | - | HuggingFace token for private models (optional) |

### Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_MODEL_LEN` | `32768` | Maximum context length (reduce if OOM) |
| `GPU_MEMORY_UTIL` | `0.95` | GPU memory utilization (0.5-0.99) |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `DATA_PARALLEL_SIZE` | `1` | Number of replicas for data parallelism |
| `MAX_NUM_SEQS` | `12` | Max concurrent sequences in batch |
| `MAX_BATCHED_TOKENS` | `10240` | Max tokens per batch (higher = better throughput) |

### Multimodal Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_LIMIT` | `5` | Max images per request |
| `VIDEO_LIMIT` | `1` | Max videos per request |
| `AUDIO_LIMIT` | `0` | Max audio files per request |

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | API port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Advanced Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_CHUNKED_PREFILL` | `true` | Enable chunked prefill for better TTFT |
| `TRUST_REMOTE_CODE` | `true` | Trust remote code (required for Qwen models) |
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN` | Attention backend (FLASH_ATTN, XFORMERS) |
| `VLLM_IMAGE_FETCH_TIMEOUT` | `30` | Image URL fetch timeout (seconds) |
| `VLLM_VIDEO_FETCH_TIMEOUT` | `60` | Video URL fetch timeout (seconds) |

## Model Swapping Examples

### Qwen Models (Vision + Video)

```bash
# Qwen2-VL 7B (default)
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct

# Qwen2-VL 72B (requires 4-8 GPUs with tensor parallelism)
MODEL_NAME=Qwen/Qwen2.5-VL-72B-Instruct
TENSOR_PARALLEL_SIZE=8
MAX_MODEL_LEN=65536
```

### Other Vision-Language Models

```bash
# LLaVA 1.5
MODEL_NAME=llava-hf/llava-1.5-7b-hf
CHAT_TEMPLATE=template_llava.jinja

# Phi-3.5 Vision
MODEL_NAME=microsoft/Phi-3.5-vision-instruct
MAX_MODEL_LEN=4096

# Pixtral 12B
MODEL_NAME=mistralai/Pixtral-12B-2409
```

### Quantized Models

```bash
# AWQ 4-bit quantized Qwen2-VL
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct-AWQ
GPU_MEMORY_UTIL=0.90

# GPTQ 4-bit quantized
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4
```

## API Usage

### Python Client (OpenAI SDK)

```python
from openai import OpenAI

# Configure client
client = OpenAI(
    api_key="EMPTY",
    base_url="http://your-runpod-url:8000/v1"
)

# Text-only completion
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    max_tokens=500,
    temperature=0.7
)
print(response.choices[0].message.content)

# Image analysis (URL)
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {
                "url": "https://example.com/image.jpg"
            }}
        ]
    }],
    max_tokens=300
)
print(response.choices[0].message.content)

# Image analysis (base64)
import base64
from io import BytesIO
from PIL import Image

# Encode image
with Image.open("photo.jpg") as img:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            }}
        ]
    }],
    max_tokens=500
)

# Video analysis (S3 URL)
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {
                "url": "https://s3.amazonaws.com/bucket/video.mp4"
            }}
        ]
    }],
    max_tokens=1000
)

# Streaming response
stream = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[
        {"role": "user", "content": "Write a short story about AI"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Server info
curl http://localhost:8000/info

# Text completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100
  }'

# Image analysis
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }],
    "max_tokens": 300
  }'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

## Monitoring

### Logs

Structured JSON logs are written to stdout:

```json
{
  "timestamp": "2025-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "vllm-api",
  "message": "Request completed",
  "request_id": "abc-123",
  "model": "Qwen/Qwen2-VL-7B-Instruct",
  "duration_ms": 1234.56,
  "ttft_ms": 89.12,
  "input_tokens": 50,
  "output_tokens": 200,
  "tokens_per_sec": 162.3,
  "modalities": ["text", "image"]
}
```

### Prometheus Metrics

Access metrics at `http://localhost:8000/metrics`

Key metrics:
- `vllm:request_success_total` - Total successful requests
- `vllm:time_to_first_token_seconds` - TTFT distribution
- `vllm:time_per_output_token_seconds` - ITL distribution
- `vllm:request_prompt_tokens` - Input token counts
- `vllm:request_generation_tokens` - Output token counts

### Optional Monitoring Stack

Start Prometheus and Grafana:

```bash
docker-compose --profile monitoring up -d
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Performance Tuning

### Single GPU (A100 40GB)

```bash
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct
MAX_MODEL_LEN=32768
GPU_MEMORY_UTIL=0.95
MAX_NUM_SEQS=12
MAX_BATCHED_TOKENS=10240
```

### Multi-GPU Tensor Parallelism (4x A100)

```bash
MODEL_NAME=Qwen/Qwen2.5-VL-72B-Instruct
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=65536
GPU_MEMORY_UTIL=0.95
MAX_NUM_SEQS=8
MAX_BATCHED_TOKENS=8192
```

### High Throughput Configuration

```bash
MAX_BATCHED_TOKENS=20480  # Larger batches
MAX_NUM_SEQS=16           # More concurrent sequences
ENABLE_CHUNKED_PREFILL=true
```

### Low Latency Configuration

```bash
MAX_BATCHED_TOKENS=2048   # Smaller batches
MAX_NUM_SEQS=8            # Fewer concurrent sequences
```

### Memory-Constrained (reduce if OOM)

```bash
MAX_MODEL_LEN=16384       # Reduce context length
GPU_MEMORY_UTIL=0.85      # Lower GPU memory usage
MAX_NUM_SEQS=6            # Fewer concurrent requests
```

## Troubleshooting

### Model fails to download

- Check HF_TOKEN is set correctly for private models
- Ensure RunPod pod has internet access
- Increase startup timeout (models can be large)

### Out of Memory (OOM)

- Reduce `MAX_MODEL_LEN`
- Lower `GPU_MEMORY_UTIL` to 0.85-0.90
- Decrease `MAX_NUM_SEQS`
- Use quantized model (AWQ, GPTQ)
- Increase `TENSOR_PARALLEL_SIZE` for multi-GPU

### Slow inference

- Increase `MAX_BATCHED_TOKENS` for better throughput
- Enable `ENABLE_CHUNKED_PREFILL=true`
- Check GPU utilization in logs
- Use more GPUs with tensor parallelism

### Health check fails

- Wait longer (model download can take 3-5 minutes)
- Check logs: `docker logs vllm-inference`
- Verify GPU is accessible: `nvidia-smi`
- Check CUDA compatibility

## Project Structure

```
vllm-inference-api/
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Local development setup
├── requirements.txt            # Python dependencies
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI application
│   ├── config.py              # Configuration management
│   └── utils/
│       ├── __init__.py
│       └── logging.py         # Structured logging
├── scripts/
│   └── start.sh               # Startup script
└── monitoring/
    └── prometheus.yml         # Prometheus configuration
```

## License

MIT License - feel free to use in your projects!

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

For issues specific to:
- **vLLM**: https://github.com/vllm-project/vllm
- **RunPod**: https://docs.runpod.io
- **This project**: Open a GitHub issue

## Acknowledgments

Built with:
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [RunPod](https://runpod.io/) - GPU cloud platform
