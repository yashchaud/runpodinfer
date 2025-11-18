# Quick Start Guide

## 1. Local Testing (5 minutes)

```bash
# Clone/navigate to directory
cd vllm-inference-api

# (Optional) Set HuggingFace token
export HF_TOKEN=your_token_here

# Start with docker-compose
docker-compose up --build

# Wait for "vLLM server is ready" message (~3-5 minutes for first run)

# In another terminal, test the API
python3 test_api.py http://localhost:8000
```

## 2. Deploy to RunPod (10 minutes)

### Step 1: Build and Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Build and push (replace 'yourusername' with your Docker Hub username)
chmod +x build_and_push.sh
./build_and_push.sh yourusername latest
```

### Step 2: Create RunPod Template

1. Go to https://www.runpod.io/console/templates
2. Click "New Template"
3. Fill in:
   - **Template Name**: vLLM Inference API
   - **Container Image**: `yourusername/vllm-inference-api:latest`
   - **Container Disk**: 50 GB (or more for larger models)
   - **Expose HTTP Ports**: `8000`
   - **Environment Variables**: Add these:

```
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct
MAX_MODEL_LEN=32768
GPU_MEMORY_UTIL=0.95
MAX_NUM_SEQS=12
MAX_BATCHED_TOKENS=10240
IMAGE_LIMIT=5
VIDEO_LIMIT=1
LOG_LEVEL=INFO
```

4. Click "Save Template"

### Step 3: Deploy Pod

1. Go to "GPU Pods"
2. Click "Deploy"
3. Select your template
4. Choose GPU (A100 40GB recommended for 7B models)
5. Click "Deploy On-Demand" or "Deploy Spot"
6. Wait for pod to start (~5 minutes including model download)
7. Click on pod → "Connect" → Note the HTTP endpoint URL

### Step 4: Test Your Deployment

```python
from openai import OpenAI

# Replace with your RunPod URL
client = OpenAI(
    api_key="EMPTY",
    base_url="https://your-pod-id-8000.proxy.runpod.net/v1"
)

# Test text completion
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Test image analysis
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
    }]
)
print(response.choices[0].message.content)
```

## 3. Swap Models (30 seconds)

Just change the `MODEL_NAME` environment variable in your RunPod template:

```
# Original (Qwen2-VL 7B)
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct

# Switch to LLaVA
MODEL_NAME=llava-hf/llava-1.5-7b-hf

# Switch to Phi-3 Vision
MODEL_NAME=microsoft/Phi-3.5-vision-instruct

# Switch to quantized model (uses less memory)
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct-AWQ
```

After changing, restart your pod. The new model will download automatically.

## 4. Monitoring

### Health Check
```bash
curl https://your-pod-8000.proxy.runpod.net/health
```

### View Metrics
```bash
curl https://your-pod-8000.proxy.runpod.net/metrics
```

### Check Logs (in RunPod Console)
1. Click on your pod
2. Click "Logs" tab
3. Look for JSON-formatted logs with request metrics

Example log entry:
```json
{
  "timestamp": "2025-01-15T10:30:45.123Z",
  "level": "INFO",
  "message": "Request completed",
  "request_id": "abc-123",
  "duration_ms": 1234.56,
  "ttft_ms": 89.12,
  "tokens_per_sec": 162.3,
  "modalities": ["text", "image"]
}
```

## 5. Troubleshooting

### "Health check failed" or pod won't start
- Wait longer (model download takes 3-5 minutes)
- Check logs for errors
- Verify GPU is available: SSH into pod and run `nvidia-smi`

### Out of Memory (OOM)
Update environment variables:
```
MAX_MODEL_LEN=16384
GPU_MEMORY_UTIL=0.85
MAX_NUM_SEQS=6
```
Or use a quantized model or larger GPU.

### Slow inference
- Increase `MAX_BATCHED_TOKENS=20480`
- Use multiple GPUs with `TENSOR_PARALLEL_SIZE=2` or higher
- Enable chunked prefill: `ENABLE_CHUNKED_PREFILL=true`

### Model not found
- Check `MODEL_NAME` is correct (must match HuggingFace model ID)
- For private models, ensure `HF_TOKEN` is set
- Verify pod has internet access

## 6. Cost Optimization

### Use Spot Instances
- 70% cheaper than on-demand
- RunPod automatically migrates before termination

### Use Smaller/Quantized Models
```
# 4-bit quantized uses ~50% less memory
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct-AWQ
```

### Right-size Your GPU
- 7B models: RTX 4090 (24GB) or A40 (48GB)
- 13B models: A100 40GB
- 72B models: 4x A100 40GB or 2x A100 80GB

### Auto-pause When Idle
- Set up RunPod auto-pause for inactive pods
- Or use serverless endpoints (pay per request)

## Support

- **Documentation**: See [README.md](README.md) for full details
- **Issues**: Open a GitHub issue
- **vLLM Docs**: https://docs.vllm.ai
- **RunPod Docs**: https://docs.runpod.io

That's it! You now have a production-ready multimodal inference API that you can swap models on with a single environment variable change. 🚀
