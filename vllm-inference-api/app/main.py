"""
FastAPI wrapper for vLLM inference server.
Provides health checks, metrics, and proxy to OpenAI-compatible API.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .config import config
from .utils import log_health_check, log_startup, setup_logger

# Setup logger
logger = setup_logger("vllm-api", config.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    log_startup(logger, config.to_dict())
    logger.info(f"FastAPI wrapper listening on {config.HOST}:{config.PORT}")
    logger.info(f"Proxying to vLLM server at http://localhost:{config.VLLM_PORT}")

    # Wait for vLLM to be ready
    await wait_for_vllm_ready()

    yield

    # Shutdown
    logger.info("Shutting down FastAPI wrapper")


async def wait_for_vllm_ready(max_retries: int = 60, retry_delay: int = 5):
    """Wait for vLLM server to be ready."""
    vllm_url = f"http://localhost:{config.VLLM_PORT}/health"

    for i in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(vllm_url, timeout=2.0)
                if response.status_code == 200:
                    logger.info("vLLM server is ready")
                    return
        except Exception as e:
            if i == 0:
                logger.info(
                    f"Waiting for vLLM server to start... (attempt {i + 1}/{max_retries})"
                )
            elif i % 6 == 0:  # Log every 30 seconds
                logger.info(
                    f"Still waiting for vLLM server... (attempt {i + 1}/{max_retries})"
                )

        await asyncio.sleep(retry_delay)

    logger.error("vLLM server failed to start within timeout period")
    raise RuntimeError("vLLM server not ready")


# Create FastAPI app
app = FastAPI(
    title="vLLM Inference API",
    description="OpenAI-compatible API for vLLM multimodal inference",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for RunPod and orchestrators.
    Returns server status and basic configuration.
    """
    try:
        # Check vLLM health
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:{config.VLLM_PORT}/health", timeout=5.0
            )
            vllm_healthy = response.status_code == 200

        # Get GPU stats if available
        gpu_stats = await get_gpu_stats()

        health_data = {
            "status": "healthy" if vllm_healthy else "unhealthy",
            "model": config.MODEL_NAME,
            "vllm_status": "ready" if vllm_healthy else "not_ready",
            "gpu_stats": gpu_stats,
        }

        log_health_check(logger, vllm_healthy, health_data)

        return JSONResponse(
            content=health_data, status_code=200 if vllm_healthy else 503
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)}, status_code=503
        )


@app.get("/metrics")
async def metrics():
    """
    Proxy Prometheus metrics from vLLM server.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:{config.VLLM_PORT}/metrics", timeout=10.0
            )
            return Response(
                content=response.content,
                media_type="text/plain",
                status_code=response.status_code,
            )
    except Exception as e:
        logger.error(f"Failed to fetch metrics: {str(e)}")
        raise HTTPException(status_code=503, detail="Metrics unavailable")


@app.get("/info")
async def info():
    """
    Get server information and configuration.
    """
    return {
        "model": config.MODEL_NAME,
        "configuration": config.to_dict(),
        "multimodal_support": {
            "image": config.IMAGE_LIMIT > 0,
            "video": config.VIDEO_LIMIT > 0,
            "audio": config.AUDIO_LIMIT > 0,
        },
        "limits": {
            "max_images_per_request": config.IMAGE_LIMIT,
            "max_videos_per_request": config.VIDEO_LIMIT,
            "max_audio_per_request": config.AUDIO_LIMIT,
            "max_model_len": config.MAX_MODEL_LEN,
        },
    }


@app.get("/v1/models")
async def list_models():
    """
    Proxy models list from vLLM (OpenAI-compatible endpoint).
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://localhost:{config.VLLM_PORT}/v1/models", timeout=10.0
            )
            return JSONResponse(
                content=response.json(), status_code=response.status_code
            )
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=503, detail="Models endpoint unavailable")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Proxy chat completions to vLLM server (OpenAI-compatible).
    Supports streaming and non-streaming responses.
    Handles multimodal inputs: text, images, videos.
    """
    request_id = str(uuid.uuid4())

    try:
        body = await request.json()
        is_streaming = body.get("stream", False)

        # Extract modalities from request
        modalities = extract_modalities(body)

        # Log request start
        logger.info(
            "Received chat completion request",
            extra={
                "request_id": request_id,
                "model": body.get("model", "unknown"),
                "stream": is_streaming,
                "modalities": modalities,
            },
        )

        if is_streaming:
            return StreamingResponse(
                stream_vllm_response(request_id, body), media_type="text/event-stream"
            )
        else:
            return await proxy_vllm_request(request_id, body)

    except Exception as e:
        logger.error(
            f"Request failed: {str(e)}",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise HTTPException(status_code=500, detail=str(e))


async def proxy_vllm_request(request_id: str, body: dict) -> JSONResponse:
    """Proxy non-streaming request to vLLM."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://localhost:{config.VLLM_PORT}/v1/chat/completions", json=body
            )

            result = response.json()

            # Log completion
            usage = result.get("usage", {})
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            )

            return JSONResponse(content=result, status_code=response.status_code)

    except Exception as e:
        logger.error(
            f"Proxy request failed: {str(e)}", extra={"request_id": request_id}
        )
        raise


async def stream_vllm_response(request_id: str, body: dict) -> AsyncIterator[str]:
    """Stream response from vLLM server."""
    first_token = True

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"http://localhost:{config.VLLM_PORT}/v1/chat/completions",
                json=body,
            ) as response:
                async for chunk in response.aiter_bytes():
                    if first_token:
                        logger.info(
                            "First token received", extra={"request_id": request_id}
                        )
                        first_token = False

                    yield chunk

        logger.info("Streaming completed", extra={"request_id": request_id})

    except Exception as e:
        logger.error(f"Streaming failed: {str(e)}", extra={"request_id": request_id})
        raise


def extract_modalities(body: dict) -> list[str]:
    """Extract modalities from chat completion request."""
    modalities = set(["text"])  # Always has text

    messages = body.get("messages", [])
    for message in messages:
        content = message.get("content")

        if isinstance(content, list):
            for item in content:
                item_type = item.get("type", "")

                if item_type == "image_url":
                    modalities.add("image")
                elif item_type == "video_url":
                    modalities.add("video")
                elif item_type == "input_audio":
                    modalities.add("audio")

    return sorted(list(modalities))


async def get_gpu_stats() -> Optional[dict]:
    """Get GPU statistics if available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}

        gpu_count = torch.cuda.device_count()
        gpu_stats = {"available": True, "count": gpu_count, "devices": []}

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            memory_total = props.total_memory / 1024**3  # GB

            gpu_stats["devices"].append(
                {
                    "id": i,
                    "name": props.name,
                    "memory_total_gb": round(memory_total, 2),
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2),
                    "memory_utilization": round(
                        (memory_allocated / memory_total) * 100, 1
                    ),
                }
            )

        return gpu_stats

    except Exception as e:
        logger.warning(f"Failed to get GPU stats: {str(e)}")
        return {"available": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower(),
    )
