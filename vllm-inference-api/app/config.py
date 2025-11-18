"""
Configuration management for vLLM inference API.
All settings are configurable via environment variables for easy model swapping.
"""

import os
from typing import Optional


class Config:
    """Central configuration for vLLM server and API."""

    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)
    TRUST_REMOTE_CODE: bool = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"

    # Context and Memory Settings
    MAX_MODEL_LEN: int = int(os.getenv("MAX_MODEL_LEN", "32768"))
    GPU_MEMORY_UTIL: float = float(os.getenv("GPU_MEMORY_UTIL", "0.95"))

    # Parallelism Settings
    TENSOR_PARALLEL_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    DATA_PARALLEL_SIZE: int = int(os.getenv("DATA_PARALLEL_SIZE", "1"))

    # Batching and Performance
    MAX_NUM_SEQS: int = int(os.getenv("MAX_NUM_SEQS", "12"))
    MAX_BATCHED_TOKENS: int = int(os.getenv("MAX_BATCHED_TOKENS", "10240"))
    ENABLE_CHUNKED_PREFILL: bool = os.getenv("ENABLE_CHUNKED_PREFILL", "true").lower() == "true"

    # Multimodal Limits
    IMAGE_LIMIT: int = int(os.getenv("IMAGE_LIMIT", "5"))
    VIDEO_LIMIT: int = int(os.getenv("VIDEO_LIMIT", "1"))
    AUDIO_LIMIT: int = int(os.getenv("AUDIO_LIMIT", "0"))

    # Network Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    VLLM_PORT: int = int(os.getenv("VLLM_PORT", "8001"))

    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DISABLE_LOG_REQUESTS: bool = os.getenv("DISABLE_LOG_REQUESTS", "false").lower() == "true"

    # Timeout Settings
    IMAGE_FETCH_TIMEOUT: int = int(os.getenv("VLLM_IMAGE_FETCH_TIMEOUT", "30"))
    VIDEO_FETCH_TIMEOUT: int = int(os.getenv("VLLM_VIDEO_FETCH_TIMEOUT", "60"))
    ENGINE_ITERATION_TIMEOUT: int = int(os.getenv("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60"))

    # Attention Backend
    ATTENTION_BACKEND: str = os.getenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

    # Chat Template (optional)
    CHAT_TEMPLATE: Optional[str] = os.getenv("CHAT_TEMPLATE", None)

    @classmethod
    def get_vllm_command(cls) -> list[str]:
        """Generate vLLM server command with all configuration options."""
        cmd = [
            "vllm", "serve", cls.MODEL_NAME,
            "--host", cls.HOST,
            "--port", str(cls.VLLM_PORT),
            "--max-model-len", str(cls.MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(cls.GPU_MEMORY_UTIL),
            "--max-num-seqs", str(cls.MAX_NUM_SEQS),
            "--max-num-batched-tokens", str(cls.MAX_BATCHED_TOKENS),
        ]

        # Parallelism
        if cls.TENSOR_PARALLEL_SIZE > 1:
            cmd.extend(["--tensor-parallel-size", str(cls.TENSOR_PARALLEL_SIZE)])
            # Use data parallelism for multimodal encoder
            cmd.extend(["--mm-encoder-tp-mode", "data"])

        if cls.DATA_PARALLEL_SIZE > 1:
            cmd.extend(["--data-parallel-size", str(cls.DATA_PARALLEL_SIZE)])

        # Multimodal limits
        mm_limits = {}
        if cls.IMAGE_LIMIT > 0:
            mm_limits["image"] = cls.IMAGE_LIMIT
        if cls.VIDEO_LIMIT > 0:
            mm_limits["video"] = cls.VIDEO_LIMIT
        if cls.AUDIO_LIMIT > 0:
            mm_limits["audio"] = cls.AUDIO_LIMIT

        if mm_limits:
            import json
            cmd.extend(["--limit-mm-per-prompt", json.dumps(mm_limits)])

        # Performance optimizations
        if cls.ENABLE_CHUNKED_PREFILL:
            cmd.append("--enable-chunked-prefill")

        # Trust remote code (required for Qwen models)
        if cls.TRUST_REMOTE_CODE:
            cmd.append("--trust-remote-code")

        # Chat template
        if cls.CHAT_TEMPLATE:
            cmd.extend(["--chat-template", cls.CHAT_TEMPLATE])

        # Disable request logging for performance
        if cls.DISABLE_LOG_REQUESTS:
            cmd.append("--disable-log-requests")

        return cmd

    @classmethod
    def get_env_vars(cls) -> dict[str, str]:
        """Get environment variables to set for vLLM process."""
        env_vars = {
            "VLLM_ATTENTION_BACKEND": cls.ATTENTION_BACKEND,
            "VLLM_IMAGE_FETCH_TIMEOUT": str(cls.IMAGE_FETCH_TIMEOUT),
            "VLLM_VIDEO_FETCH_TIMEOUT": str(cls.VIDEO_FETCH_TIMEOUT),
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": str(cls.ENGINE_ITERATION_TIMEOUT),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }

        # Add HuggingFace token if provided
        if cls.HF_TOKEN:
            env_vars["HF_TOKEN"] = cls.HF_TOKEN

        return env_vars

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of warnings/errors."""
        issues = []

        # Check tensor parallelism
        if cls.TENSOR_PARALLEL_SIZE > 1:
            try:
                import torch
                gpu_count = torch.cuda.device_count()
                if cls.TENSOR_PARALLEL_SIZE > gpu_count:
                    issues.append(
                        f"TENSOR_PARALLEL_SIZE ({cls.TENSOR_PARALLEL_SIZE}) > "
                        f"available GPUs ({gpu_count})"
                    )
            except ImportError:
                issues.append("torch not installed, cannot validate GPU count")

        # Check memory utilization
        if cls.GPU_MEMORY_UTIL < 0.5 or cls.GPU_MEMORY_UTIL > 0.99:
            issues.append(
                f"GPU_MEMORY_UTIL ({cls.GPU_MEMORY_UTIL}) outside recommended range [0.5, 0.99]"
            )

        # Check max model length
        if cls.MAX_MODEL_LEN > 128000:
            issues.append(
                f"MAX_MODEL_LEN ({cls.MAX_MODEL_LEN}) is very large, may cause OOM"
            )

        return issues

    @classmethod
    def to_dict(cls) -> dict:
        """Convert configuration to dictionary for logging/display."""
        return {
            "model_name": cls.MODEL_NAME,
            "max_model_len": cls.MAX_MODEL_LEN,
            "gpu_memory_util": cls.GPU_MEMORY_UTIL,
            "tensor_parallel_size": cls.TENSOR_PARALLEL_SIZE,
            "data_parallel_size": cls.DATA_PARALLEL_SIZE,
            "max_num_seqs": cls.MAX_NUM_SEQS,
            "max_batched_tokens": cls.MAX_BATCHED_TOKENS,
            "image_limit": cls.IMAGE_LIMIT,
            "video_limit": cls.VIDEO_LIMIT,
            "audio_limit": cls.AUDIO_LIMIT,
            "host": cls.HOST,
            "port": cls.PORT,
            "vllm_port": cls.VLLM_PORT,
        }


# Singleton instance
config = Config()
