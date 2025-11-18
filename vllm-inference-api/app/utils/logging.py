"""
Structured logging utility for vLLM inference API.
Provides JSON-formatted logs for easy parsing and monitoring.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "model"):
            log_data["model"] = record.model
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "tokens_per_sec"):
            log_data["tokens_per_sec"] = record.tokens_per_sec
        if hasattr(record, "input_tokens"):
            log_data["input_tokens"] = record.input_tokens
        if hasattr(record, "output_tokens"):
            log_data["output_tokens"] = record.output_tokens
        if hasattr(record, "modalities"):
            log_data["modalities"] = record.modalities
        if hasattr(record, "error"):
            log_data["error"] = record.error
        if hasattr(record, "ttft_ms"):
            log_data["ttft_ms"] = record.ttft_ms

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a structured JSON logger.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    # Don't propagate to root logger
    logger.propagate = False

    return logger


class RequestLogger:
    """Helper class for logging request metrics."""

    def __init__(self, logger: logging.Logger, request_id: str, model: str):
        self.logger = logger
        self.request_id = request_id
        self.model = model
        self.start_time = time.time()
        self.first_token_time: Optional[float] = None
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.modalities: list[str] = []

    def set_modalities(self, modalities: list[str]):
        """Set the modalities used in this request."""
        self.modalities = modalities

    def set_input_tokens(self, count: int):
        """Set the input token count."""
        self.input_tokens = count

    def set_output_tokens(self, count: int):
        """Set the output token count."""
        self.output_tokens = count

    def mark_first_token(self):
        """Mark the time when first token was generated."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def log_completion(self, success: bool = True, error: Optional[str] = None):
        """Log request completion with all metrics."""
        duration_ms = (time.time() - self.start_time) * 1000

        log_data = {
            "request_id": self.request_id,
            "model": self.model,
            "duration_ms": round(duration_ms, 2),
            "modalities": self.modalities if self.modalities else ["text"],
        }

        # Calculate TTFT if available
        if self.first_token_time:
            ttft_ms = (self.first_token_time - self.start_time) * 1000
            log_data["ttft_ms"] = round(ttft_ms, 2)

        # Add token counts if available
        if self.input_tokens is not None:
            log_data["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            log_data["output_tokens"] = self.output_tokens

            # Calculate tokens per second
            if self.output_tokens > 0 and duration_ms > 0:
                tokens_per_sec = (self.output_tokens / duration_ms) * 1000
                log_data["tokens_per_sec"] = round(tokens_per_sec, 2)

        # Log with appropriate level
        if not success:
            log_data["error"] = error
            self.logger.error("Request failed", extra=log_data)
        elif duration_ms > 30000:  # Warn if request took > 30s
            self.logger.warning("Slow request", extra=log_data)
        else:
            self.logger.info("Request completed", extra=log_data)


@contextmanager
def log_request(logger: logging.Logger, request_id: str, model: str):
    """
    Context manager for logging request lifecycle.

    Usage:
        with log_request(logger, "req-123", "model-name") as req_logger:
            # Process request
            req_logger.set_modalities(["text", "image"])
            req_logger.mark_first_token()
            req_logger.set_output_tokens(100)
    """
    req_logger = RequestLogger(logger, request_id, model)
    try:
        yield req_logger
        req_logger.log_completion(success=True)
    except Exception as e:
        req_logger.log_completion(success=False, error=str(e))
        raise


def log_startup(logger: logging.Logger, config_dict: dict[str, Any]):
    """Log server startup with configuration."""
    logger.info("Starting vLLM inference server", extra={"config": config_dict})


def log_health_check(
    logger: logging.Logger, healthy: bool, details: Optional[dict] = None
):
    """Log health check results."""
    log_data = {"healthy": healthy}
    if details:
        log_data.update(details)

    if healthy:
        logger.debug("Health check passed", extra=log_data)
    else:
        logger.warning("Health check failed", extra=log_data)


def log_gpu_stats(logger: logging.Logger, gpu_stats: dict[str, Any]):
    """Log GPU utilization statistics."""
    logger.info("GPU statistics", extra=gpu_stats)
