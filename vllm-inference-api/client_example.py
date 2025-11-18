#!/usr/bin/env python3
"""
Example client library for vLLM Inference API.
Demonstrates all supported features: text, images, videos, streaming.
"""

import base64
from io import BytesIO
from typing import Iterator, Optional

from openai import OpenAI
from PIL import Image


class VLLMClient:
    """
    Client wrapper for vLLM Inference API.

    Usage:
        client = VLLMClient("https://your-pod-8000.proxy.runpod.net")
        response = client.chat("Hello, how are you?")
        print(response)
    """

    def __init__(self, base_url: str, api_key: str = "EMPTY"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the API (e.g., "http://localhost:8000")
            api_key: API key (use "EMPTY" for no authentication)
        """
        self.base_url = base_url.rstrip("/")
        self.client = OpenAI(api_key=api_key, base_url=f"{self.base_url}/v1")

    def chat(
        self,
        message: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Simple text chat completion.

        Args:
            message: User message
            model: Model name (optional, server uses configured model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            stream: Whether to stream the response

        Returns:
            Generated text or iterator of text chunks if streaming
        """
        messages = [{"role": "user", "content": message}]

        response = self.client.chat.completions.create(
            model=model or "default",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        if stream:
            return self._stream_response(response)
        else:
            return response.choices[0].message.content

    def chat_with_image_url(
        self,
        message: str,
        image_url: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Chat completion with image URL.

        Args:
            message: User message about the image
            image_url: URL of the image (HTTP/HTTPS or S3)
            model: Model name (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=model or "default",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    def chat_with_image_file(
        self,
        message: str,
        image_path: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Chat completion with local image file.

        Args:
            message: User message about the image
            image_path: Path to local image file
            model: Model name (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        # Encode image to base64
        with Image.open(image_path) as img:
            buffered = BytesIO()
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        image_data_uri = f"data:image/png;base64,{img_base64}"

        return self.chat_with_image_url(
            message=message,
            image_url=image_data_uri,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat_with_video_url(
        self,
        message: str,
        video_url: str,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """
        Chat completion with video URL.

        Args:
            message: User message about the video
            video_url: URL of the video (HTTP/HTTPS or S3)
            model: Model name (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=model or "default",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    def chat_multimodal(
        self,
        message: str,
        image_urls: Optional[list[str]] = None,
        video_urls: Optional[list[str]] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """
        Chat completion with multiple images and/or videos.

        Args:
            message: User message
            image_urls: List of image URLs
            video_urls: List of video URLs
            model: Model name (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        content = [{"type": "text", "text": message}]

        # Add images
        if image_urls:
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})

        # Add videos
        if video_urls:
            for url in video_urls:
                content.append({"type": "video_url", "video_url": {"url": url}})

        messages = [{"role": "user", "content": content}]

        response = self.client.chat.completions.create(
            model=model or "default",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    def stream_chat(
        self,
        message: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """
        Stream chat completion.

        Args:
            message: User message
            model: Model name (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Text chunks as they are generated
        """
        return self.chat(
            message=message,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

    def _stream_response(self, stream) -> Iterator[str]:
        """Helper to process streaming response."""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def health(self) -> dict:
        """
        Check API health status.

        Returns:
            Health status dictionary
        """
        import requests

        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def info(self) -> dict:
        """
        Get server information and configuration.

        Returns:
            Server info dictionary
        """
        import requests

        response = requests.get(f"{self.base_url}/info")
        return response.json()


# Example usage
if __name__ == "__main__":
    import sys

    # Get API URL from command line or use default
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"Connecting to: {api_url}")
    print("=" * 60)

    # Initialize client
    client = VLLMClient(api_url)

    # Example 1: Health check
    print("\n1. Health Check")
    print("-" * 60)
    health = client.health()
    print(f"Status: {health.get('status')}")
    print(f"Model: {health.get('model')}")

    # Example 2: Simple chat
    print("\n2. Simple Chat")
    print("-" * 60)
    response = client.chat("What is the capital of France?", max_tokens=50)
    print(f"Response: {response}")

    # Example 3: Streaming chat
    print("\n3. Streaming Chat")
    print("-" * 60)
    print("Response: ", end="", flush=True)
    for chunk in client.stream_chat("Count from 1 to 5.", max_tokens=100):
        print(chunk, end="", flush=True)
    print()

    # Example 4: Image analysis (URL)
    print("\n4. Image Analysis (URL)")
    print("-" * 60)
    try:
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        response = client.chat_with_image_url(
            "Describe this image briefly.", image_url, max_tokens=100
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 5: Multiple images
    print("\n5. Multiple Images")
    print("-" * 60)
    try:
        response = client.chat_multimodal(
            "Compare these two images. What are the main differences?",
            image_urls=[
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
            ],
            max_tokens=200,
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error (expected): {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
