#!/usr/bin/env python3
"""
Test script for vLLM Inference API.
Tests text-only, image, and streaming capabilities.
"""

import sys

from openai import OpenAI

# Configure client
API_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
client = OpenAI(api_key="EMPTY", base_url=f"{API_URL}/v1")

print("=" * 60)
print("vLLM Inference API Test Suite")
print("=" * 60)

# Test 1: Health Check
print("\n[Test 1] Health Check")
print("-" * 60)
import requests

try:
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Test 2: Server Info
print("\n[Test 2] Server Info")
print("-" * 60)
try:
    response = requests.get(f"{API_URL}/info")
    print(f"Model: {response.json()['model']}")
    print(f"Multimodal: {response.json()['multimodal_support']}")
    print(f"Limits: {response.json()['limits']}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Text Completion
print("\n[Test 3] Text Completion")
print("-" * 60)
try:
    response = client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
        max_tokens=50,
        temperature=0.1,
    )
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.total_tokens}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Streaming
print("\n[Test 4] Streaming Response")
print("-" * 60)
try:
    stream = client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        stream=True,
        max_tokens=100,
    )

    print("Stream: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
except Exception as e:
    print(f"Error: {e}")

# Test 5: Image Analysis (URL)
print("\n[Test 5] Image Analysis (URL)")
print("-" * 60)
try:
    # Using a sample image from internet
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    response = client.chat.completions.create(
        model="test",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=100,
    )
    print(f"Image URL: {image_url[:60]}...")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error (expected if model doesn't support vision): {e}")

# Test 6: Base64 Image
print("\n[Test 6] Image Analysis (Base64)")
print("-" * 60)
try:
    import base64
    from io import BytesIO

    from PIL import Image

    # Create a simple test image
    img = Image.new("RGB", (100, 100), color="red")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model="test",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ],
            }
        ],
        max_tokens=50,
    )
    print(f"Generated red 100x100 image")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error (expected if model doesn't support vision): {e}")

print("\n" + "=" * 60)
print("Test Suite Completed!")
print("=" * 60)
