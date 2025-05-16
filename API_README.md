# CyberRAGLLM API

This document provides information on how to use the CyberRAGLLM API, which exposes OpenAI-compatible endpoints for the CyberRAGLLM application.

## Overview

CyberRAGLLM is a graph-based workflow application that answers cybersecurity questions using Retrieval-Augmented Generation (RAG) and web search. The API allows other programs to interact with CyberRAGLLM using the same interface as OpenAI's API.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the API server:
   ```
   python -m uvicorn server:app
   ```

   The server will start on `http://localhost:8000` by default.

## API Endpoints

### Root Endpoint

- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns basic API information.
- **Response Example**:
  ```json
  {
    "message": "CyberRAGLLM API is running",
    "version": "1.0.0",
    "endpoints": {
      "/v1/chat/completions": "OpenAI-compatible chat completions endpoint"
    }
  }
  ```

### Chat Completions

- **URL**: `/v1/chat/completions`
- **Method**: `POST`
- **Description**: OpenAI-compatible chat completions endpoint.
- **Request Format**:
  ```json
  {
    "model": "model_name",
    "messages": [
      {
        "role": "user",
        "content": "Your question here"
      }
    ],
    "temperature": 0,
    "max_retries": 3
  }
  ```
- **Response Format**:
  ```json
  {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "model_name",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "The answer to your question..."
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 20,
      "total_tokens": 30
    }
  }
  ```

## Example Usage

### Python

```python
import requests

API_URL = "http://localhost:8000/v1/chat/completions"

request_data = {
    "model": "hf.co/safe049/mistral-v0.3-7b-cybersecurity:latest",
    "messages": [
        {
            "role": "user",
            "content": "What are the most common cybersecurity threats?"
        }
    ],
    "temperature": 0,
    "max_retries": 3
}

response = requests.post(API_URL, json=request_data)
result = response.json()
answer = result["choices"][0]["message"]["content"]
print(answer)
```

### cURL

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hf.co/safe049/mistral-v0.3-7b-cybersecurity:latest",
    "messages": [
      {
        "role": "user",
        "content": "What are the most common cybersecurity threats?"
      }
    ],
    "temperature": 0,
    "max_retries": 3
  }'
```

## OpenAI Client Library Compatibility

Since the API is designed to be compatible with OpenAI's API, you can use OpenAI's client libraries by simply changing the base URL:

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",  # Not used but required
    base_url="http://localhost:8000/v1"  # Point to your local server
)

response = client.chat.completions.create(
    model="hf.co/safe049/mistral-v0.3-7b-cybersecurity:latest",
    messages=[
        {"role": "user", "content": "What are the most common cybersecurity threats?"}
    ]
)

print(response.choices[0].message.content)
```

## Notes

- The API currently supports only the chat completions endpoint.
- The `model` parameter is passed through to the response but doesn't affect the underlying model used.
- Token counts in the usage field are rough estimates.