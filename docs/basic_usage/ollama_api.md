# Ollama-Compatible API

SGLang provides Ollama API compatibility, allowing you to use the Ollama CLI and Python library with SGLang as the inference backend.

## Prerequisites

```bash
# Install the Ollama Python library (for Python client usage)
pip install ollama
```

> **Note**: You don't need the Ollama server installed - SGLang acts as the backend. You only need the `ollama` CLI or Python library as the client.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET, HEAD | Health check for Ollama CLI |
| `/api/tags` | GET | List available models |
| `/api/chat` | POST | Chat completions (streaming & non-streaming) |
| `/api/generate` | POST | Text generation (streaming & non-streaming) |
| `/api/show` | POST | Model information |

## Quick Start

### 1. Launch SGLang Server

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 30001 \
    --host 0.0.0.0
```

> **Note**: The model name used with `ollama run` must match exactly what you passed to `--model`.

### 2. Use Ollama CLI

```bash
# List available models
OLLAMA_HOST=http://localhost:30001 ollama list

# Interactive chat
OLLAMA_HOST=http://localhost:30001 ollama run "Qwen/Qwen2.5-1.5B-Instruct"
```

If connecting to a remote server behind a firewall:

```bash
# SSH tunnel
ssh -L 30001:localhost:30001 user@gpu-server -N &

# Then use Ollama CLI as above
OLLAMA_HOST=http://localhost:30001 ollama list
```

### 3. Use Ollama Python Library

```python
import ollama

client = ollama.Client(host='http://localhost:30001')

# Non-streaming
response = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])

# Streaming
stream = client.chat(
    model='Qwen/Qwen2.5-1.5B-Instruct',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

## Smart Router

For intelligent routing between local Ollama (fast) and remote SGLang (powerful) using an LLM judge, see the [Smart Router documentation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/ollama/README.md).

## Summary

| Component | Purpose |
|-----------|---------|
| **Ollama API** | Familiar CLI/API that developers already know |
| **SGLang Backend** | High-performance inference engine |
| **Smart Router** | Intelligent routing - fast local for simple tasks, powerful remote for complex tasks |
