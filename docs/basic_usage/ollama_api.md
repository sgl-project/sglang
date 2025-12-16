# Ollama-Compatible API

SGLang provides Ollama API compatibility, allowing you to use the Ollama CLI and Python library with SGLang as the inference backend. This also includes a Smart Router for intelligent routing between local and remote models.

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

The Smart Router intelligently routes requests between a local Ollama instance (fast, lightweight) and a remote SGLang server (powerful) using an LLM judge.

### How It Works

```
User Request
     │
     ▼
┌─────────────────────┐
│     LLM Judge       │  Classifies as SIMPLE or COMPLEX
│  (local model)      │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  SIMPLE → Local     │  Fast response from local Ollama
│  COMPLEX → Remote   │  Powerful response from SGLang
└─────────────────────┘
```

The LLM judge analyzes each request and decides:
- **SIMPLE**: Quick responses, greetings, factual questions, definitions, basic Q&A
- **COMPLEX**: Deep reasoning, multi-step analysis, long explanations, creative writing

### Setup

**Terminal 1: Local Ollama**
```bash
ollama pull llama3.2  # or any local model
ollama serve
```

**Terminal 2: Remote SGLang (GPU)**
```bash
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 30001 \
    --host 0.0.0.0
```

**Terminal 3: SSH Tunnel (if needed)**
```bash
ssh -L 30001:localhost:30001 user@gpu-server -N &
```

### Configuration

```python
from sglang.srt.entrypoints.ollama.smart_router import SmartRouter

router = SmartRouter(
    # Local Ollama
    local_host="http://localhost:11434",
    local_model="llama3.2",

    # Remote SGLang
    remote_host="http://localhost:30001",
    remote_model="Qwen/Qwen2.5-1.5B-Instruct",

    # LLM Judge (optional, defaults to local_model)
    judge_model="llama3.2",
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `local_host` | `str` | URL of local Ollama instance |
| `local_model` | `str` | Model name for local Ollama |
| `remote_host` | `str` | URL of remote SGLang server |
| `remote_model` | `str` | Model name for remote SGLang |
| `judge_model` | `str` | Model used for routing decisions (defaults to `local_model`) |

### Usage

```python
# Auto-routing via LLM judge
response = router.chat("Hello!", verbose=True)
# [Router] LLM Judge: SIMPLE
# [Router] -> Local Ollama | Model: llama3.2

response = router.chat("Explain quantum computing in detail", verbose=True)
# [Router] LLM Judge: COMPLEX
# [Router] -> Remote SGLang | Model: Qwen/Qwen2.5-1.5B-Instruct

# Force routing (skip LLM judge)
response = router.chat("question", force_local=True)
response = router.chat("question", force_remote=True)

# Streaming
for chunk in router.chat_stream("Tell me a story"):
    print(chunk['message']['content'], end='', flush=True)
```

## Summary

| Component | Purpose |
|-----------|---------|
| **Ollama API** | Familiar CLI/API that developers already know |
| **SGLang Backend** | High-performance inference engine |
| **Smart Router** | Intelligent routing - fast local for simple tasks, powerful remote for complex tasks |
