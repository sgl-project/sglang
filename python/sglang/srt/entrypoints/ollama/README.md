# SGLang Ollama Integration

Ollama API compatibility for SGLang, plus a Smart Router for intelligent routing between local and remote models.

## Features

1. **Ollama-compatible API** - Use Ollama CLI/library with SGLang backend
2. **Smart Router** - LLM-based routing between local and remote models

---

## Quick Start: Ollama API

### 1. Start SGLang Server (GPU)

```bash
# Start with any HuggingFace model
python -m sglang.launch_server \
    --model <YOUR_MODEL> \
    --port 30001 \
    --host 0.0.0.0
```

### 2. Connect from Local Machine

```bash
# SSH tunnel if behind firewall
ssh -L 30001:localhost:30001 user@gpu-server -N &

# Use Ollama CLI
OLLAMA_HOST=http://localhost:30001 ollama list
OLLAMA_HOST=http://localhost:30001 ollama run "<MODEL_NAME>"
```

### 3. Ollama Python Library

```python
import ollama

client = ollama.Client(host='http://localhost:30001')
response = client.chat(
    model='<MODEL_NAME>',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response['message']['content'])
```

---

## Smart Router

Intelligently routes requests between local Ollama and remote SGLang using an LLM judge.

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

The LLM judge (running on local Ollama) analyzes each request and decides:
- **SIMPLE**: Quick responses, greetings, factual questions, definitions, basic Q&A
- **COMPLEX**: Deep reasoning, multi-step analysis, long explanations, creative writing

### Setup

**Terminal 1: Local Ollama**
```bash
ollama serve
ollama pull <LOCAL_MODEL>  # e.g., llama3.2, mistral, phi3
```

**Terminal 2: Remote SGLang (GPU)**
```bash
ssh user@gpu-server
python -m sglang.launch_server --model <REMOTE_MODEL> --port 30001 --host 0.0.0.0
```

**Terminal 3: Smart Router**
```bash
ssh -L 30001:localhost:30001 user@gpu-server -N &
python python/sglang/srt/entrypoints/ollama/smart_router.py
```

### Configuration

```python
from sglang.srt.entrypoints.ollama.smart_router import SmartRouter

router = SmartRouter(
    # Local Ollama
    local_host="http://localhost:11434",
    local_model="llama3.2",  # or any Ollama model

    # Remote SGLang
    remote_host="http://localhost:30001",
    remote_model="Qwen/Qwen2.5-1.5B-Instruct",  # or any HuggingFace model

    # LLM Judge (optional, defaults to local_model)
    judge_model="llama3.2",
)
```

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
    print(chunk['message']['content'], end='')
```

---

## Value

- **Ollama**: Simple CLI/API developers already know
- **SGLang**: High-performance inference
- **Smart Router**: Intelligent routing - fast local for simple tasks, powerful remote for complex tasks
