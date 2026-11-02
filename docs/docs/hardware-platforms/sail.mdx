---
title: "T-HEAD SAILs"
metatags:
    description: "Run SGLang on T-HEAD SAILs."
---

This document describes how to run SGLang on T-HEAD SAILs. If you
encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Prerequisites

Install the T-HEAD SAIL driver and the PPU SDK before installing SGLang, and export
the `PPU_SDK` environment variable pointing at the SDK installation (for
example, `/usr/local/PPU_SDK`). The Triton SAIL backend requires this variable
when compiling kernels at runtime.

Verify that the device is visible:

```bash Command
python3 -c 'import torch; print(torch.cuda.get_device_name())'
# Expected: a ZW-series device, e.g. "ZW810E"
```

## Install SGLang

### Install from Source

Start from an environment with the prerequisites above installed.

```bash Command
git clone https://github.com/sgl-project/sglang.git
cd sglang

python -m pip install --upgrade pip wheel

cp python/pyproject_other.toml python/pyproject.toml
python -m pip install -e "python[all_sail]" \
  --no-build-isolation

# Install sailify
python -m pip install sailify==1.0.0

# Build sgl-kernel for SAIL from source
cd python/sglang/kernels/aot
python setup_sail.py install
```

Launch the server:

```bash Command
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 30000
```

When the server displays `The server is fired up and ready to roll!`, startup is
successful.

## Verify

You can send an OpenAI-compatible request to the engine:

```bash Command
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 32
  }'
```

Or run a benchmark:

```bash Command
python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 4000 \
  --random-input 128 \
  --random-output 128
```
