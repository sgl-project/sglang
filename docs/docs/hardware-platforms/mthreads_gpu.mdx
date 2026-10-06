---
title: "Moore Threads GPUs"
metatags:
    description: "Run SGLang on Moore Threads GPUs."
---

This document describes how to run SGLang on Moore Threads GPUs. If you
encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

SGLang uses the MUSA runtime through
[`torch_musa`](https://github.com/MooreThreads/torch_musa) and
[`torchada`](https://github.com/MooreThreads/torchada). The
`python[all_musa]` extra installs the MUSA torch, Triton, TileLang, MATE, and
runtime Python stack from the Moore Threads package index, with public PyPI as
the fallback for common Python dependencies.

## Prerequisites

Install the Moore Threads driver and MUSA toolkit before installing SGLang. For
MTT S5000, follow the [official driver installation guide](https://docs.mthreads.com/driver-linux-server/driver-linux-server-doc-online/MTT_S5000/install_guide).

## Install SGLang

You can install SGLang using one of the methods below.

### Install from Source

Start from an environment with the prerequisites above installed.

```bash Command
git clone https://github.com/sgl-project/sglang.git
cd sglang

python -m pip install --upgrade pip "setuptools<82" wheel
cp python/pyproject_other.toml python/pyproject.toml
python -m pip install -e "python[all_musa]" \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --extra-index-url https://pypi.org/simple \
  --trusted-host dl.mthreads.com \
  --no-build-isolation

# Ensure the MUSA Triton wheel (with triton.backends.mtgpu) is installed.
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --trusted-host dl.mthreads.com \
  triton==3.2.0
python -c 'import triton.backends.mtgpu'

cd python/sglang/kernels/aot
cp pyproject_musa.toml pyproject.toml
MTGPU_TARGET=mp_31 python setup_musa.py install
```

### Install Using Docker

The Dockerfile installs the MUSA runtime and Python stack declared by
`python/pyproject_other.toml`.

```bash Command
git clone https://github.com/sgl-project/sglang.git
cd sglang

docker build -f docker/musa.Dockerfile -t sglang:main-musa .
```

Run the image with MUSA devices exposed by the host container toolkit.

```bash Command
alias drun='docker run -it --rm --network=host \
  --env MTHREADS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env MTHREADS_DRIVER_CAPABILITIES=all \
  --shm-size=32g \
  -v ~/.cache/huggingface:/root/.cache/huggingface'

drun sglang:main-musa \
  python3 -c 'import triton.backends.mtgpu; print("MUSA Triton backend OK")'
```

Launch the server:

```bash Command
drun sglang:main-musa \
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
drun sglang:main-musa \
  python3 -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --num-prompts 4000 \
  --random-input 128 \
  --random-output 128
```
