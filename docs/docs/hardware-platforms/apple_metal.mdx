---
title: "Apple Silicon with Metal"
metatags:
    description: "Run SGLang on Apple Silicon using the Metal backend."
---

This document describes how to run the SGLang serving runtime on Apple Silicon
using [MLX](https://opensource.apple.com/projects/mlx/). SGLang Diffusion uses
PyTorch MPS instead; see its [installation guide](/docs/sglang-diffusion/installation#platform-specific-apple-mps).
If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Prerequisites

The MLX runtime requires Apple Silicon with macOS 14 or newer, stable PyTorch
2.13.x, and stable MLX 0.32.0 or newer. The `srt_mps` extra installs PyTorch
2.13.0 and MLX 0.32.0 or newer; startup accepts stable PyTorch 2.13 patch
releases and newer stable MLX releases.

With `SGLANG_USE_MLX=1`, SGLang validates both framework versions and Metal
availability during argument initialization and stops before resolving or
downloading a model when the runtime is incompatible.

Building the optional native Metal kernels in `sgl-kernel` requires the Metal
shader compiler from the full Xcode application. The standalone Xcode Command
Line Tools are not sufficient. After installing Xcode, select it with:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

Verify the compiler with `xcrun -sdk macosx metal --version`.

## Install SGLang

You can install SGLang using one of the methods below.

### Install from Source

```bash
# Use the default branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Create and activate a virtual environment
uv venv -p 3.12 sglang-metal
source sglang-metal/bin/activate

# (Optional) Compile sgl-kernel
uv pip install --upgrade pip
uv run python/sglang/kernels/aot/setup_metal.py install

# Install sglang python package along with diffusion support
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```

## Launch of the Serving Engine

Launch the server with:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model <MODEL_ID_OR_PATH> \
  --disable-cuda-graph \
  --host 0.0.0.0
```

**Key Parameters Explained:**

1. `SGLANG_USE_MLX=1` - Enables the use of MLX as the SGLang runtime backend (if disabled, SGLang will fall back to `torch.mps`, which has less support)
2. `--disable-cuda-graph` - Disables usage of CUDA graph, which is not relevant for Apple Metal.
3. `--disable-overlap-schedule` - Disables overlap scheduling (enabled/not present by default) achieved using MLX's `async_eval()`
4. `SGLANG_MLX_USE_CUSTOM_ROPE=1` - Enables the optional custom Metal RoPE kernel. It is disabled by default, so the MLX backend uses the standard RoPE path unless you opt in for A/B testing.
5. `SGLANG_MLX_FUSE_SWIGLU=1` - Enables the use of fused Swish-Gated Linear Unit kernel (disabled by default)
6. `SGLANG_MLX_CLEAR_CACHE_STEPS=256` - Sets the number of decode steps before clearing the MLX cache (256 by default)

## Quantization

The MLX backend supports two quantization paths on Apple Silicon:

1. **Pre-quantized HF repos.** Any `mlx-community/<model>-4bit` (or `-8bit`) repo loads directly through `mlx_lm.load(...)` — no extra flag needed.
   ```bash
   SGLANG_USE_MLX=1 python -m sglang.launch_server \
     --model-path mlx-community/Qwen3-0.6B-4bit \
     --disable-cuda-graph
   ```
2. **On-the-fly quantization.** For any fp16 model, pass `--quantization mlx_q4` or `--quantization mlx_q8` to have sglang quantize the weights at load time via `mlx_lm.utils.quantize_model` (group size 64, the mlx-community default). The quantized weights stay in process memory; the on-disk model is untouched.
   ```bash
   SGLANG_USE_MLX=1 python -m sglang.launch_server \
     --model-path Qwen/Qwen3-0.6B \
     --quantization mlx_q4 \
     --disable-cuda-graph
   ```
   Expected log line:
   ```
   Quantizing MLX model on-the-fly: bits=4 group_size=64 (preset=mlx_q4)
   Quantization complete in 0.13s — active mem: 1.11 GB -> 0.31 GB (71.9% reduction)
   ```
   The MLX backend silently ignores `--quantization mlx_q4` when the model is already quantized in its HF config (path 1), so the same flag is safe to pass either way.

## Benchmarking with Requests

`sglang.benchmark.one_batch` calls the synchronous prefill/decode methods directly without going through the scheduler and the overlap code path.

`sglang.benchmark.offline_throughput` can toggle overlap scheduling as it uses the scheduler and the overlap code path by using the flag `--disable-overlap-schedule`.

### Throughput Testing

Basic synchronous one batch throughput:
```bash
SGLANG_USE_MLX=1 python -m sglang.benchmark.one_batch \
  --model-path <MODEL_ID_OR_PATH> \
  --disable-cuda-graph \
  --tp-size 1 \
  --batch-size 1 \
  --input-len 60 \
  --output-len 10
```

Synchronous offline throughput:
```bash
SGLANG_USE_MLX=1 python -m sglang.benchmark.offline_throughput \
  --model-path <MODEL_ID_OR_PATH> \
  --disable-cuda-graph \
  --num-prompts 1 \
  --disable-overlap-schedule
```

Asynchronous offline throughput:
```bash
SGLANG_USE_MLX=1 python -m sglang.benchmark.offline_throughput \
  --model-path <MODEL_ID_OR_PATH> \
  --disable-cuda-graph \
  --num-prompts 1
```
