# SGLang Diffusion

SGLang Diffusion is an inference framework for accelerated image and video generation using diffusion models. It provides an end-to-end unified pipeline with optimized kernels and an efficient scheduler loop.

## Key Features

- **Broad Model Support**: Wan series, FastWan series, Hunyuan, Qwen-Image, Qwen-Image-Edit, Flux, Z-Image, GLM-Image, and more
- **Fast Inference**: Optimized kernels, efficient scheduler loop, and Cache-DiT acceleration
- **Ease of Use**: OpenAI-compatible API, CLI, and Python SDK
- **Multi-Platform**: NVIDIA GPUs (H100, H200, A100, B200, 4090), AMD GPUs (MI300X, MI325X) and Ascend NPU (A2, A3)

---

## Quick Start

### Installation

```bash
uv pip install "sglang[diffusion]" --prerelease=allow
```

See [Installation Guide](installation.md) for more installation methods and ROCm-specific instructions.

### Basic Usage

Generate an image with the CLI:

```bash
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains" \
    --save-output
```

Or start a server with the OpenAI-compatible API:

```bash
sglang serve --model-path Qwen/Qwen-Image --port 30010
```

---

## Documentation

### Getting Started

- **[Installation](installation.md)** - Install SGLang Diffusion via pip, uv, Docker, or from source
- **[Compatibility Matrix](compatibility_matrix.md)** - Supported models and optimization compatibility

### Usage

- **[CLI Documentation](api/cli.md)** - Command-line interface for `sglang generate` and `sglang serve`
- **[OpenAI API](api/openai_api.md)** - OpenAI-compatible API for image/video generation and LoRA management

### Performance Optimization

- **[Performance Overview](performance/index.md)** - Overview of all performance optimization strategies
- **[Attention Backends](performance/attention_backends.md)** - Available attention backends (FlashAttention, SageAttention, etc.)
- **[Caching Strategies](performance/cache/)** - Cache-DiT and TeaCache acceleration
- **[Profiling](performance/profiling.md)** - Profiling techniques with PyTorch Profiler and Nsight Systems

### Reference

- **[Environment Variables](environment_variables.md)** - Configuration via environment variables
- **[Support New Models](support_new_models.md)** - Guide for adding new diffusion models
- **[Contributing](contributing.md)** - Contribution guidelines and commit message conventions
- **[CI Performance](ci_perf.md)** - Performance baseline generation script

---

## CLI Quick Reference

### Generate (one-off generation)

```bash
sglang generate --model-path <MODEL> --prompt "<PROMPT>" --save-output
```

### Serve (HTTP server)

```bash
sglang serve --model-path <MODEL> --port 30010
```

### Enable Cache-DiT acceleration

```bash
SGLANG_CACHE_DIT_ENABLED=true sglang generate --model-path <MODEL> --prompt "<PROMPT>"
```

---

## References

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Cache-DiT](https://github.com/vipshop/cache-dit)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [xDiT](https://github.com/xdit-project/xDiT)
- [Diffusers](https://github.com/huggingface/diffusers)
