# SGLang Diffusion

SGLang Diffusion is a high-performance inference framework for image and video generation. It provides native SGLang pipelines, diffusers backend support, optimized kernels, and an OpenAI-compatible server.

## Key Features

- Broad model support across Wan, Hunyuan, Qwen-Image, FLUX, Z-Image, GLM-Image, and more
- Fast inference with optimized kernels, scheduler improvements, and caching acceleration
- Multiple interfaces: `sglang generate`, `sglang serve`, and an OpenAI-compatible API
- Multi-platform support for NVIDIA, AMD, Ascend, Apple Silicon, and Moore Threads

## Quick Start

```bash
uv pip install "sglang[diffusion]" --prerelease=allow
```

```bash
sglang generate --model-path Qwen/Qwen-Image \
  --prompt "A beautiful sunset over the mountains" \
  --save-output
```

```bash
sglang serve --model-path Qwen/Qwen-Image --port 30010
```

## Start Here

### Getting Started

- **[Installation](installation.md)** - Install SGLang Diffusion via pip, uv, Docker, or from source
- **[Compatibility Matrix](compatibility_matrix.md)** - Supported models and optimization compatibility

### Usage

- **[CLI Documentation](api/cli.md)** - Command-line interface for `sglang generate` and `sglang serve`
- **[Quantization](quantization.md)** - Quantized transformer checkpoint usage and supported quantization families
- **[OpenAI API](api/openai_api.md)** - OpenAI-compatible API for image/video generation and LoRA management
- **[Post-Processing](api/post_processing.md)** - Frame interpolation (RIFE) and upscaling (Real-ESRGAN)

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
