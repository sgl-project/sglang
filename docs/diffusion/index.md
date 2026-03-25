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

- [Installation](installation.md): install SGLang Diffusion and platform dependencies
- [Compatibility Matrix](compatibility_matrix.md): check model and optimization support
- [Usage](usage.md): CLI, OpenAI-compatible API, post-processing, and quantization
- [Performance](performance/index.md): attention backends, caching, and profiling
- [Reference](reference.md): environment variables and runtime configuration
- [Development](development.md): add models, run CI perf baselines, and contribute

## References

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Cache-DiT](https://github.com/vipshop/cache-dit)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [xDiT](https://github.com/xdit-project/xDiT)
- [Diffusers](https://github.com/huggingface/diffusers)
