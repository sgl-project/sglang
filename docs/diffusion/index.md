# SGLang Diffusion

SGLang Diffusion is a high-performance inference framework for image and video generation. It provides native SGLang pipelines, diffusers backend support, an OpenAI-compatible server, and an optimized kernel stack built on both precompiled `sgl-kernel` operators and JIT kernels for key inference paths.

## Key Features

- Broad model support across Wan, Hunyuan, Qwen-Image, FLUX, Z-Image, GLM-Image, and more
- Fast inference with `sgl-kernel`, JIT kernels, scheduler improvements, and caching acceleration
- Multiple interfaces: `sglang generate`, `sglang serve`, and an OpenAI-compatible API
- Multi-platform support for NVIDIA, AMD, Intel XPU, Ascend, Apple Silicon, and Moore Threads

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
- [Compatibility Matrix](compatibility_matrix.md): check model, optimization, and component override support
- [CLI](api/cli.md): run one-off generation jobs or launch a persistent server
- [OpenAI-Compatible API](api/openai_api.md): send image and video requests to the HTTP server
- [Attention Backends](performance/attention_backends.md): choose the best backend for your model and hardware
- [Caching Acceleration](performance/cache/index.md): use Cache-DiT or TeaCache to reduce denoising cost
- [Quantization](quantization.md): load quantized transformer checkpoints
- [Contributing](contributing.md): contribution workflow, adding new models, and CI perf baselines

## Additional Documentation

- [Post-Processing](api/post_processing.md): frame interpolation and upscaling
- [Performance Overview](performance/index.md): overview of attention, caching, and profiling
- [Environment Variables](environment_variables.md): platform, caching, storage, and debugging configuration
- [Support New Models](support_new_models.md): implementation guide for new diffusion pipelines
- [CI Performance](ci_perf.md): performance baseline generation

## References

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Cache-DiT](https://github.com/vipshop/cache-dit)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [xDiT](https://github.com/xdit-project/xDiT)
- [Diffusers](https://github.com/huggingface/diffusers)
