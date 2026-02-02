# Diffusion Models

> **Documentation has moved.** All SGLang Diffusion (image/video) documentation is now in **[docs/diffusion/](../diffusion/README.md)**. This page is kept for backward compatibility and as a brief overview.

SGLang Diffusion is an inference framework for accelerated image and video generation using diffusion models. It provides an end-to-end unified pipeline with optimized kernels from sgl-kernel and an efficient scheduler loop.

## Key Features

- **Broad Model Support**: Wan series, FastWan series, Hunyuan, Qwen-Image, Qwen-Image-Edit, Flux, Z-Image, GLM-Image, and more
- **Fast Inference**: Optimized kernels from sgl-kernel, efficient scheduler loop, and Cache-DiT acceleration
- **Ease of Use**: OpenAI-compatible API, CLI, and Python SDK
- **Multi-Platform**: NVIDIA GPUs (H100, H200, A100, B200, 4090) and AMD GPUs (MI300X, MI325X)

## Full Documentation

All detailed documentation lives under **[docs/diffusion/](../diffusion/README.md)**:

| Topic | Link |
|-------|------|
| **Install** (pip, source, Docker) | [install.md](../diffusion/install.md) |
| **Install (ROCm/AMD)** | [install_rocm.md](../diffusion/install_rocm.md) |
| **CLI** (serve, generate, config) | [cli.md](../diffusion/cli.md) |
| **OpenAI API** (images, videos, LoRA) | [openai_api.md](../diffusion/openai_api.md) |
| **Compatibility matrix & LoRAs** | [support_matrix.md](../diffusion/support_matrix.md) |
| **Caching** (overview) | [cache/caching.md](../diffusion/cache/caching.md) |
| **Cache-DiT** | [cache/cache_dit.md](../diffusion/cache/cache_dit.md) |
| **TeaCache** | [cache/teacache.md](../diffusion/cache/teacache.md) |
| **Environment variables** | [environment_variables.md](../diffusion/environment_variables.md) |
| **Attention backends** | [attention_backends.md](../diffusion/attention_backends.md) |
| **Profiling** | [profiling.md](../diffusion/profiling.md) |
| **CI perf baselines** | [ci_perf.md](../diffusion/ci_perf.md) |
| **Contributing** | [contributing.md](../diffusion/contributing.md) |
| **Support new models** (developer guide) | [support_new_models.md](../diffusion/support_new_models.md) |

For the canonical entry point, see **[SGLang Diffusion Documentation](../diffusion/README.md)**.
