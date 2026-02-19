# Performance Optimization

SGLang-Diffusion provides multiple performance optimization strategies to accelerate inference. This section covers all available performance tuning options.

## Overview

| Optimization | Type | Description |
|--------------|------|-------------|
| **Cache-DiT** | Caching | Block-level caching with DBCache, TaylorSeer, and SCM |
| **TeaCache** | Caching | Timestep-level caching using L1 similarity |
| **Attention Backends** | Kernel | Optimized attention implementations (FlashAttention, SageAttention, etc.) |
| **Profiling** | Diagnostics | PyTorch Profiler and Nsight Systems guidance |

## Caching Strategies

SGLang supports two complementary caching approaches:

### Cache-DiT

[Cache-DiT](https://github.com/vipshop/cache-dit) provides block-level caching with advanced strategies. It can achieve up to **1.69x speedup**.

**Quick Start:**
```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

**Key Features:**
- **DBCache**: Dynamic block-level caching based on residual differences
- **TaylorSeer**: Taylor expansion-based calibration for optimized caching
- **SCM**: Step-level computation masking for additional speedup

See [Cache-DiT Documentation](cache/cache_dit.md) for detailed configuration.

### TeaCache

TeaCache (Temporal similarity-based caching) accelerates diffusion inference by detecting when consecutive denoising steps are similar enough to skip computation entirely.

**Quick Overview:**
- Tracks L1 distance between modulated inputs across timesteps
- When accumulated distance is below threshold, reuses cached residual
- Supports CFG with separate positive/negative caches

**Supported Models:** Wan (wan2.1, wan2.2), Hunyuan (HunyuanVideo), Z-Image

See [TeaCache Documentation](cache/teacache.md) for detailed configuration.

## Attention Backends

Different attention backends offer varying performance characteristics depending on your hardware and model:

- **FlashAttention**: Fastest on NVIDIA GPUs with fp16/bf16
- **SageAttention**: Alternative optimized implementation
- **xformers**: Memory-efficient attention
- **SDPA**: PyTorch native scaled dot-product attention

See [Attention Backends](attention_backends.md) for platform support and configuration options.

## Profiling

To diagnose performance bottlenecks, SGLang-Diffusion supports profiling tools:

- **PyTorch Profiler**: Built-in Python profiling
- **Nsight Systems**: GPU kernel-level analysis

See [Profiling Guide](profiling.md) for detailed instructions.

## References

- [Cache-DiT Repository](https://github.com/vipshop/cache-dit)
- [TeaCache Paper](https://arxiv.org/abs/2411.14324)
