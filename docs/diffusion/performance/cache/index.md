# Caching Acceleration for Diffusion Models

SGLang provides multiple caching acceleration strategies for Diffusion Transformer (DiT) models. These strategies can significantly reduce inference time by skipping redundant computation.

## Overview

SGLang supports two complementary caching approaches:

| Strategy | Scope | Mechanism | Best For |
|----------|-------|-----------|----------|
| **Cache-DiT** | Block-level | Skip individual transformer blocks dynamically | Advanced, higher speedup |
| **TeaCache** | Timestep-level | Skip entire denoising steps based on L1 similarity | Simple, built-in |



## Cache-DiT

[Cache-DiT](https://github.com/vipshop/cache-dit) provides block-level caching with
advanced strategies like DBCache and TaylorSeer. It can achieve up to **1.69x speedup**.

See [cache_dit.md](cache_dit.md) for detailed configuration.

### Quick Start

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

### Key Features

- **DBCache**: Dynamic block-level caching based on residual differences
- **TaylorSeer**: Taylor expansion-based calibration for optimized caching
- **SCM**: Step-level computation masking for additional speedup

## TeaCache

TeaCache (Temporal similarity-based caching) accelerates diffusion inference by detecting when consecutive denoising steps are similar enough to skip computation entirely.

See [teacache.md](teacache.md) for detailed documentation.

### Quick Overview

- Tracks L1 distance between modulated inputs across timesteps
- When accumulated distance is below threshold, reuses cached residual
- Supports CFG with separate positive/negative caches

### Supported Models

- Wan (wan2.1, wan2.2)
- Hunyuan (HunyuanVideo)
- Z-Image

For Flux and Qwen models, TeaCache is automatically disabled when CFG is enabled.

## References

- [Cache-DiT Repository](https://github.com/vipshop/cache-dit)
- [TeaCache Paper](https://arxiv.org/abs/2411.14324)
