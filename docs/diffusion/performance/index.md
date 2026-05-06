# Performance

This section covers the main performance levers for SGLang Diffusion: attention backends, caching acceleration, and profiling.

## Overview

| Optimization | Type | Description |
|--------------|------|-------------|
| **Cache-DiT** | Caching | Block-level caching with DBCache, TaylorSeer, and SCM |
| **TeaCache** | Caching | Timestep-level caching based on temporal similarity |
| **Attention Backends** | Kernel | Optimized attention implementations (FlashAttention, SageAttention, etc.) |
| **Profiling** | Diagnostics | PyTorch Profiler and Nsight Systems guidance |

## Start Here

- Use [Attention Backends](attention_backends.md) to choose the best backend for your model and hardware.
- Use [Caching Acceleration](cache/index.md) to reduce denoising cost with Cache-DiT or TeaCache.
- Use [Profiling](profiling.md) when you need to diagnose a bottleneck rather than guess.

## Caching at a Glance

- [Cache-DiT](cache/cache_dit.md) is block-level caching for diffusers pipelines and higher speedup-oriented tuning.
- [TeaCache](cache/teacache.md) is timestep-level caching built into SGLang model families.

```{toctree}
:maxdepth: 1

attention_backends
cache/index
profiling
```

## Current Baseline Snapshot

For Ring SP benchmark details, see:

- [Ring SP Performance](ring_sp_performance.md)

## References

- [Cache-DiT Repository](https://github.com/vipshop/cache-dit)
- [TeaCache Paper](https://arxiv.org/abs/2411.14324)
