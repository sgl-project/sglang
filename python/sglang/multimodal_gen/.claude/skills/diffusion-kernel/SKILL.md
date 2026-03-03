---
name: diffusion-kernel
description: Index for SGLang Diffusion kernel development skills.
---

# Diffusion Kernel Skills

## Directory Layout

```
python/sglang/multimodal_gen/.claude/skills/diffusion-kernel/
├── SKILL.md
├── add-triton-kernel.md
├── diffusion-benchmark-and-profile.md
├── nsight-profiler.md
└── use-efficient-diffusion-kernels.md
```

## Index

- [add-triton-kernel.md](./add-triton-kernel.md)

  Step-by-step guide for adding a new Triton kernel to SGLang Diffusion's `jit_kernel` module, including authoring, autotune, `torch.compile` compatibility, integration, and tests.

- [use-efficient-diffusion-kernels.md](./use-efficient-diffusion-kernels.md)

  Practical guidance for using SGLang Diffusion fused kernels and fast CUDA paths, including constraints, fallbacks, and where the fused ops are wired into the runtime.

- [diffusion-benchmark-and-profile.md](./diffusion-benchmark-and-profile.md)

  End-to-end benchmarking and profiling guide for SGLang Diffusion models, including denoise latency measurement, per-layer breakdown, and regression tracking.

- [nsight-profiler.md](./nsight-profiler.md)

  Advanced profiling skill for NVIDIA Nsight Systems / Nsight Compute: collecting traces, reading reports, and interpreting kernel-level performance metrics.
