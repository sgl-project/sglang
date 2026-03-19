---
name: diffusion-kernel
description: Index for SGLang Diffusion kernel development skills.
---

# Diffusion Kernel Skills

## Rule: Follow User Kernel Language Preference

If the user explicitly states a preference for **Triton** or **CUDA**, follow that preference when implementing and optimizing kernels (even if the other option could work). Do not “pick for convenience”.

## Directory Layout

```
python/sglang/multimodal_gen/.claude/skills/diffusion-kernel/
├── SKILL.md
├── add-triton-kernel.md
├── add-cuda-kernel.md
├── diffusion-benchmark-and-profile.md
├── nsight-profiler.md
├── use-efficient-diffusion-kernels.md
├── references/
│   ├── kernel-templates.md          # Copy-paste CUDA kernel templates (sglang JIT style)
│   ├── troubleshooting.md           # Build/perf/integration issues & fixes
│   ├── h100-optimization-guide.md   # H100 (sm_90) deep dive
│   ├── a100-optimization-guide.md   # A100 (sm_80) deep dive
│   └── t4-optimization-guide.md     # T4 (sm_75, FP16 only) deep dive
└── scripts/
    ├── bench_diffusion_rmsnorm.py   # RMSNorm micro-benchmark vs PyTorch
    └── bench_diffusion_denoise.py   # End-to-end denoise benchmark (sglang generate)
```

## Index

Before running any benchmark, profiler, or kernel-validation command, use
`scripts/diffusion_skill_env.py` to derive the repo root from `sglang.__file__`,
verify the repo is writable, export `FLASHINFER_DISABLE_VERSION_CHECK=1`, and
choose idle GPU(s) before starting perf work.

- [scripts/diffusion_skill_env.py](scripts/diffusion_skill_env.py)

  Shared preflight helper for all diffusion skill commands. Use it to print the repo root, create benchmark/profile output directories, and choose idle GPUs before running `sglang generate`, torch profiler, nsys, or ncu.

- [add-triton-kernel.md](./add-triton-kernel.md)

  Step-by-step guide for adding a new Triton kernel to SGLang Diffusion's `jit_kernel/diffusion/triton/` module, including authoring, autotune, `torch.compile` compatibility, integration, and tests. Use for fused elementwise ops, norm variants, RoPE variants, or when NPU/CPU fallback is needed.

- [add-cuda-kernel.md](./add-cuda-kernel.md)

  Step-by-step guide for adding a JIT CUDA kernel. CUDA source goes in `jit_kernel/csrc/diffusion/<op>.cuh`; Python wrapper at `jit_kernel/diffusion/<op>.py`. Uses SGLang's JIT compilation system (`load_jit`, `cache_once`) and internal abstractions (`TensorMatcher`, `device::AlignedVector`, `host::LaunchKernel`, `device::warp::reduce_sum`). Use for bandwidth-bound reductions (RMSNorm, LayerNorm) or ops needing fine-grained vectorization and shared memory control. Adapted from [HuggingFace kernels cuda-kernels skill](https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels).

- [use-efficient-diffusion-kernels.md](./use-efficient-diffusion-kernels.md)

  Practical guidance for using SGLang Diffusion fused kernels and fast CUDA paths, including constraints, fallbacks, and where the fused ops are wired into the runtime.

- [diffusion-benchmark-and-profile.md](./diffusion-benchmark-and-profile.md)

  Denoise-stage benchmark and profiling guide for SGLang Diffusion models. Three profiling levels: Level 1 (torch.profiler — kernel time ranking), Level 2 (nsys — category breakdown), Level 3 (ncu — per-kernel bandwidth/occupancy/roofline analysis). **ncu is critical for kernel optimization** — always use it when writing or tuning custom kernels to verify hardware saturation.

- [nsight-profiler.md](./nsight-profiler.md)

  Advanced profiling skill for NVIDIA Nsight Systems / Nsight Compute: collecting traces, reading reports, and interpreting kernel-level performance metrics.

## References (GPU optimization guides, templates, troubleshooting)

Loaded by `add-cuda-kernel.md`. Adapted from [HuggingFace kernels cuda-kernels skill](https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels).

- [references/kernel-templates.md](references/kernel-templates.md) — copy-paste ready sglang JIT CUDA templates: element-wise (SiLU), row-reduction (RMSNorm), fused AdaLN, Python wrapper, test, benchmark
- [references/troubleshooting.md](references/troubleshooting.md) — build errors, performance issues, torch.compile compatibility, kernel injection pitfalls
- [references/h100-optimization-guide.md](references/h100-optimization-guide.md) — H100 (sm_90): AlignedVector benchmarks, warp reductions, occupancy, TMA, PDL
- [references/a100-optimization-guide.md](references/a100-optimization-guide.md) — A100 (sm_80): cp.async, TF32, 2:4 sparsity, H100→A100 migration checklist
- [references/t4-optimization-guide.md](references/t4-optimization-guide.md) — T4 (sm_75): FP16 only, 320 GB/s bandwidth, 64 KB shared mem, 16 GB memory management

## Scripts (runnable benchmarks)

- [scripts/diffusion_skill_env.py](scripts/diffusion_skill_env.py) — preflight helper: repo root discovery via `sglang.__file__`, write-access probe, benchmark/profile output directories, idle GPU selection
- [scripts/bench_diffusion_rmsnorm.py](scripts/bench_diffusion_rmsnorm.py) — RMSNorm micro-benchmark: JIT CUDA vs PyTorch, correctness check, bandwidth efficiency analysis
- [scripts/bench_diffusion_denoise.py](scripts/bench_diffusion_denoise.py) — end-to-end denoise benchmark via `sglang generate`, baseline vs custom kernels comparison table
