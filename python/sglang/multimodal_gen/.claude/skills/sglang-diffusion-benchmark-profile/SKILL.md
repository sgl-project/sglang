---
name: sglang-diffusion-benchmark-profile
description: Use when benchmarking denoise latency or profiling a diffusion bottleneck in SGLang.
---

# SGLang Diffusion Benchmark and Profile

Use this skill when measuring denoise performance, finding the slow op, checking whether an existing fast path can solve it, or verifying a kernel change in `sglang.multimodal_gen`.

This skill covers diagnosis and fast-path reuse:
- To write a new Triton kernel, use [../sglang-diffusion-triton-kernel/SKILL.md](../sglang-diffusion-triton-kernel/SKILL.md)
- To write a new CUDA JIT kernel, use [../sglang-diffusion-cuda-kernel/SKILL.md](../sglang-diffusion-cuda-kernel/SKILL.md)

## Preflight

Before running any benchmark, profiler, or kernel-validation command:
- use `scripts/diffusion_skill_env.py` to derive the repo root from `sglang.__file__`
- verify the repo is writable
- export `HF_TOKEN` before using gated Hugging Face models such as `black-forest-labs/FLUX.*`
- export `FLASHINFER_DISABLE_VERSION_CHECK=1`
- choose idle GPU(s) before starting perf work

## Main Reference

- [benchmark-and-profile.md](benchmark-and-profile.md) — canonical denoise benchmark and profiling workflow; includes `torch.profiler`, `nsys`, and `ncu`
- [existing-fast-paths.md](existing-fast-paths.md) — map bottlenecks to existing fused kernels and runtime fast paths before writing new code
- [nsight-profiler.md](nsight-profiler.md) — Nsight Systems / Nsight Compute metric interpretation
- [scripts/diffusion_skill_env.py](scripts/diffusion_skill_env.py) — preflight helper: repo root discovery via `sglang.__file__`, write-access probe, benchmark/profile output directories, idle GPU selection
- [scripts/bench_diffusion_rmsnorm.py](scripts/bench_diffusion_rmsnorm.py) — RMSNorm micro-benchmark: JIT CUDA vs PyTorch, correctness check, bandwidth efficiency analysis
- [scripts/bench_diffusion_denoise.py](scripts/bench_diffusion_denoise.py) — end-to-end denoise benchmark preset runner via `sglang generate`; save perf dumps by label and compare them with `compare_perf.py`
