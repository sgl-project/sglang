---
name: sglang-diffusion-benchmark-profile
description: Use when benchmarking denoise latency or profiling a diffusion bottleneck in SGLang.
---

# SGLang Diffusion Benchmark and Profile

Use this skill when measuring denoise performance, finding the slow op, checking whether an existing fast path can solve it, or verifying that a hotspot is real before any kernel work in `sglang.multimodal_gen`.

This skill is diagnosis-first. It owns:
- checked-in denoise benchmark presets
- perf dump collection and before/after comparison
- `torch.profiler` trace capture and quick hotspot ranking
- mapping hot kernels back to known fast paths and fusion families
- handing confirmed kernel work to a specialized optimization skill such as [../sglang-diffusion-ako4all-kernel/SKILL.md](../sglang-diffusion-ako4all-kernel/SKILL.md)

This skill does not own low-level kernel authoring or standalone Nsight workflows.

## Preflight

Before running any benchmark, profiler, or kernel-validation command:
- use `scripts/diffusion_skill_env.py` to derive the repo root from `sglang.__file__`
- verify the repo is writable
- export `HF_TOKEN` before using gated Hugging Face models such as `black-forest-labs/FLUX.*`
- export `FLASHINFER_DISABLE_VERSION_CHECK=1`
- choose idle GPU(s) before starting perf work

## Native Backend Gate

All diffusion benchmark and profiling results owned by this skill must come from the native SGLang diffusion backend.

Treat any of the following as a hard stop condition:
- `Falling back to diffusers backend`
- `Using diffusers backend`
- `Loaded diffusers pipeline`

If any benchmark, perf-dump, or `torch.profiler` command prints one of those signals:
- stop the workflow immediately
- do not keep the generated numbers or traces as SGLang benchmark evidence
- do not continue to hotspot classification or kernel work
- first fix model resolution, pipeline selection, overlay/materialization, or other backend-selection issues so the model runs on the native SGLang diffusion path

## Main Reference

- [benchmark-and-profile.md](benchmark-and-profile.md) — canonical denoise benchmark, perf dump, and `torch.profiler` workflow; uses the checked-in nightly-aligned presets, plus `LTX-2`, `LTX-2.3` one-stage, and `LTX-2.3` two-stage benchmark recipes
- [existing-fast-paths.md](existing-fast-paths.md) — map bottlenecks to existing fused kernels, packed QKV paths, fused `QK norm + RoPE`, and distributed overlap patterns before proposing new code
- [scripts/diffusion_skill_env.py](scripts/diffusion_skill_env.py) — preflight helper: repo root discovery via `sglang.__file__`, write-access probe, benchmark/profile output directories, idle GPU selection
- [scripts/bench_diffusion_denoise.py](scripts/bench_diffusion_denoise.py) — end-to-end denoise benchmark preset runner via `sglang generate`; pins `--backend=sglang`, supports `--no-torch-compile`, and saves perf dumps by label for `compare_perf.py`

## Opportunity Discovery Rule

Before calling a diffusion hotspot "new", first classify it with `existing-fast-paths.md`.

Always rule out these existing families first:
- Z-Image residual-form modulation
- fused diffusion `QK norm + RoPE`
- NVFP4 / Nunchaku packed QKV
- Nunchaku fused GELU MLP
- Ulysses / USP attention overlap
- turbo-layer async all-to-all overlap
- `torch.compile` compute / communication reorder
- dual-stream diffusion execution

If the user explicitly requires `torch.compile` to stay off, do not use the
default benchmark preset invocation unchanged. Either pass the checked-in
benchmark helper its no-compile switch or run the equivalent manual command
without `--enable-torch-compile`.

For FLUX-family manual profiling runs with a quantized transformer override:
- use `sglang generate` directly
- pass the override as `--transformer-path <dir>`
- prefer `--prompt-path <file>` when also fixing `--output-file-name`
- if the base model is already cached locally and the machine has unreliable HF access, use the local cached `--model-path` plus `HF_HUB_OFFLINE=1`
- remember that `--profile` changes latency substantially; use the non-profile perf dump for the real before/after benchmark claim
