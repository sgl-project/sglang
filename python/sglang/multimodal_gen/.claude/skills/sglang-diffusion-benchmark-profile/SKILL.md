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
- packaging confirmed kernel work with enough evidence for the appropriate kernel, Nsight, or framework-specific optimization workflow

This skill does not own low-level kernel authoring or standalone Nsight workflows.

## Preflight

Before running any benchmark, profiler, or kernel-validation command:
- use `scripts/diffusion_skill_env.py` to derive the repo root from `sglang.__file__`
- verify the repo is writable
- export `HF_TOKEN` before using gated Hugging Face models such as `black-forest-labs/FLUX.*`
- export `FLASHINFER_DISABLE_VERSION_CHECK=1`
- set `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1` when comparing stage-level
  denoise/decode timings; the preset helper sets it by default unless the
  caller explicitly overrides it
- for downloaded checkpoints, use the preset helper's task-owned
  `--model-cache-root` together with `--cleanup-model-cache`; verify the JSONL
  ledger reports zero residual weight files before moving to the next model
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

- [benchmark-and-profile.md](benchmark-and-profile.md) — canonical denoise benchmark, perf dump, and `torch.profiler` workflow; uses checked-in nightly-aligned presets plus current-source extras such as LongCat-Image, SANA-Video, LingBot Video MoE, Cosmos3 Edge/distilled, LTX-2.5 and its diffusion decoder, MiniMax-H3, FLUX.2 Klein, Ideogram4, ERNIE/GLM/SANA image models, FastWan2.2, `LTX-2.3`, HunyuanVideo, MOVA, Helios, image edit, and Hunyuan3D shape
- [existing-fast-paths.md](existing-fast-paths.md) — map bottlenecks to existing fused kernels, packed QKV paths, fused `QK norm + RoPE`, distributed overlap patterns, and open optimization PRs before proposing new code
- [scripts/diffusion_skill_env.py](scripts/diffusion_skill_env.py) — preflight helper: repo root discovery via `sglang.__file__`, write-access probe, benchmark/profile output directories, idle GPU selection
- [scripts/bench_diffusion_denoise.py](scripts/bench_diffusion_denoise.py) — end-to-end denoise benchmark preset runner via `sglang generate`; defaults to eager, supports opt-in `--torch-compile`, forces H3 to its eager consistency mode, enables synchronized stage attribution, validates nightly preset drift, and can clean an isolated model cache in a `finally` block with a JSONL ledger

## Opportunity Discovery Rule

Before calling a diffusion hotspot "new", first classify it with `existing-fast-paths.md`.

Always rule out these existing families first:
- HunyuanVideo VAE GroupNorm+SiLU
- LTX upsampler GroupNorm+SiLU
- Z-Image bf16-native Triton RMSNorm scale/tanh-residual modulation
- SANA packed self-attention Q/K/V and cross-attention K/V GEMMs
- SANA-Video's packed projections and request-scoped BF16-input linear
  attention at `quality=high`; keep the second attention GEMM in FP32 and
  compare against `quality=lossless` before changing its precision further
- SANA-Video reuse of SANA's bit-exact bias/activation, residual-gate, and
  LayerNorm-modulation fast paths before adding video-only kernels
- MiniMax-H3 indexed modulation, fused QK norm + RoPE, packed Ulysses QKV,
  USP relayout, and batched TP AdaLN collectives
- bit-exact diffusion adaLN modulation and fused LayerNorm + modulation for
  FLUX.1, GLM-Image, and SANA
- request-scoped `quality=high` DiT and VAE fast paths
- Wan causal-VAE cache/padding and DupUp3D data-movement fusions
- fused diffusion `QK norm + RoPE`
- LTX2 split RoPE
- LTX2 residual-gate add
- LTX-2.5 diffusion-decoder NATTEN selection before interpreting a
  FlexAttention fallback trace
- varlen USP attention pack/scatter
- NVFP4 / Nunchaku packed QKV
- Nunchaku fused GELU MLP
- Ulysses / USP attention overlap
- turbo-layer async all-to-all overlap
- `torch.compile` compute / communication reorder
- breakable CUDA graph capture for supported fixed-resolution pipelines
- dual-stream diffusion execution

The checked-in helper defaults to eager. Use `--torch-compile` only for a
controlled comparator, never for the eager ground truth. The legacy
`--no-torch-compile` spelling remains accepted but is redundant.

MiniMax-H3 is always an eager consistency case on current main. Use
`--model minimax-h3-t2va`; its preset writes the H3 request fields through a
generated config and suppresses the helper's global compile default.

For FLUX-family manual profiling runs with a quantized transformer override:
- use `sglang generate` directly
- pass the override as `--transformer-path <dir>`
- prefer `--prompt-path <file>` when also fixing `--output-file-name`
- if the base model is already cached locally and the machine has unreliable HF access, use the local cached `--model-path` plus `HF_HUB_OFFLINE=1`
- remember that `--profile` changes latency substantially; use the non-profile perf dump for the real before/after benchmark claim
