---
name: sglang-diffusion-benchmark-profile
description: Use when benchmarking denoise latency or profiling a diffusion bottleneck in SGLang.
---

# SGLang Diffusion Benchmark and Profile

Use this skill when measuring denoise performance, finding the slow op, checking whether an existing fast path can solve it, or verifying that a hotspot is real before any kernel work in `sglang.multimodal_gen`.

This skill is diagnosis-first. It owns:
- checked-in denoise benchmark presets
- same-GPU quality/BCG applicability checks with repeated lossless, extra-high, and high rows
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
- choose idle GPU(s) before starting perf work; for a comparison matrix, hold
  the same GPU set and verify it has no foreign process at every run boundary

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

- [benchmark-and-profile.md](benchmark-and-profile.md) — canonical denoise benchmark, perf dump, and `torch.profiler` workflow; uses checked-in nightly-aligned presets plus current-source extras such as LongCat image/edit, Qwen base edit/layered, SD3.5, SANA-Video/SANA-WM, LingBot Video/World, Cosmos3 Edge/Super I2V/distilled and the explicit Super TP2 x CFG2 comparator, LTX-2.5 and its diffusion decoder, MiniMax-H3, FLUX.2 Klein, Ideogram4, ERNIE/GLM/SANA image models, FastWan2.1/2.2, the Blackwell-only Wan2.2 NVFP4 comparator, `LTX-2.3`, HunyuanVideo, MOVA, Helios, image edit, Hunyuan3D shape, and a separate Pi0.5 action-policy lane
- [existing-fast-paths.md](existing-fast-paths.md) — map bottlenecks to existing fused kernels, packed QKV paths, fused `QK norm + RoPE`, distributed overlap patterns, and open optimization PRs before proposing new code
- [scripts/diffusion_skill_env.py](scripts/diffusion_skill_env.py) — preflight helper: repo root discovery from the skill's owning checkout before falling back to `sglang.__file__`, write-access probe, benchmark/profile output directories, idle GPU selection
- [scripts/bench_diffusion_denoise.py](scripts/bench_diffusion_denoise.py) — end-to-end denoise benchmark preset runner via `sglang generate`; defaults to eager/lossless, supports explicit quality and BCG comparators plus a same-GPU applicability matrix, rejects invalid BCG capture/fallback logs and late high-quality DiT fusion mounts, forces H3 to its eager consistency mode, enables synchronized stage attribution, validates nightly preset drift, and can clean one isolated model cache after the full matrix in a `finally` block with a JSONL ledger

## Opportunity Discovery Rule

Before calling a diffusion hotspot "new", first classify it with `existing-fast-paths.md`.

Always rule out these existing families first:
- HunyuanVideo VAE GroupNorm+SiLU
- LTX upsampler GroupNorm+SiLU
- Z-Image bf16-native Triton RMSNorm scale/tanh-residual modulation
- SANA packed self-attention Q/K/V and cross-attention K/V GEMMs
- SANA-Video's packed projections and request-scoped BF16-input linear
  attention at `quality=extra-high` or `quality=high`; keep the second attention GEMM in FP32 and
  compare against `quality=lossless` before changing its precision further
- SANA-Video reuse of SANA's bit-exact bias/activation, residual-gate, and
  LayerNorm-modulation fast paths before adding video-only kernels
- MiniMax-H3 indexed modulation, fused QK norm + RoPE, packed Ulysses QKV,
  USP relayout, and batched TP AdaLN collectives
- bit-exact diffusion adaLN modulation and fused LayerNorm + modulation for
  FLUX.1, GLM-Image, and SANA
- request-scoped DiT and VAE fast paths at `quality=extra-high` or `quality=high`
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

For kernel/BCG discovery, run `--quality-bcg-matrix`. It executes Eager/BCG as
A-B-B-A at `lossless`, then repeats the pair at `extra-high` and `high`, on
one locked GPU set and one isolated checkpoint cache. The extra-high/high+BCG
rows are applicability checks, not presumed-valid performance cells. A BCG row is invalid unless the log
contains `[Diffusion BCG] captured` and contains no support-disable,
capture-failure, serving-signature-miss, or late quality-fusion marker. In
particular, a request-scoped DiT fusion mounted after lossless warmup capture
would be bypassed by replay; reject that row even when capture and signature
checks pass. For video presets, the helper declares both the request resolution
and `--warmup-num-frames` so the synthetic BCG warmup captures the requested
temporal shape. Treat any remaining temporal or conditioning signature miss as
Eager fallback, not as a valid BCG measurement.

A zero process exit is not sufficient evidence: every accepted row must also
contain its requested perf dump and a generated image, video, audio, or 3D mesh
file.
The helper gives every cell a unique output name and rejects missing artifacts.

On machines with a read-only Hugging Face cache, combine
`--model-cache-root <task-owned-dir>` with one or more
`--seed-model-cache-root <read-only-HF-home-or-hub>` options. The helper exposes
cached repos through a task-owned copy-on-write directory overlay, downloads
misses only into the isolated cache, and removes links plus new downloads in
its normal cleanup finally block without modifying the seed cache.

Keep prompt, negative prompt, seed, shape, steps, guidance, dtype, topology,
and residency fixed. Lossless comparisons require byte-identical artifacts.
For `quality=extra-high` and `quality=high`, report aggregate and worst-frame SSIM/PSNR; the repository
defaults are 0.95/28 dB for images and 0.92/24 dB for video unless the model's
checked-in consistency metadata defines a different threshold. A performance
PR needs repeated saved-request e2e improvement of at least 1.5%, a
representative profile, and before/after image or video evidence.

MiniMax-H3 is always an eager consistency case on current main. Use
`--model minimax-h3-t2va`; its preset writes the H3 request fields through a
generated config and suppresses the helper's global compile default. Do not
turn the model's nominal BCG support gate into a performance claim: prompt-
dependent packed-sequence host boundaries can differ between warmup and the
serving request. A valid H3 BCG experiment must prove that every captured
segment replays, keeps the MP4 byte-identical, and does not trade latency for
the extra graph memory.

For FLUX-family manual profiling runs with a quantized transformer override:
- use `sglang generate` directly
- pass the override as `--transformer-path <dir>`
- prefer `--prompt-path <file>` when also fixing `--output-file-name`
- if the base model is already cached locally and the machine has unreliable HF access, use the local cached `--model-path` plus `HF_HUB_OFFLINE=1`
- remember that `--profile` changes latency substantially; use the non-profile perf dump for the real before/after benchmark claim
