---
name: diffusion-optimal-perf
description: Guide for achieving optimal performance with SGLang-Diffusion. Covers all perf-related CLI flags, env vars, and best practices for lossless and lossy speedup.
---

# SGLang-Diffusion: Optimal Performance Guide

Use this guide when a user asks how to speed up diffusion inference, reduce latency, lower VRAM usage, or tune SGLang-Diffusion for production.

Before running any `sglang generate` command below inside the diffusion container:
- derive the repo root from `python3 -c "import os, sglang; print(os.path.abspath(os.path.join(os.path.dirname(sglang.__file__), '..', '..')))"` and `cd` there
- export `FLASHINFER_DISABLE_VERSION_CHECK=1`
- verify the repo is writable if you expect perf dumps or outputs
- choose idle GPU(s) first; reuse `diffusion-kernel/scripts/diffusion_skill_env.py` when doing perf work

Reference: [SGLang-Diffusion Advanced Optimizations Blog](https://lmsys.org/blog/2026-02-16-sglang-diffusion-advanced-optimizations/)

---

## Section 1: Lossless Optimizations

These options **do not** affect output quality. The generated images/videos are numerically identical (or within floating-point rounding) to the baseline.

| Option | CLI Flag / Env Var | What It Does | Speedup | Limitations / Notes |
|---|---|---|---|---|
| **torch.compile** | `--enable-torch-compile` | Applies `torch.compile` to the DiT forward pass, fusing ops and reducing kernel launch overhead. | ~1.2–1.5x on denoising | First request is slow (compilation). May cause minor precision drifts due to [PyTorch issue #145213](https://github.com/pytorch/pytorch/issues/145213). Pair with `--warmup` for best results. |
| **Warmup** | `--warmup` | Runs dummy forward passes to warm up CUDA caches, JIT, and `torch.compile`. Eliminates cold-start penalty. | Removes first-request latency spike | Adds startup time. Without `--warmup-resolutions`, warmup happens on first request. |
| **Warmup Resolutions** | `--warmup-resolutions 256x256 720x720` | Pre-compiles and warms up specific resolutions at server startup (instead of lazily on first request). | Faster first request per resolution | Each resolution adds to startup time. Serving mode only; useful when you know your target resolutions in advance. |
| **Multi-GPU (SP)** | `--num-gpus N --ulysses-degree N` | Sequence parallelism across GPUs. Shards sequence tokens (not frames) to minimize padding. | Near-linear scaling with N GPUs | Requires NCCL; inter-GPU bandwidth matters. `ulysses_degree * ring_degree = sp_degree`. |
| **CFG Parallel** | `--enable-cfg-parallel` | Runs conditional and unconditional CFG branches in parallel across GPUs. **For CFG models with multi-GPU, always prefer `--enable-cfg-parallel` + Ulysses over pure Ulysses** — it is generally faster at the same GPU count due to better compute-to-communication ratio and elimination of sequential branch execution. | Typically faster than pure SP for CFG models | Requires `num_gpus >= 2`. Halves the Ulysses group size (e.g. 8 GPU → two 4-GPU groups). Only for models that use CFG. |
| **Layerwise Offload** | `--dit-layerwise-offload` | Async layer-by-layer H2D prefetch with compute overlap. Only ~2 DiT layers reside on GPU at a time, dramatically reducing VRAM. For **video models** (where per-layer compute >> H2D transfer), the memcpy is completely hidden behind computation — **zero-cost offload** that saves VRAM without speed penalty ([PR #15511](https://github.com/sgl-project/sglang/pull/15511)). | Saves VRAM (40 GB → ~11 GB for Wan A14B); zero or near-zero speed cost for video models | Enabled by default for Wan/MOVA video models. Incompatible with Cache-DiT. For **image models** or highly parallelized setups (many GPUs, small per-GPU compute), the copy stream may not be fully hidden and can cause slowdown. |
| **Offload Prefetch Size** | `--dit-offload-prefetch-size F` | Fine-grained control over layerwise offload: how many layers to prefetch ahead. `0.0` = 1 layer (min VRAM), `0.1` = 10% of layers, `≥1` = absolute layer count. | Tune for cases where default offload has copy stream interference (e.g. image models). 0.05–0.1 is a good starting point. | Values ≥ 0.5 approach no-offload VRAM with worse performance. See [PR #17693](https://github.com/sgl-project/sglang/pull/17693) for benchmarks on image models. |
| **FSDP Inference** | `--use-fsdp-inference` | Uses PyTorch FSDP to shard model weights across GPUs with prefetch. Low latency, low VRAM. | Reduces per-GPU VRAM | Mutually exclusive with `--dit-layerwise-offload`. More overhead than SP on high-bandwidth interconnects. |
| **CPU Offload (components)** | `--text-encoder-cpu-offload`, `--image-encoder-cpu-offload`, `--vae-cpu-offload`, `--dit-cpu-offload` | Offloads specific pipeline components to CPU when not in use. | Reduces peak VRAM | Adds H2D transfer latency when the component is needed. Auto-enabled for low-VRAM GPUs (<30 GB). **Tip:** after the first request completes, the console prints a peak VRAM analysis with suggestions on which offload flags can be safely disabled — look for the `"Components that could stay resident"` log line. |
| **Pin CPU Memory** | `--pin-cpu-memory` | Uses pinned (page-locked) memory for CPU offload transfers. | Faster H2D transfers | Slightly higher host memory usage. Enabled by default; disable only as workaround for CUDA errors. |
| **Attention Backend (lossless)** | `--attention-backend fa` | Selects lossless attention kernel: `fa` (FlashAttention 2/3/4), `torch_sdpa`. FA is the fastest lossless option. | FA >> SDPA for long sequences | FA requires compatible GPU (Ampere+). `fa3`/`fa4` are aliased to `fa`. Ring attention only works with `fa` or `sage_attn`. |
| **Parallel Folding** | *(automatic when SP > 1)* | Reuses the SP process group as TP for the T5 text encoder, so text encoding is parallelized "for free". | Faster text encoding on multi-GPU | Automatic; no user action needed. Only applies to T5-based pipelines. |

---

## Section 2: Lossy Optimizations

These options **trade output quality** for speed or VRAM savings. Results will differ from the baseline.

| Option | CLI Flag / Env Var | What It Does | Speedup | Quality Impact / Limitations |
|---|---|---|---|---|
| **Approximate Attention** | `--attention-backend sage_attn` / `sage_attn_3` / `sliding_tile_attn` / `video_sparse_attn` / `sparse_video_gen_2_attn` / `vmoba_attn` / `sla_attn` / `sage_sla_attn` | Replaces exact attention with approximate or sparse variants. `sage_attn`: INT8/FP8 quantized Q·K; `sliding_tile_attn`: spatial-temporal tile skipping; others: model-specific sparse patterns. | ~1.5–2x on attention (varies by backend) | Quality degradation varies by backend and model. `sage_attn` is the most general; sparse backends (`sliding_tile_attn`, `video_sparse_attn`, etc.) are video-model-specific and may require config files (e.g. `--mask-strategy-file-path` for STA). Requires corresponding packages installed. |
| **Cache-DiT** | `SGLANG_CACHE_DIT_ENABLED=true` + `--cache-dit-config <path>` | Caches intermediate residuals across denoising steps and skips redundant computations via a Selective Computation Mask (SCM). | ~1.5–2x on supported models | Quality depends on SCM config. Incompatible with `--dit-layerwise-offload`. Requires correct per-model config YAML. |
| **Quantized Models (Nunchaku / SVDQuant)** | `--enable-svdquant --transformer-weights-path <path>` + optional `--quantization-precision int4\|nvfp4`, `--quantization-rank 32` | W4A4-style quantization via [Nunchaku](https://nunchaku.tech). Reduces DiT weight memory by ~4x. Precision/rank can be auto-inferred from weight filename or set explicitly. | ~1.5–2x compute speedup | Lossy quantization; quality depends on rank and precision. Requires pre-quantized weights. Ampere (SM8x) or SM12x only (no Hopper SM90). Higher rank = better quality but more memory. |
| **Pre-quantized Weights** | `--transformer-weights-path <path>` | Load any pre-quantized transformer weights (FP8, INT8, etc.) from a single `.safetensors` file, a directory, or a HuggingFace repo ID. | ~1.3–1.5x compute (dtype dependent) | Requires pre-converted weights (e.g. via `tools/convert_hf_to_fp8.py` for FP8). Quality slightly worse than BF16; varies by quantization format. |
| **Component Precision Override** | `--dit-precision fp16`, `--vae-precision fp16\|bf16` | On-the-fly dtype conversion for individual components. E.g. convert a BF16 model to FP16 at load time, or run VAE in BF16 instead of FP32. | Reduces memory; FP16 can be faster on some GPUs | May affect numerical stability. VAE is FP32 by default for accuracy; lowering it is lossy. DiT defaults to BF16. |
| **Fewer Inference Steps** | `--num-inference-steps N` (sampling param) | Reduces the number of denoising steps. Fewer steps = faster. | Linear speedup | Quality degrades with too few steps. Model-dependent optimal range. |

---

## Quick Recipes

### Maximum speed, video model, multi-GPU, lossless (Wan A14B, 8 GPUs)

```bash
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --num-gpus 8 --enable-cfg-parallel --ulysses-degree 4 \
  --enable-torch-compile --warmup \
  --text-encoder-cpu-offload true \
  --prompt "..." --save-output
```

Note: `--dit-layerwise-offload` is enabled by default for Wan/MOVA video models and is zero-cost (H2D fully overlapped with compute). No need to disable it.

### Maximum speed, image model, single GPU, lossless

```bash
sglang generate --model-path <IMAGE_MODEL> \
  --enable-torch-compile --warmup \
  --dit-layerwise-offload false \
  --prompt "..." --save-output
```

Note: for image models, per-layer compute is smaller, so layerwise offload may not fully hide H2D transfer. Disable it if VRAM allows.

### Low VRAM, decent speed (single GPU)

```bash
sglang generate --model-path <MODEL> \
  --enable-torch-compile --warmup \
  --dit-layerwise-offload --dit-offload-prefetch-size 0.1 \
  --text-encoder-cpu-offload true --vae-cpu-offload true \
  --prompt "..." --save-output
```

### Maximum speed, lossy (SageAttention + Cache-DiT)

```bash
SGLANG_CACHE_DIT_ENABLED=true sglang generate --model-path <MODEL> \
  --attention-backend sage_attn \
  --cache-dit-config <config.yaml> \
  --enable-torch-compile --warmup \
  --dit-layerwise-offload false \
  --prompt "..." --save-output
```

---

## Tips

- **Benchmarking**: always use `--warmup` and look for the line ending with `(with warmup excluded)` for accurate timing.
- **Perf dump**: use `--perf-dump-path result.json` to save structured metrics, then compare with `python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json result.json`.
- **Offload tuning**: after the first request, the runtime logs peak GPU memory and which components could stay resident. Use this to decide which `--*-cpu-offload` flags to disable.
- **Backend selection**: `--backend sglang` (default, auto-detected) enables all native optimizations (fused kernels, SP, etc.). `--backend diffusers` falls back to vanilla Diffusers pipelines but supports `--cache-dit-config` and diffusers attention backends.
