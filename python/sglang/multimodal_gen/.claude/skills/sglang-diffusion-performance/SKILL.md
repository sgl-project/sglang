---
name: sglang-diffusion-performance
description: Use when choosing the fastest SGLang Diffusion flags for a model, GPU, and VRAM budget.
---

# SGLang Diffusion Performance Tuning

Use this skill when the user wants the fastest command line, lower VRAM, or the right performance flags for a specific model and GPU setup.

Before running any `sglang generate` command below inside the diffusion container:
- use `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py` to derive the repo root, verify write access, and choose idle GPU(s)
- export `HF_TOKEN` first when the selected model lives in a gated Hugging Face repo such as `black-forest-labs/FLUX.*`
- export `FLASHINFER_DISABLE_VERSION_CHECK=1`
- `cd` to the repo root resolved from `sglang.__file__`

Reference: [SGLang-Diffusion Advanced Optimizations Blog](https://lmsys.org/blog/2026-02-16-sglang-diffusion-advanced-optimizations/)

---

## Section 1: Lossless Optimizations

These options are intended to preserve output quality. In practice, some paths (most notably `torch.compile`) can still introduce small floating-point drift, so validate on your target model when numerical parity matters.

| Option | CLI Flag / Env Var | What It Does | Speedup | Limitations / Notes |
|---|---|---|---|---|
| **torch.compile** | `--enable-torch-compile` | Applies `torch.compile` to the DiT forward pass, fusing ops and reducing kernel launch overhead. | ~1.2–1.5x on denoising | First request is slow (compilation). May cause minor precision drifts due to [PyTorch issue #145213](https://github.com/pytorch/pytorch/issues/145213). Pair with `--warmup` for best results. |
| **Warmup** | `--warmup` | Runs dummy forward passes to warm up CUDA caches, JIT, and `torch.compile`. Eliminates cold-start penalty. | Removes first-request latency spike | Adds startup time. Without `--warmup-resolutions`, warmup happens on first request. |
| **Warmup Resolutions** | `--warmup-resolutions 256x256 720x720` | Pre-compiles and warms up specific resolutions at server startup (instead of lazily on first request). | Faster first request per resolution | Each resolution adds to startup time. Serving mode only; useful when you know your target resolutions in advance. |
| **Multi-GPU (SP)** | `--num-gpus N --ulysses-degree N` | Sequence parallelism across GPUs. Shards sequence tokens (not frames) to minimize padding. | Near-linear scaling with N GPUs | Requires NCCL; inter-GPU bandwidth matters. `ulysses_degree * ring_degree = sp_degree`. For Wan2.2 video, start by benchmarking pure Ulysses before assuming a mixed Ulysses/Ring layout is fastest. |
| **CFG Parallel** | `--enable-cfg-parallel` | Runs conditional and unconditional CFG branches in parallel across GPUs. For CFG models on multi-GPU, benchmark this against pure Ulysses on your topology instead of assuming one always wins. | Often faster than pure SP for CFG models | Requires `num_gpus >= 2`. Halves the Ulysses group size (e.g. 8 GPU → two 4-GPU groups). Only for models that use CFG. Nightly coverage configs may intentionally use smaller Ulysses groups to keep ring behavior exercised; that does not automatically make them the lowest-latency choice. |
| **Layerwise Offload** | `--dit-layerwise-offload` | Async layer-by-layer H2D prefetch with compute overlap. Only ~2 DiT layers reside on GPU at a time, dramatically reducing VRAM. For some video models the copy stream can be almost fully hidden behind compute ([PR #15511](https://github.com/sgl-project/sglang/pull/15511)). | Saves VRAM (40 GB → ~11 GB for Wan A14B); can be near-zero speed cost on the right workload | Enabled by default for Wan/MOVA video models. Incompatible with Cache-DiT. For **image models** or highly parallelized setups (many GPUs, small per-GPU compute), the copy stream may not be fully hidden and can cause slowdown. |
| **Offload Prefetch Size** | `--dit-offload-prefetch-size F` | Fine-grained control over layerwise offload: how many layers to prefetch ahead. `0.0` = 1 layer (min VRAM), `0.1` = 10% of layers, `≥1` = absolute layer count. | Tune for cases where default offload has copy stream interference (e.g. image models). 0.05–0.1 is a good starting point. | Values ≥ 0.5 approach no-offload VRAM with worse performance. See [PR #17693](https://github.com/sgl-project/sglang/pull/17693) for benchmarks on image models. |
| **FSDP Inference** | `--use-fsdp-inference` | Uses PyTorch FSDP to shard model weights across GPUs with prefetch. Low latency, low VRAM. | Reduces per-GPU VRAM | Mutually exclusive with `--dit-layerwise-offload`. More overhead than SP on high-bandwidth interconnects. |
| **CPU Offload (components)** | `--text-encoder-cpu-offload`, `--image-encoder-cpu-offload`, `--vae-cpu-offload`, `--dit-cpu-offload` | Offloads specific pipeline components to CPU when not in use. | Reduces peak VRAM | Adds H2D transfer latency when the component is needed. Auto-enabled for low-VRAM GPUs (<30 GB). **Tip:** after the first request completes, the console prints a peak VRAM analysis with suggestions on which offload flags can be safely disabled — look for the `"Components that could stay resident"` log line. |
| **Pin CPU Memory** | `--pin-cpu-memory` | Uses pinned (page-locked) memory for CPU offload transfers. | Faster H2D transfers | Slightly higher host memory usage. Enabled by default; disable only as workaround for CUDA errors. |
| **Attention Backend (lossless)** | `--attention-backend fa` | Selects a lossless attention kernel for SGLang-native pipelines: `fa` (FlashAttention 2/3/4 alias) or `torch_sdpa`. | FA is usually faster than SDPA on long sequences | FA requires compatible GPU (Ampere+). For `--backend diffusers`, valid backend names differ; use the names documented in `docs/diffusion/performance/attention_backends.md`. |
| **Parallel Folding** | *(automatic when SP > 1)* | Reuses the SP process group as TP for the T5 text encoder, so text encoding is parallelized "for free". | Faster text encoding on multi-GPU | Automatic; no user action needed. Only applies to T5-based pipelines. |

---

## Section 2: Lossy Optimizations

These options **trade output quality** for speed or VRAM savings. Results will differ from the baseline.

| Option | CLI Flag / Env Var | What It Does | Speedup | Quality Impact / Limitations |
|---|---|---|---|---|
| **Approximate Attention** | `--attention-backend sage_attn` / `sage_attn_3` / `sliding_tile_attn` / `video_sparse_attn` / `sparse_video_gen_2_attn` / `vmoba_attn` / `sla_attn` / `sage_sla_attn` | Replaces exact attention with approximate or sparse variants. `sage_attn`: INT8/FP8 quantized Q·K; `sliding_tile_attn`: spatial-temporal tile skipping; others: model-specific sparse patterns. | ~1.5–2x on attention (varies by backend) | Quality degradation varies by backend and model. `sage_attn` is the most general; sparse backends (`sliding_tile_attn`, `video_sparse_attn`, etc.) are video-model-specific and may require config files (e.g. `--mask-strategy-file-path` for STA). Requires corresponding packages installed. |
| **Cache-DiT** | `SGLANG_CACHE_DIT_ENABLED=true` + `--cache-dit-config <path>` | Caches intermediate residuals across denoising steps and skips redundant computations via a Selective Computation Mask (SCM). | ~1.5–2x on supported models | Quality depends on SCM config. Incompatible with `--dit-layerwise-offload`. Requires correct per-model config YAML. |
| **Quantized Models (Nunchaku / SVDQuant)** | `--enable-svdquant --transformer-weights-path <path>` + optional `--quantization-precision int4\|nvfp4`, `--quantization-rank 32` | W4A4-style quantization via [Nunchaku](https://nunchaku.tech). Reduces DiT weight memory by ~4x. Precision/rank can be auto-inferred from weight filename or set explicitly. | ~1.5–2x compute speedup | Lossy quantization; quality depends on rank and precision. Requires pre-quantized weights. Ampere (SM8x) or SM12x only (no Hopper SM90). Higher rank = better quality but more memory. |
| **Pre-quantized Weights** | `--transformer-weights-path <path>` | Load any pre-quantized transformer weights (FP8, INT8, etc.) from a single `.safetensors` file, a directory, or a HuggingFace repo ID. | ~1.3–1.5x compute (dtype dependent) | Requires a validated quantized transformer override, such as one produced by `tools/build_modelopt_fp8_transformer.py` for ModelOpt FP8. Quality slightly worse than BF16; varies by quantization format. |
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

Note: `--dit-layerwise-offload` is enabled by default for Wan/MOVA video models and is often a good default, but still benchmark it on your exact workload if latency matters.

For Wan2.2 specifically:
- the nightly-aligned 4-GPU benchmark may use `--enable-cfg-parallel --ulysses-degree=2` to keep CFG and ring behavior covered
- that is a **coverage** choice, not a guaranteed best-performance choice
- for pure latency tuning, benchmark pure Ulysses too, for example `--ulysses-degree=4 --ring-degree=1` on 4 GPUs
- on 8 GPUs, compare pure `--ulysses-degree=8` against `--enable-cfg-parallel --ulysses-degree=4`

### Nightly-aligned model, single GPU: LTX-2 two-stage

```bash
sglang generate --model-path Lightricks/LTX-2 \
  --pipeline-class-name LTX2TwoStagePipeline \
  --prompt "A beautiful sunset over the ocean" \
  --negative-prompt "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static." \
  --width 1536 --height 1024 \
  --num-frames 121 --fps 24 \
  --seed 1234 --num-gpus 1 \
  --enable-torch-compile --warmup --save-output
```

Note: this generate recipe is aligned with the nightly comparison case `ltx2_twostage_t2v`. After [PR #20707](https://github.com/sgl-project/sglang/pull/20707), `LTX2TwoStagePipeline` is a native path and auto-resolves the spatial upsampler plus distilled LoRA from the same model snapshot unless you override them.

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
- **Wan2.2-I2V sizing**: after [PR #21390](https://github.com/sgl-project/sglang/pull/21390), explicit `--width/--height` on `Wan2.2-I2V-A14B` control the target area while preserving the condition-image aspect ratio.
- **Merged diffusion fast paths**: before proposing a new kernel or overlap scheme, check `sglang-diffusion-benchmark-profile/existing-fast-paths.md`. It now covers merged Z-Image residual-form modulation, fused diffusion `QK norm + RoPE`, and existing multi-GPU overlap families such as Ulysses / USP and turbo-layer async all-to-all.
- **NVFP4 trace interpretation**: on FLUX.2 NVFP4 and Nunchaku-style checkpoints, packed QKV is expected. SGLang intentionally uses fused projection modules such as `to_qkv` / `to_added_qkv` instead of separate `to_q` / `to_k` / `to_v`, so a split-QKV trace usually means the quantized path did not engage rather than a brand new fusion opportunity.
- **Hotspot workflow split**: use `sglang-diffusion-benchmark-profile` to prove and classify a slowdown with perf dumps plus `torch.profiler`; hand concrete kernel work to `sglang-diffusion-ako4all-kernel` or another specialized optimization skill instead of expanding the benchmark skill.
