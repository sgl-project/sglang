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

## Native Backend Gate

Performance numbers are useful only when the intended backend actually ran.

- Treat any log containing `Falling back to diffusers backend`, `Using diffusers backend`, or `Loaded diffusers pipeline` as invalid for native SGLang performance tuning.
- Use `--backend diffusers` only for an explicit diffusers baseline. For native recipes, leave the default backend or pin `--backend sglang`.
- If a fallback happened, fix pipeline registration/model-path/config issues first, then rerun. Do not compare perf dumps collected from a fallback run.
- When the runtime auto-selects parallel settings because the user omitted them, keep the result as an auto-tuned baseline. For reproducible tuning, pin `--num-gpus`, `--ulysses-degree`, `--ring-degree`, and `--enable-cfg-parallel` explicitly.

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
| **Layerwise Offload** | `--dit-layerwise-offload` | Async layer-by-layer H2D prefetch with compute overlap. Only ~2 DiT layers reside on GPU at a time, dramatically reducing VRAM. For some video models the copy stream can be almost fully hidden behind compute. | Saves VRAM (40 GB → ~11 GB for Wan A14B); can be near-zero speed cost on the right workload | Enabled by default for Wan/MOVA video models. Incompatible with Cache-DiT. For **image models** or highly parallelized setups (many GPUs, small per-GPU compute), the copy stream may not be fully hidden and can cause slowdown. |
| **Offload Prefetch Size** | `--dit-offload-prefetch-size F` | Fine-grained control over layerwise offload: how many layers to prefetch ahead. `0.0` = 1 layer (min VRAM), `0.1` = 10% of layers, `≥1` = absolute layer count. | Tune for cases where default offload has copy stream interference (e.g. image models). 0.05–0.1 is a good starting point. | Values ≥ 0.5 approach no-offload VRAM with worse performance. Use lower values when copy overlap is weak; disable offload when memory allows and latency dominates. |
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
| **Cache-DiT** | Native: `SGLANG_CACHE_DIT_ENABLED=true` plus `SGLANG_CACHE_DIT_*` env vars. Diffusers backend: `--backend diffusers --cache-dit-config <yaml-or-json>` | Caches intermediate residuals across denoising steps and skips redundant computations via DBCache, TaylorSeer, and optional SCM. | ~1.5-2x on supported models | Quality depends on cache policy. Incompatible with `--dit-layerwise-offload`. Do not pass `--cache-dit-config` for native SGLang tuning unless you are intentionally using the diffusers backend flow. |
| **Quantized Models (Nunchaku / SVDQuant)** | `--enable-svdquant --transformer-weights-path <path>` + optional `--quantization-precision int4\|nvfp4`, `--quantization-rank 32` | W4A4-style quantization via [Nunchaku](https://nunchaku.tech). Reduces DiT weight memory by ~4x. Precision/rank can be auto-inferred from weight filename or set explicitly. | ~1.5–2x compute speedup | Lossy quantization; quality depends on rank and precision. Requires pre-quantized weights. Ampere (SM8x) or SM12x only (no Hopper SM90). Higher rank = better quality but more memory. |
| **Pre-quantized Transformer Override** | `--transformer-path <dir-or-repo>` / `--transformer-weights-path <path>` | Load a quantized transformer component or raw transformer weights. For converted ModelOpt FP8/NVFP4 directories, prefer `--transformer-path`; use `--transformer-weights-path` for weight-only artifacts the model loader expects. | ~1.3–1.5x compute (dtype dependent) | Requires a validated quantized transformer override, such as one produced by the ModelOpt helper tools. Quality is usually slightly worse than BF16 and depends on the format, fallback layers, and calibration scope. |
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

### Nightly-aligned model, 2 GPUs: LTX-2 two-stage

```bash
sglang generate --model-path Lightricks/LTX-2 \
  --pipeline-class-name LTX2TwoStagePipeline \
  --prompt "A cat and a dog baking a cake together in a kitchen." \
  --width 768 --height 512 \
  --num-frames 121 --num-inference-steps 50 --guidance-scale 4.0 \
  --seed 42 --num-gpus 2 --enable-cfg-parallel \
  --enable-torch-compile --warmup --save-output
```

Note: this generate recipe is aligned with the nightly comparison case `ltx2_twostage_t2v`. `LTX2TwoStagePipeline` is a native path and auto-resolves the spatial upsampler plus distilled LoRA from the same model snapshot unless you override them.

### Nightly-aligned model, 2 GPUs: LTX-2.3 TI2V two-stage

```bash
sglang generate --model-path Lightricks/LTX-2.3 \
  --pipeline-class-name LTX2TwoStagePipeline \
  --prompt "The cat starts walking slowly towards the camera." \
  --image-path "${ASSET_DIR}/cat.png" \
  --width 768 --height 512 \
  --num-frames 121 --num-inference-steps 50 --guidance-scale 4.0 \
  --seed 42 --num-gpus 2 \
  --enable-torch-compile --warmup --save-output
```

Note: this matches the nightly comparison case `ltx2.3_twostage_ti2v_2gpus`. Download `${ASSET_DIR}/cat.png` with the benchmark/profile skill before running it.

### Native baseline, 2 GPUs: LTX-2.3 one-stage

```bash
sglang generate --model-path Lightricks/LTX-2.3 \
  --prompt "A beautiful sunset over the ocean" \
  --negative-prompt "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static." \
  --width 768 --height 512 \
  --num-frames 121 --fps 24 \
  --num-inference-steps 30 --guidance-scale 3.0 \
  --seed 1234 --num-gpus 2 \
  --enable-torch-compile --warmup --save-output
```

Note: use this as the native `LTX2Pipeline` baseline for `LTX-2.3`. It keeps the validated one-stage resolution and explicit `LTX-2.3` sampling defaults, and matches the `ltx23-one-stage` benchmark preset in `sglang-diffusion-benchmark-profile`.

### Skill-only stress target, 2 GPUs: LTX-2.3 two-stage high resolution

```bash
sglang generate --model-path Lightricks/LTX-2.3 \
  --pipeline-class-name LTX2TwoStagePipeline \
  --prompt "A beautiful sunset over the ocean" \
  --negative-prompt "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static." \
  --width 1536 --height 1024 \
  --num-frames 121 --fps 24 \
  --num-inference-steps 30 --guidance-scale 3.0 \
  --seed 1234 --num-gpus 2 \
  --enable-torch-compile --warmup --save-output
```

Note: this is a high-resolution stress target for the native `LTX-2.3` two-stage path. It matches the skill-only `ltx23-two-stage` benchmark preset, not a nightly comparison case.

### Maximum speed, image model, single GPU, lossless

```bash
sglang generate --model-path <IMAGE_MODEL> \
  --enable-torch-compile --warmup \
  --dit-layerwise-offload false \
  --dit-cpu-offload false \
  --prompt "..." --save-output
```

Note: for image models, per-layer compute is smaller, so layerwise offload may not fully hide H2D transfer. Disable DiT layerwise and CPU offload if VRAM allows; otherwise a large image DiT can stay resident on CPU and make the denoise loop H2D-bound.

### Image-edit baselines: JoyAI and FireRed

```bash
sglang generate --backend=sglang \
  --model-path jdopensource/JoyAI-Image-Edit-Diffusers \
  --prompt "Make the cat wear a red hat" \
  --image-path "${ASSET_DIR}/cat.png" \
  --width 1024 --height 1024 \
  --num-inference-steps 40 --guidance-scale 4.0 \
  --num-gpus 2 --enable-cfg-parallel --ulysses-degree 1 \
  --dit-layerwise-offload false --dit-cpu-offload false \
  --enable-torch-compile --warmup --save-output
```

```bash
sglang generate --backend=sglang \
  --model-path FireRedTeam/FireRed-Image-Edit-1.1 \
  --prompt "Make the cat wear a red hat" \
  --image-path "${ASSET_DIR}/cat.png" \
  --width 1024 --height 1024 \
  --num-inference-steps 40 --guidance-scale 4.0 \
  --num-gpus 2 --enable-cfg-parallel --ulysses-degree 1 \
  --dit-layerwise-offload false --dit-cpu-offload false \
  --enable-torch-compile --warmup --save-output
```

Use `FireRedTeam/FireRed-Image-Edit-1.0` in the same command when comparing
FireRed 1.0. These are native image-edit paths; keep the reference image, prompt,
seed, and output size fixed when comparing denoise numbers. On H100, 2-GPU CFG
parallel was faster than the otherwise matching 2-GPU Ulysses command: FireRed
1.0 improved from 13419.15 ms to 10955.90 ms, and FireRed 1.1 improved from
13414.72 ms to 10934.21 ms.

### Hunyuan3D shape baseline

```bash
OUTPUT_DIR=$(python3 "$ENV_PY" print-output-dir --kind benchmarks --mkdir)
CONFIG_DIR="${OUTPUT_DIR}/generated_configs"
mkdir -p "${CONFIG_DIR}"
printf '{"paint_enable": false}\n' > "${CONFIG_DIR}/hunyuan3d-shape.json"

sglang generate --backend=sglang \
  --model-path tencent/Hunyuan3D-2 \
  --prompt "generate 3d mesh" \
  --image-path "${ASSET_DIR}/cat.png" \
  --config "${CONFIG_DIR}/hunyuan3d-shape.json" \
  --num-inference-steps 50 --guidance-scale 5.0 \
  --dit-layerwise-offload false --dit-cpu-offload false \
  --enable-torch-compile --warmup --save-output
```

For Hunyuan3D, treat `Hunyuan3DShapeDenoisingStage` as the primary latency
metric. Mesh export and paint stages are useful end-to-end checks but should not
drive DiT optimization decisions.

### Low VRAM, decent speed (single GPU)

```bash
sglang generate --model-path <MODEL> \
  --enable-torch-compile --warmup \
  --dit-layerwise-offload --dit-offload-prefetch-size 0.1 \
  --text-encoder-cpu-offload true --vae-cpu-offload true \
  --prompt "..." --save-output
```

### Maximum speed, lossy native path (SageAttention + Cache-DiT)

```bash
SGLANG_CACHE_DIT_ENABLED=true sglang generate --model-path <MODEL> \
  --attention-backend sage_attn \
  --dit-layerwise-offload false \
  --enable-torch-compile --warmup \
  --prompt "..." --save-output
```

Add native Cache-DiT knobs such as `SGLANG_CACHE_DIT_SCM_PRESET=medium`,
`SGLANG_CACHE_DIT_RDT=0.24`, or `SGLANG_CACHE_DIT_TAYLORSEER=true` only after
you have a BF16 baseline output to compare against.

For a diffusers-backend Cache-DiT YAML/JSON config baseline, make the fallback
explicit:

```bash
sglang generate --backend diffusers --model-path <MODEL> \
  --cache-dit-config <config.yaml> \
  --dit-layerwise-offload false \
  --prompt "..." --save-output
```

---

## Model-Specific Starting Points

Use these as first commands to benchmark, not as universal winners.

| Model family | First performance shape | Starting flags | Notes |
|---|---|---|---|
| FLUX.1 / FLUX.2 image | 1024x1024, 50 steps, 1 GPU | `--enable-torch-compile --warmup --dit-layerwise-offload false` | `black-forest-labs/FLUX.*` repos are gated; for FP8/NVFP4 use validated `--transformer-path` or `--transformer-weights-path` flows from the quant skill. |
| Qwen-Image / Qwen-Image-Edit | 1024x1024, 50 steps, 1 GPU | `--enable-torch-compile --warmup`; optionally native `SGLANG_CACHE_DIT_ENABLED=true` | Cache-DiT is lossy. For edit tasks, keep reference image, seed, and output size fixed. |
| Z-Image-Turbo | 1024x1024, 9 steps, guidance 4.0 | `--enable-torch-compile --warmup` | Mainline has Z-Image tanh/gate norm fusions; PR #21912 tracks FP8 plus CUDA Graph work. |
| Wan2.2 A14B T2V/I2V | 720p, 81 frames | Nightly: `--num-gpus 4 --enable-cfg-parallel --ulysses-degree 2 --text-encoder-cpu-offload --pin-cpu-memory` | For lowest latency, also benchmark pure Ulysses on the same GPUs. |
| Wan2.2 TI2V 5B | 720p, 81 frames, 1 GPU | `--enable-torch-compile --warmup` | Keep the input image and motion prompt fixed when comparing sparse attention or Cache-DiT. |
| LTX-2 / LTX-2.3 | 768x512, 121 frames, 2 GPUs | `--pipeline-class-name LTX2TwoStagePipeline --enable-torch-compile --warmup`; LTX-2 nightly also uses `--enable-cfg-parallel` | Use the benchmark/profile skill presets for exact nightly alignment. PRs #22441, #24025, and #23736 track additional LTX2 perf/parallel work. |
| HunyuanVideo | 848x480 or 720p class video | `--text-encoder-cpu-offload --pin-cpu-memory --enable-torch-compile --warmup` | Check VAE decode separately. GroupNorm+SiLU is default-eligible in mainline when wrapper guards pass; use `bench_group_norm_silu.py` when VAE residual blocks are hot. |
| JoyAI-Image-Edit | 1024-class TI2I, 40 steps, guidance 4.0 | `--backend=sglang --num-gpus 2 --enable-cfg-parallel --ulysses-degree 1 --enable-torch-compile --warmup --dit-layerwise-offload false --dit-cpu-offload false` | Newly supported image-edit path. Keep the input image, prompt, seed, and output size fixed; 2-GPU CFG parallel is the validated H100 starting point. |
| FireRed-Image-Edit 1.0 / 1.1 | 1024x1024 image edit, 40 steps, guidance 4.0 | `--backend=sglang --num-gpus 2 --enable-cfg-parallel --ulysses-degree 1 --enable-torch-compile --warmup --dit-layerwise-offload false --dit-cpu-offload false` | Uses the native `QwenImageEditPlusPipeline` path. 2-GPU CFG parallel is the validated H100 starting point; benchmark 1.0 and 1.1 separately because checkpoint differences can change denoise latency. |
| Hunyuan3D-2 shape | Shape generation, 50 steps, guidance 5.0 | `--backend=sglang --enable-torch-compile --warmup --dit-layerwise-offload false --dit-cpu-offload false` | Focus on `Hunyuan3DShapeDenoisingStage`; keep mesh export/paint timings separate from denoise. |
| MOVA / Helios | Use the benchmark/profile presets first | `--enable-torch-compile --warmup`; pin offload flags explicitly | PR #20530 tracks MOVA fused RMSNorm+RoPE; PR #24059 tracks Helios fused norm modulation. |

## Open PR Watchlist

As of 2026-05-02, these performance PRs were open. Treat them as direction and
prior art until merged:

- Fusion/kernel: #24025 LTX2 QK norm, #24059 Helios norm modulation, #24117 Z-Image packed QKV, #19488 Wan elementwise cross-block fusion, #19249 Z-Image gate/norm fusion, #20429 Qwen-Image layernorm/modulation, #20530 MOVA RMSNorm+RoPE.
- VAE/decode: #22531 LTX2 parallel VAE, #20927 batched tiled VAE decode.
- Runtime/parallel/cache: #22805 FLUX.2 packed QKV for A2A, #21742 hybrid attention schedule, #24053 USP replicated-prefix fix, #21613 TeaCache refactor, #24227 WanVideo TeaCache fix, #18764 dynamic batching, #24200 disaggregated diffusion.

## Tips

- **Benchmarking**: always use `--warmup` and look for the line ending with `(with warmup excluded)` for accurate timing.
- **Perf dump**: use `--perf-dump-path result.json` to save structured metrics, then compare with `python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json result.json`.
- **Offload tuning**: after the first request, the runtime logs peak GPU memory and which components could stay resident. Use this to decide which `--*-cpu-offload` flags to disable.
- **Backend selection**: `--backend sglang` (default, auto-detected) enables native optimizations (fused kernels, SP, native Cache-DiT env knobs, etc.). `--backend diffusers` falls back to Diffusers pipelines and is the path that accepts `--cache-dit-config` plus diffusers attention backend names.
- **Wan2.2-I2V sizing**: explicit `--width/--height` on `Wan2.2-I2V-A14B` control the target area while preserving the condition-image aspect ratio.
- **Mainline diffusion fast paths**: before proposing a new kernel or overlap scheme, check `sglang-diffusion-benchmark-profile/existing-fast-paths.md`. It covers GroupNorm+SiLU, Z-Image residual-form modulation, fused diffusion `QK norm + RoPE`, packed QKV/NVFP4 expectations, and existing multi-GPU overlap families such as Ulysses / USP and turbo-layer async all-to-all.
- **NVFP4 trace interpretation**: on FLUX.2 NVFP4 and Nunchaku-style checkpoints, packed QKV is expected. SGLang intentionally uses fused projection modules such as `to_qkv` / `to_added_qkv` instead of separate `to_q` / `to_k` / `to_v`, so a split-QKV trace usually means the quantized path did not engage rather than a brand new fusion opportunity.
- **Hotspot workflow split**: use `sglang-diffusion-benchmark-profile` to prove and classify a slowdown with perf dumps plus `torch.profiler`; hand concrete kernel work to `sglang-diffusion-ako4all-kernel` or another specialized optimization skill instead of expanding the benchmark skill.
