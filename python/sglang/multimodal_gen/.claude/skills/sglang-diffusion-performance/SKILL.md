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
- when a run downloads weights, use a task-owned cache and delete that model's
  cache after its eager/BCG/quality/profile group finishes; the benchmark
  skill's `--quality-bcg-matrix --model-cache-root --cleanup-model-cache`
  keeps one cache for the group and writes a zero-residual cleanup ledger
- hold one idle GPU set for the complete A/B matrix and verify no foreign
  process appears at run boundaries
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
| **Performance Mode** | `--performance-mode auto\|speed\|memory\|manual` (`--mode` alias) | Applies model-aware residency, FSDP/CFG, and compile defaults without overriding explicit flags. `auto` is the safe default; `speed` favors GPU residency; `memory` favors offload; `manual` leaves performance args explicit. | Fastest way to establish a sensible deployment baseline | `speed` may OOM and enables `torch.compile` only when the model deployment config allows it. Explicit offload/FSDP/parallelism/compile flags win. Use `manual` for controlled A/B benchmarks. |
| **torch.compile** | `--enable-torch-compile` | Applies `torch.compile` to the DiT forward pass. Treat it as a measured comparator, not an assumed upgrade. | Model- and shape-dependent; recent B300 coverage found eager or valid BCG faster or within 1% for every valid compile control | First request is slow and some models time out or drift numerically. Keep eager as the ground truth, use a warmup watchdog, and validate the target model. See the [H200/B300 survey](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/issues/21). |
| **Breakable CUDA Graph** | `--enable-breakable-cuda-graph` plus optional `--warmup-resolutions <WxH...>` and `--bcg-text-buckets ...` | Captures fixed-resolution DiT segments while leaving attention/collectives eager, reducing launch overhead on supported pipelines. | Large on launch-bound paths; merged SANA and LTX-2 cases show material e2e gains | Mutually exclusive with `torch.compile` and Cache-DiT; BCG takes priority. The model's default resolution is captured automatically; declare every additional production resolution. Current support is model-specific (Ideogram4, LTX-2/2.3, LongCat-Image, MiniMax-H3, Qwen-Image, SANA1.5, SANA-Video, Z-Image, GLM-Image), but an allowlisted model is not automatically a validated recipe. A valid run must log capture and no disable/failure/signature miss. `--warmup-resolutions` covers only `WxH`; video frame/conditioning mismatches can still fall back to Eager. MiniMax-H3 remains eager in the validated deployment because prompt-dependent packed host boundaries can miss the captured signature. |
| **Warmup** | `--warmup-mode request` | Runs dummy forward passes to warm up CUDA caches, JIT, and `torch.compile`. Eliminates cold-start penalty. | Removes first-request latency spike | Adds startup time. Without `--warmup-resolutions`, warmup happens on first request. |
| **Warmup Resolutions** | `--warmup-resolutions 256x256 720x720` | Pre-compiles and warms up specific resolutions at server startup (instead of lazily on first request). | Faster first request per resolution | Each resolution adds to startup time. Serving mode only; useful when you know your target resolutions in advance. |
| **Multi-GPU (SP)** | `--num-gpus N --ulysses-degree N` | Sequence parallelism across GPUs. Shards sequence tokens (not frames) to minimize padding. | Near-linear scaling with N GPUs | Requires NCCL; inter-GPU bandwidth matters. `ulysses_degree * ring_degree = sp_degree`. For Wan2.2 video, start by benchmarking pure Ulysses before assuming a mixed Ulysses/Ring layout is fastest. |
| **Cross-node SP** | `--nnodes`, `--node-rank`, `--dist-init-addr` with total `--num-gpus`; combine node-local Ulysses with cross-node Ring | Extends sequence parallel groups across multiple nodes. | Capacity and long-sequence scaling beyond one host | Prefer Ulysses within a node and Ring across nodes; all-to-all is usually the least cross-node-friendly. Use `--encoder-parallel replicate` today and verify the model's Ring admission and determinism. MiniMax-H3 is the current end-to-end validated recipe. |
| **CFG Parallel** | `--enable-cfg-parallel` | Runs conditional and unconditional CFG branches in parallel across GPUs. For CFG models on multi-GPU, benchmark this against pure Ulysses on your topology instead of assuming one always wins. | Often faster than pure SP for CFG models | Requires `num_gpus >= 2`. Halves the Ulysses group size (e.g. 8 GPU → two 4-GPU groups). Only for models that use CFG. Nightly coverage configs may intentionally use smaller Ulysses groups to keep ring behavior exercised; that does not automatically make them the lowest-latency choice. |
| **Layerwise Offload** | `--dit-layerwise-offload` | Async layer-by-layer H2D prefetch with compute overlap. Only ~2 DiT layers reside on GPU at a time, dramatically reducing VRAM. For some video models the copy stream can be almost fully hidden behind compute. | Saves VRAM (40 GB → ~11 GB for Wan A14B); can be near-zero speed cost on the right workload | Enabled by default for Wan/MOVA video models. Compatible with Cache-DiT (skipped blocks are not streamed). For **image models** or highly parallelized setups (many GPUs, small per-GPU compute), the copy stream may not be fully hidden and can cause slowdown. |
| **Offload Prefetch Size** | `--dit-offload-prefetch-size F` | Fine-grained control over layerwise offload: how many layers to prefetch ahead. `0.0` = 1 layer (min VRAM), `0.1` = 10% of layers, `≥1` = absolute layer count. | Tune for cases where default offload has copy stream interference (e.g. image models). 0.05–0.1 is a good starting point. | Values ≥ 0.5 approach no-offload VRAM with worse performance. Use lower values when copy overlap is weak; disable offload when memory allows and latency dominates. |
| **FSDP Inference** | `--use-fsdp-inference` | Uses PyTorch FSDP to shard model weights across GPUs with prefetch. Low latency, low VRAM. | Reduces per-GPU VRAM | Mutually exclusive with `--dit-layerwise-offload`. More overhead than SP on high-bandwidth interconnects. |
| **CPU Offload (components)** | `--text-encoder-cpu-offload`, `--image-encoder-cpu-offload`, `--vae-cpu-offload`, `--dit-cpu-offload` | Offloads specific pipeline components to CPU when not in use. | Reduces peak VRAM | Adds H2D transfer latency when the component is needed. Auto-enabled for low-VRAM GPUs (<30 GB). **Tip:** after the first request completes, the console prints a peak VRAM analysis with suggestions on which offload flags can be safely disabled — look for the `"Components that could stay resident"` log line. |
| **Pin CPU Memory** | `--pin-cpu-memory` | Uses pinned (page-locked) memory for CPU offload transfers. | Faster H2D transfers | Slightly higher host memory usage. Enabled by default; disable only as workaround for CUDA errors. |
| **Attention Backend (lossless)** | `--attention-backend fa` | Selects a lossless attention kernel for SGLang-native pipelines: `fa` (FlashAttention 2/3/4 alias) or `torch_sdpa`. | FA is usually faster than SDPA on long sequences | FA requires compatible GPU (Ampere+). For `--backend diffusers`, valid backend names differ; use the names documented in `docs/docs/sglang-diffusion/attention_backends.mdx`. |
| **Parallel Folding** | *(automatic when SP > 1)* | Reuses the SP process group as TP for the T5 text encoder, so text encoding is parallelized "for free". | Faster text encoding on multi-GPU | Automatic; no user action needed. Only applies to T5-based pipelines. |

### Single-GPU large-VRAM notes (measured on 1x B300, 275 GB, SM103)

A single large-VRAM card changes two common assumptions:

- **Launch-bound small models gain the most from BCG.** SANA1.5-1.6B (image)
  denoise dropped 0.70s -> 0.23s (-67%) with `--enable-breakable-cuda-graph`;
  SANA-Video (832x480, 17 frames) dropped 1.24s -> 0.96s (-22%) once
  `--warmup-num-frames 17` matched the served frame count. Both bit-identical.
  Compute-bound models (LongCat-Image, Qwen-Image, Z-Image, Cosmos3-Edge,
  FLUX.1-dev, Wan2.1-1.3B) saw no BCG or `torch.compile` gain — profiles show
  GEMM + flash-attention + already-fused norm/GELU saturating the device.
- **Component CPU offload is often a pessimization, not a free win.** Many
  presets enable `--text-encoder-cpu-offload` for memory-bound cards, but the
  whole model fits in 275 GB, so the H2D/D2H round trip is pure overhead.
  Dropping it on LingBot-Video-MoE was 8% faster end to end (bit-identical).
  On a large-VRAM single GPU, re-test each `*-cpu-offload` flag before keeping it.

---

## Section 2: Lossy Optimizations

These options **trade output quality** for speed or VRAM savings. Results will differ from the baseline.

| Option | CLI Flag / Env Var | What It Does | Speedup | Quality Impact / Limitations |
|---|---|---|---|---|
| **Request Quality Fast Paths** | `--quality {extra-high,high}` (`lossless` is default) | `extra-high` mounts only request-gated DiT/VAE fusions. `high` includes that complete set and may add model-owned approximate paths such as Cache-DiT or lower-precision decode. | Model- and shape-specific | Support is per model and may be a no-op. Keep `--quality lossless` as the A/B ground truth, then compare `extra-high` before `high` to isolate fusion wins. Report aggregate and worst-frame SSIM/PSNR for every non-bit-exact path; defaults are 0.95/28 dB for images and 0.92/24 dB for video unless checked-in model metadata overrides them. Do not confuse this with `--output-quality`, which controls file compression. |
| **Approximate Attention** | Server-wide: `--attention-backend sage_attn` / `sage_attn_3` / `sliding_tile_attn` / `video_sparse_attn` / `sparse_video_gen_2_attn` / `vmoba_attn` / `sla_attn` / `sage_sla_attn`. Per-request (dense drop-ins only): `--attention-backend-override sage_attn` sampling param / API `extra_body` — valid values `fa`, `torch_sdpa`, `sage_attn`, `sage_attn_3`; rejected (with a log) under BCG, torch.compile, sparse server backends, or a non-ring-capable target with ring parallelism. | Replaces exact attention with approximate or sparse variants. `sage_attn`: INT8/FP8 quantized Q·K; `sliding_tile_attn`: spatial-temporal tile skipping; others: model-specific sparse patterns. | ~1.5–2x on attention (varies by backend) | Quality degradation varies by backend and model. `sage_attn` is the most general; sparse backends (`sliding_tile_attn`, `video_sparse_attn`, etc.) are video-model-specific, may require config files (e.g. `--mask-strategy-file-path` for STA), and are server-level only. Requires corresponding packages installed. |
| **Cache-DiT** | Native: per-request `--enable-cache-dit true\|false` + `--cache-dit-params <json>` (sampling params; also via API `extra_body`). `SGLANG_CACHE_DIT_ENABLED` / `SGLANG_CACHE_DIT_*` env vars are the server-wide defaults for requests that leave them unset. Diffusers backend: `--backend diffusers --cache-dit-config <yaml-or-json>` | Caches intermediate residuals across denoising steps and skips redundant computations via DBCache, TaylorSeer, and optional SCM. | ~1.5-2x on supported models | Quality depends on cache policy. Compatible with `--dit-layerwise-offload`: skipped blocks are not streamed, and the first layer after a skip may sync-load. Models that touch every layer before the block loop (for example a full-stack AdaLN prepass) must keep that prepass off while caching. Do not pass `--cache-dit-config` for native SGLang tuning unless you are intentionally using the diffusers backend flow. |
| **CFG Gating** | Per-request `--cfg-gate-step 0.5` (sampling param; also via API `extra_body`). `SGLANG_DIFFUSION_CFG_GATE_STEP` is the server-wide default (1.0 = off). | After the given fraction of denoising steps, reuses the cached cond-uncond residual instead of running the unconditional branch each step. | Up to ~2x on the gated tail of CFG models (skips one of two branches) | Lossy; no-op without classifier-free guidance or with `--enable-cfg-parallel`. Lower fractions gate earlier and drift more. |
| **TeaCache** | `--enable-teacache` (uses model sampling presets) | Reuses residuals when adjacent denoising steps are sufficiently similar. | Model- and threshold-dependent | Approximate and model-specific. Mutually exclusive with Spectrum. Fix prompt/seed/shape/steps and validate temporal consistency, not only single frames. |
| **Spectrum** | `--enable-spectrum` plus optional `--spectrum-*` controls | Forecasts DiT features and skips selected denoising steps. | Defaults target an accuracy/speed tradeoff; aggressive windows can be much faster | Native `sglang generate` only for FLUX.1, Wan, HunyuanVideo, and SD3; not FLUX.2 or server requests. Mutually exclusive with TeaCache. `--debug` adds shadow validation and is not representative latency. |
| **Progressive Resolution** | `--progressive-mode dct_rewind --progressive-levels N --progressive-delta D` | Runs early denoising at lower latent resolution, then spectrally upsamples and switches to the target resolution. | Model- and schedule-dependent | Approximate and pipeline-specific. Keep the switch schedule fixed and compare detail, composition, and temporal stability. |
| **Causal KV-Cache Quantization** | `--kv-cache-quant int4\|int2` plus optional `--kv-cache-quant-*` controls | Compresses completed causal KV-cache chunks with Quant-VideoGen PRQ while keeping the mutable/current chunk and recent chunks in BF16. | Primarily a long-session memory saving | Currently limited to LingBot World realtime causal serving; requires `quant-videogen`. INT4 is the starting point; INT2 saves more memory with more error. It quantizes cache state, not checkpoint weights. |
| **Quantized Models (Nunchaku / SVDQuant)** | `--enable-svdquant --transformer-weights-path <path>` + optional `--quantization-precision int4\|nvfp4`, `--quantization-rank 32` | W4A4-style quantization via [Nunchaku](https://nunchaku.tech). Reduces DiT weight memory by ~4x. Precision/rank can be auto-inferred from weight filename or set explicitly. | ~1.5–2x compute speedup | Lossy quantization; quality depends on rank and precision. Requires pre-quantized weights. Ampere (SM8x) or SM12x only (no Hopper SM90). Higher rank = better quality but more memory. |
| **GGUF Transformer** | `--transformer-weights-path <file.gguf\|owner/repo:QUANT>` | Loads a community-quantized DiT from one `.gguf`; other components stay on the base model. **Shrinks the checkpoint, not the peak VRAM** — offload already bounds peak, so reach for this when the *download* or the host RAM offload pins is the problem (MiniMax-H3 17.5 vs 61.7 GiB), not when VRAM is. For a 24 GB card `kitchen_int8` is the faster option if you can afford the full BF16 checkpoint on disk. | None; expect a small slowdown from per-step dequantization | Lossy (4-bit families ~0.997 cosine vs BF16). CUDA only, `--tp-size 1`, no FSDP, no LoRA, no `--quantization`, no `--enable-svdquant`, and mutually exclusive with the H3 AdaLN cache/online flags — each rejected at startup. Validated on MiniMax-H3 `fl2va` Q4_K_M, 1 GPU. |
| **Pre-quantized Transformer Override** | `--transformer-path <dir-or-repo>` / `--transformer-weights-path <path>` | Load a quantized transformer component or raw transformer weights. For converted ModelOpt FP8/NVFP4 directories, prefer `--transformer-path`; use `--transformer-weights-path` for weight-only artifacts the model loader expects. | ~1.3–1.5x compute (dtype dependent) | Requires a validated quantized transformer override, such as one produced by the ModelOpt helper tools. Quality is usually slightly worse than BF16 and depends on the format, fallback layers, and calibration scope. |
| **Component Precision Override** | `--dit-precision fp16`, `--vae-precision fp16\|bf16` | On-the-fly dtype conversion for individual components. E.g. convert a BF16 model to FP16 at load time, or run VAE in BF16 instead of FP32. | Reduces memory; FP16 can be faster on some GPUs | May affect numerical stability. VAE is FP32 by default for accuracy; lowering it is lossy. DiT defaults to BF16. |
| **Fewer Inference Steps** | `--num-inference-steps N` (sampling param) | Reduces the number of denoising steps. Fewer steps = faster. | Linear speedup | Quality degrades with too few steps. Model-dependent optimal range. |

---

## Quick Recipes

### MiniMax-H3 first: lossless joint video/audio

H3 has a stricter contract than the generic recipes below. Keep its DiT eager
for consistency ground truth, use Ulysses rather than Ring, do not enable CFG
parallel, and leave the released overlapping tiled video-VAE decode in place.

Four H200 GPUs can keep the complete BF16/FP32 pipeline resident:

```bash
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --ulysses-degree 4 \
  --performance-mode speed \
  --enable-torch-compile false \
  --port 30010
```

On 4x H100 80 GB, start from the fastest measured lossless resident topology:

```bash
sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 4 \
  --tp-size 2 \
  --ulysses-degree 2 \
  --performance-mode speed \
  --enable-torch-compile false \
  --port 30010
```

On B200/B300, the verified resident sweep uses 8 GPUs with Ulysses8. H3 also
has a verified 4x B200 FSDP-capacity path, but FSDP all-gathers are a memory
policy rather than the default latency choice. Benchmark the target topology
with the H3 driver from `sglang-diffusion-benchmark-profile`.

Use the FL2VA partition for both `t2va` and `fl2va`; use
`--model-variant ref2va` for image/video/audio reference conditioning. The root
IDs are `MiniMaxAI/MiniMax-H3` on Hugging Face and `MiniMax/MiniMax-H3` on
ModelScope. Do not point `--model-path` at a partition subdirectory.

Current H3 restrictions:

- `torch.compile` is opt-in experimentation only because it changes numerical
  output; it is not a lossless baseline
- Ring attention and CFG parallel are incompatible with the packed single
  denoising branch
- SageAttention is rejected for the current packed multi-segment attention
- `--vae-config.parallel-decode-mode spatial`, `spatial_shard`, and patch VAE
  decode are rejected after mismatches; use the default tiled recipe
- Breakable CUDA Graph is opt-in and signature-specific; the validated
  1344x768 Ref2VA capture uses `--bcg-text-buckets 5504`, but it did not show a
  measured speedup
- the `quality=high|medium|low` Cache-DiT profiles and online FP8 are
  approximate; keep them outside lossless comparisons

### Maximum speed, video model, multi-GPU, lossless (Wan A14B, 8 GPUs)

```bash
sglang generate --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --num-gpus 8 --enable-cfg-parallel --ulysses-degree 4 \
  --enable-torch-compile --warmup-mode request \
  --text-encoder-cpu-offload true \
  --prompt "..." --save-output
```

Note: `--dit-layerwise-offload` is enabled by default for Wan/MOVA video models and is often a good default, but still benchmark it on your exact workload if latency matters.

For Wan2.2 specifically:
- the nightly-aligned 4-GPU benchmark may use `--enable-cfg-parallel --ulysses-degree=2` to keep CFG and ring behavior covered
- that is a **coverage** choice, not a guaranteed best-performance choice
- for pure latency tuning, benchmark pure Ulysses too, for example `--ulysses-degree=4 --ring-degree=1` on 4 GPUs
- on 8 GPUs, compare pure `--ulysses-degree=8` against `--enable-cfg-parallel --ulysses-degree=4`

### Current-source model, 2 GPUs: LTX-2 two-stage

```bash
sglang generate --model-path Lightricks/LTX-2 \
  --pipeline-class-name LTX2TwoStagePipeline \
  --prompt "A cat and a dog baking a cake together in a kitchen." \
  --width 768 --height 512 \
  --num-frames 121 \
  --seed 42 --num-gpus 2 --enable-cfg-parallel \
  --enable-torch-compile --warmup-mode request --save-output
```

Note: LTX-2 is a current-source benchmark preset rather than a nightly
comparison case. The command uses runtime-default steps and guidance.
`LTX2TwoStagePipeline` is a native path and auto-resolves the spatial
upsampler plus distilled LoRA from the same model snapshot unless you override
them.

### Nightly-aligned model, 2 GPUs: LTX-2.3 TI2V two-stage

```bash
sglang generate --model-path Lightricks/LTX-2.3 \
  --pipeline-class-name LTX2TwoStagePipeline \
  --prompt "The cat starts walking slowly towards the camera." \
  --image-path "${ASSET_DIR}/cat.png" \
  --width 768 --height 512 \
  --num-frames 121 \
  --seed 42 --num-gpus 2 --cfg-parallel-size 2 \
  --enable-torch-compile --warmup-mode request --save-output
```

Note: this matches the nightly comparison case `ltx2.3_twostage_ti2v_2gpus`. The nightly config omits explicit steps and guidance, so this command omits them too and uses runtime defaults. Download `${ASSET_DIR}/cat.png` with the benchmark/profile skill before running it.

### Native baseline, 2 GPUs: LTX-2.3 one-stage

```bash
sglang generate --model-path Lightricks/LTX-2.3 \
  --prompt "A beautiful sunset over the ocean" \
  --negative-prompt "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static." \
  --width 768 --height 512 \
  --num-frames 121 --fps 24 \
  --num-inference-steps 30 --guidance-scale 3.0 \
  --seed 1234 --num-gpus 2 \
  --enable-torch-compile --warmup-mode request --save-output
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
  --enable-torch-compile --warmup-mode request --save-output
```

Note: this is a high-resolution stress target for the native `LTX-2.3` two-stage path. It matches the skill-only `ltx23-two-stage` benchmark preset, not a nightly comparison case.

### Maximum speed, image model, single GPU, lossless

```bash
sglang generate --model-path <IMAGE_MODEL> \
  --enable-torch-compile --warmup-mode request \
  --dit-layerwise-offload false \
  --dit-cpu-offload false \
  --prompt "..." --save-output
```

Note: for image models, per-layer compute is smaller, so layerwise offload may not fully hide H2D transfer. Disable DiT layerwise and CPU offload if VRAM allows; otherwise a large image DiT can stay resident on CPU and make the denoise loop H2D-bound.

### Launch-bound fixed-resolution path: Breakable CUDA Graph

```bash
sglang serve --model-path Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers \
  --performance-mode speed \
  --enable-torch-compile false \
  --enable-breakable-cuda-graph \
  --warmup-resolutions 1024x1024 \
  --port 30010
```

Keep `torch.compile` off, declare every production resolution, and benchmark
the exact prompt-length distribution. Add `--bcg-text-buckets` only when the
default buckets create excessive padding or miss a served prompt signature.
Do not keep the timing unless the log contains `[Diffusion BCG] captured` and
contains no disable, capture-failure, or `serving signature MISSED` message.
For video, also match the captured frame and conditioning shape; `WxH` alone
does not prove replay.

For a repeated discovery sweep, use the benchmark/profile helper. This runs
lossless, extra-high, and high Eager/BCG ABBA pairs on one GPU set, then deletes the
model group cache once:

```bash
python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py \
  --model <PRESET> --quality-bcg-matrix \
  --model-cache-root /path/to/task-owned/model-caches \
  --cleanup-model-cache
```

### Compare cumulative request-quality fast paths

```bash
sglang generate --model-path <MODEL> \
  --quality lossless --prompt "..." --seed 42 \
  --perf-dump-path baseline.json --save-output

sglang generate --model-path <MODEL> \
  --quality extra-high --prompt "..." --seed 42 \
  --perf-dump-path quality-extra-high.json --save-output

sglang generate --model-path <MODEL> \
  --quality high --prompt "..." --seed 42 \
  --perf-dump-path quality-high.json --save-output
```

Keep every other flag fixed and compare the generated artifact as well as the
perf dumps. `high` must retain every fusion observed under `extra-high`. If the
model has no registered request-gated or high-only sites, either tier may be a
no-op.

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
  --enable-torch-compile --warmup-mode request --save-output
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
  --enable-torch-compile --warmup-mode request --save-output
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
  --enable-torch-compile --warmup-mode request --save-output
```

For Hunyuan3D, treat `Hunyuan3DShapeDenoisingStage` as the primary latency
metric. Mesh export and paint stages are useful end-to-end checks but should not
drive DiT optimization decisions.

### Low VRAM, decent speed (single GPU)

```bash
sglang generate --model-path <MODEL> \
  --enable-torch-compile --warmup-mode request \
  --dit-layerwise-offload --dit-offload-prefetch-size 0.1 \
  --text-encoder-cpu-offload true --vae-cpu-offload true \
  --prompt "..." --save-output
```

### Maximum speed, lossy native path (SageAttention + Cache-DiT)

```bash
SGLANG_CACHE_DIT_ENABLED=true sglang generate --model-path <MODEL> \
  --attention-backend sage_attn \
  --dit-layerwise-offload false \
  --enable-torch-compile --warmup-mode request \
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
| MiniMax-H3 | 1344x768 resolved canvas, 5 seconds / 124 frames at 24 fps, 50 joint video/audio steps | H200: `--num-gpus 4 --ulysses-degree 4 --performance-mode speed --enable-torch-compile false --enable-breakable-cuda-graph false`; H100: TP2 + Ulysses2 | Root ID plus `--model-variant fl2va` for T2VA/FL2VA or `ref2va` for Ref2VA. Ulysses only; no Ring/CFG/SageAttention. Preserve tiled video-VAE decode. BCG is not part of the validated H3 recipe: warmup and serving can have different packed host boundaries, and a replay-capable experiment must still beat eager without excessive graph memory. Profile joint denoise, video VAE, audio VAE/vocoder, encoder, and collectives separately. |
| FLUX.1 / FLUX.2 image | 1024x1024, runtime-default steps/guidance, 1 GPU | `--enable-torch-compile --warmup-mode request --dit-layerwise-offload false` | `black-forest-labs/FLUX.*` repos are gated; for FP8/NVFP4 use validated `--transformer-path` or `--transformer-weights-path` flows from the quant skill. |
| FLUX.2 Klein / Klein Base | 1024x1024, runtime-default steps/guidance, 1 GPU | `--enable-torch-compile --warmup-mode request --dit-layerwise-offload false` | Current registry has `black-forest-labs/FLUX.2-klein-4B`, `FLUX.2-klein-9B`, and base variants. Klein is step-distilled; Klein Base is not. |
| Qwen-Image / Qwen-Image-2512 | 1024x1024, 50 steps, no CFG, 2x H200 | `--num-gpus 2 --tp-size 2 --performance-mode speed --dit-layerwise-offload false --enable-torch-compile false --enable-breakable-cuda-graph --warmup-mode server --warmup-resolutions 1024x1024` | Validated on H200. BCG reduced median denoise time from 124.7 to 83.1 ms/step in the same-topology run. Capture every served resolution; an uncaptured shape runs eagerly. CUDA TP should select CustomAllReduceV2 with a 32 MiB diffusion workspace: the 1024x1024 row-parallel outputs are 24 MiB and otherwise fall back to NCCL. Capture used about 5 GB more peak memory per GPU. Fixed-seed output versus eager measured 0.984 SSIM / 39.7 dB PSNR but was not bit-exact. Establish an eager baseline and remeasure BCG on other hardware or shapes. Cache-DiT remains lossy. |
| Qwen-Image-Edit | 1024x1024, runtime-default steps/guidance, 1 GPU | Start eager, then compare `--enable-torch-compile --warmup-mode request` | Keep the reference image, seed, and output size fixed. Do not transfer the Qwen-Image-2512 BCG result without a model-backed edit test. |
| Krea-2 | 1024x1024, distilled `oss_turbo` defaults (8 steps, guidance 1.0) | `--performance-mode speed --warmup-mode request` | Native `krea/Krea-2` text-to-image path with Qwen3-VL text conditioning. The repo may require HF access; keep the 8-step distilled baseline separate from non-turbo sampling experiments. |
| Z-Image / Z-Image-Turbo | 1024x1024, runtime-default steps/guidance, 1 GPU | `--enable-torch-compile --warmup-mode request` | Keep base Z-Image separate from Turbo: base uses 50-step CFG defaults, Turbo uses 9-step zero-CFG defaults. Mainline has bf16-native Triton RMSNorm scale and tanh-residual fusions. |
| Wan2.2 A14B T2V/I2V | 1280x720, 81 frames | Nightly: `--num-gpus 4 --enable-cfg-parallel --ulysses-degree 2 --text-encoder-cpu-offload --pin-cpu-memory` | For lowest latency, also benchmark pure Ulysses on the same GPUs. |
| Wan2.2 TI2V 5B | 1280x720, 81 frames, 1 GPU | `--enable-torch-compile --warmup-mode request` | Keep the input image and motion prompt fixed when comparing sparse attention or Cache-DiT. |
| Wan2.1 / FastWan / TurboWan variants | 480p or 720p video, family defaults | Compare `--quality lossless`, `--quality extra-high`, and `--quality high`, then try `--enable-torch-compile --warmup-mode request`; add `--ulysses-degree` / CFG parallel only after measuring | `extra-high` and `high` mount the Wan FFN cublasLt/NVFP4 GELU epilogues and the Wan VAE RMSNorm+SiLU fast path when their guards pass; validate video quality against lossless. Current registry includes Wan2.1, FastWan2.1, FastWan2.2 TI2V, TurboWan2.1, TurboWan2.2 I2V, and Wan2.1-Fun InP. Use the compatibility matrix and benchmark presets before choosing topology. |
| Cosmos3 Nano / Super | T2I: 1024x1024 with `--num-frames 1`; T2V/I2V: 480p/720p video | Start with `--performance-mode auto --warmup-mode request`; use `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` only for benchmark isolation, and compare compile separately | One checkpoint serves T2I/T2V/I2V. Mode is request-driven: `num_frames == 1` means T2I, `--image-path` means I2V. On GPUs with at least 120 GiB available, auto mode keeps the Cosmos3 DiT and VAE resident for every checkpoint in the family; a 1xH200 832x480x9f, 4-step eager ABBA reduced e2e from 1.576 to 0.428 seconds with exact output parity. Cosmos3 runs one DiT per pipeline, so component offload above that threshold only buys a DiT copy out to host memory and back per request -- it cost Cosmos3-Super 720p 81f T2V ~4s of ~115s on 2xH200. |
| Cosmos3 Edge / distilled Super | Edge T2I: 640x640, 35 steps, 1 GPU; distilled Super T2I: 640x640, fixed 4-step schedule, 4 GPUs | Start eager with `--performance-mode manual`; use `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` only for benchmark isolation | Edge is trained for 256p/480p shapes. Distilled checkpoints own their sigma schedule and force guidance 1.0; do not override steps or flow shift. Do not retry the closed experimental Cosmos BCG path without a new lifecycle design. |
| Ideogram 4 FP8/NVFP4 | 1024x1024, native preset defaults | `--enable-torch-compile --warmup-mode request` | Do not set `--num-inference-steps` or `--guidance-scale` directly unless you also update the Ideogram preset; sampling params derive them from `preset`. |
| ERNIE-Image / GLM-Image / SANA / SD3 | 1024-class image, family defaults | `--enable-torch-compile --warmup-mode request`; disable offload only after checking VRAM | Treat these as current native image families. Start with benchmark/profile presets for ERNIE, GLM, and SANA; use registry/config defaults for SD3 unless you add a new preset. |
| LongCat-Image | 1024x1024, 50 steps, guidance 4.5, 1 GPU | `--performance-mode manual --enable-prompt-rewrite false` for a DiT-only eager baseline; compare `--enable-breakable-cuda-graph --warmup-resolutions 1024x1024 --enable-torch-compile false` for fixed-resolution serving | Prompt rewriting is enabled by the model defaults and runs a Qwen2.5-VL component. Disable it for kernel A/B, then keep a separate end-to-end recipe with rewriting enabled. LongCat always sends a 512-token prompt body to the DiT, so BCG reuses one signature across prompt lengths without a custom text bucket. |
| SANA-Video | 832x480, 17 frames, 8 steps for CI-sized profiling; 81 frames, 50 steps for release quality | `--performance-mode manual` and eager first; compare `--enable-breakable-cuda-graph --warmup-resolutions 832x480 --warmup-num-frames 17 --enable-torch-compile false` for fixed-resolution serving | Self QKV and cross KV are already packed. The default 300-token prompt shape reuses one BCG signature without a custom text bucket. **The BCG frame count must match the served frame count**: warmup otherwise captures the sampling default (81 frames), so a 17-frame request misses the captured graph and falls back to eager — measured slower than baseline on a single B300 (1.40s vs 1.24s). With `--warmup-num-frames 17` the same run is bit-identical and 22% faster (0.96s). Check SANA's shared bit-exact conv/modulation fast paths and one-time contiguous layout before adding a new kernel. |
| LTX-2 / LTX-2.3 | 768x512 or HQ 1920x1088, 121 frames | `--pipeline-class-name LTX2TwoStagePipeline --enable-torch-compile --warmup-mode request`; HQ uses `LTX2TwoStageHQPipeline` | Use benchmark/profile presets for nightly alignment, one-stage, high-resolution stress, and HQ. Device mode choices are `original` and `resident`; `resident` is fastest but uses more VRAM. `snapshot` is a deprecated alias for `original`, so do not use it in new commands. |
| LTX-2.5 | One-stage distilled: 960x544, 121 frames, 8 steps; two-stage: 1920x1088 | `--pipeline-class-name LTX2Pipeline --performance-mode manual`; add `--use-diffusion-decoder` only for the decoder A/B | Benchmark the DiT and optional diffusion decoder as separate stages. Confirm NATTEN `na3d` is active before comparing decoder latency; a FlexAttention fallback is a different backend. Distilled weights run unguided. |
| HunyuanVideo | 848x480 or 720p class video | `--text-encoder-cpu-offload --pin-cpu-memory --enable-torch-compile --warmup-mode request` | Check VAE decode separately. GroupNorm+SiLU is default-eligible in mainline when wrapper guards pass; use `bench_group_norm_silu.py` when VAE residual blocks are hot. |
| JoyAI-Image-Edit | 1024-class TI2I, 40 steps, guidance 4.0 | `--backend=sglang --num-gpus 2 --enable-cfg-parallel --ulysses-degree 1 --enable-torch-compile --warmup-mode request --dit-layerwise-offload false --dit-cpu-offload false` | Newly supported image-edit path. Keep the input image, prompt, seed, and output size fixed; 2-GPU CFG parallel is the validated H100 starting point. |
| FireRed-Image-Edit 1.0 / 1.1 | 1024x1024 image edit, 40 steps, guidance 4.0 | `--backend=sglang --num-gpus 2 --enable-cfg-parallel --ulysses-degree 1 --enable-torch-compile --warmup-mode request --dit-layerwise-offload false --dit-cpu-offload false` | Uses the native `QwenImageEditPlusPipeline` path. 2-GPU CFG parallel is the validated H100 starting point; benchmark 1.0 and 1.1 separately because checkpoint differences can change denoise latency. |
| Hunyuan3D-2 shape | Shape generation, 50 steps, guidance 5.0 | `--backend=sglang --enable-torch-compile --warmup-mode request --dit-layerwise-offload false --dit-cpu-offload false` | Focus on `Hunyuan3DShapeDenoisingStage`; keep mesh export/paint timings separate from denoise. |
| LingBot Video MoE 30B | 384x640, 17 frames, 12 steps for the current GPU case | `--model-path robbyant/lingbot-video-moe-30b-a3b --text-encoder-cpu-offload` | Native T2V path. Prompts are structured JSON captions, not raw free text; keep that contract when comparing latency or quality. Current main can mount the fused Triton RMSNorm path at `quality=extra-high` or `quality=high`; keep `lossless` as the reference. `--text-encoder-cpu-offload` targets memory-bound multi-GPU or small-VRAM cards; on a single large-VRAM GPU (e.g. 275 GB B300) the whole model stays resident (~73 GB peak), so dropping the flag removes H2D/D2H traffic and was 8% faster end to end (3.80s -> 3.49s, bit-identical). |
| MOVA / Helios / LingBot World | Use the benchmark/profile presets or server test cases first | `--enable-torch-compile --warmup-mode request`; pin offload and topology flags explicitly | These video/realtime families have model-specific stages and condition handling. For LingBot World causal serving, keep `--kv-cache-quant off` as the exact cache baseline before testing INT4/INT2. |

## Historical PR Watchlist

Treat these performance PRs as direction and prior art only. Re-check the PR
state and the active source tree before relying on any path, flag, or claim
about whether the work has merged:

- Fusion/kernel: #24025 LTX2 QK norm, #24059 Helios norm modulation, #24117 Z-Image packed QKV, #19488 Wan elementwise cross-block fusion, #19249 Z-Image gate/norm fusion, #20429 Qwen-Image layernorm/modulation, #20530 MOVA RMSNorm+RoPE.
- Recent eager/BCG work: #34172 LTX2 quality-high fusion, #34174 automatic
  default-resolution BCG warmup, #34210 Z-Image BCG correctness, #34305/#34314
  Ideogram eager fusions, #34584 Wan TI2V modulation/RoPE, #34616 FLUX2,
  #34617 Hunyuan, #34619 GLM, #34620 ERNIE, #34928 SANA, #34929 LTX2.3,
  #34932 Cosmos3, #35724 LongCat BCG, #35728 SANA-Video high-quality linear
  attention, and #35729 SANA-Video BCG. #35961/#35969/#35981 are open
  SANA-Video, LingBot, and Wan VAE candidates. Re-check open/merged state
  before reusing a path.
- VAE/decode: #22531 LTX2 parallel VAE, #20927 batched tiled VAE decode.
- Runtime/parallel/cache: #22805 FLUX.2 packed QKV for A2A, #21742 hybrid attention schedule, #24053 USP replicated-prefix fix, #21613 TeaCache refactor, #24227 WanVideo TeaCache fix, #18764 dynamic batching, #24200 disaggregated diffusion.

## Tips

- **Benchmarking**: establish eager first (`--performance-mode manual`, compile/BCG/cache off), always use `--warmup-mode request`, and look for the line ending with `(with warmup excluded)` for accurate timing. Add compile or BCG as separate labeled controls.
- **PR gate**: use repeated same-GPU ABBA measurements and saved-request wall
  time. Require at least 1.5% mean e2e improvement for this optimization sweep;
  attach a representative baseline/candidate profile and generated-media A/B.
- **Checkpoint cleanup**: finish every variant for one model, then delete only
  its task-owned cache and verify the cleanup ledger reports zero residual
  weight files. Never point cleanup at a shared Hugging Face or ModelScope cache.
- **Preset vs experiment control**: start with `--performance-mode auto` or
  `speed` for deployment, but use `--performance-mode manual` and pin the
  relevant residency/parallelism flags for controlled A/B claims.
- **Perf dump**: use `--perf-dump-path result.json` to save structured metrics, then compare with `python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json result.json`.
- **Offload tuning**: after the first request, the runtime logs peak GPU memory and which components could stay resident. Use this to decide which `--*-cpu-offload` flags to disable.
- **Backend selection**: `--backend sglang` (default, auto-detected) enables native optimizations (fused kernels, SP, native Cache-DiT env knobs, etc.). `--backend diffusers` falls back to Diffusers pipelines and is the path that accepts `--cache-dit-config` plus diffusers attention backend names.
- **Wan2.2-I2V sizing**: explicit `--width/--height` on `Wan2.2-I2V-A14B` control the target area while preserving the condition-image aspect ratio.
- **Mainline diffusion fast paths**: before proposing a new kernel or overlap scheme, check `sglang-diffusion-benchmark-profile/existing-fast-paths.md`. It covers H3 indexed modulation, fused QK norm + RoPE, packed Ulysses QKV/USP relayout and batched TP AdaLN; FLUX/GLM/SANA bit-exact LayerNorm+modulate; request-scoped quality gates; Wan causal-VAE data movement; GroupNorm+SiLU, Z-Image bf16-native norm modulation, LTX2 split RoPE/residual-gate add, varlen USP pack/scatter, packed QKV/NVFP4, breakable CUDA graph, and existing distributed overlap families.
- **NVFP4 trace interpretation**: on FLUX.2 NVFP4 and Nunchaku-style checkpoints, packed QKV is expected. SGLang intentionally uses fused projection modules such as `to_qkv` / `to_added_qkv` instead of separate `to_q` / `to_k` / `to_v`, so a split-QKV trace usually means the quantized path did not engage rather than a brand new fusion opportunity.
- **Hotspot workflow split**: use `sglang-diffusion-benchmark-profile` to prove and classify a slowdown with perf dumps plus `torch.profiler`; hand concrete kernel work off with the perf/profile evidence attached instead of expanding the benchmark skill.
