---
name: sglang-diffusion-performance
description: Use when choosing the fastest SGLang Diffusion flags for a model, GPU, and VRAM budget.
---

# SGLang Diffusion Performance Tuning

Use this skill when the user wants the fastest command line, lower VRAM, or the right performance flags for a specific model and GPU setup.

## Start from the model's cookbook page

Tune from the deployment we already recommend, not from scratch: read
`docs/cookbook/diffusion/<Family>/` for the model first, take its command as
the baseline, then apply the options below. The cookbook page is what users
follow, so a recipe that silently disagrees with it on GPU count, parallelism
degrees, or offload flags is a recipe for a configuration nobody runs.

List `docs/cookbook/diffusion/` for the families that have a page.

For Cosmos3 Sim-Bimanual, warm up the served reference-image geometry and count
denoise graph replays on both a prompt change and an action change. Report
short-tail signature misses separately from hits on full chunks.

`test/` configs are pinned CI setups for correctness and regression checks, not
deployment advice. Use them when you need an exactly reproducible run, and note
which cookbook flags you diverged from.

Before running any `sglang generate` command below inside the diffusion container:
- use `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py` to derive the repo root, verify write access, and choose idle GPU(s)
- export `HF_TOKEN` first when the selected model lives in a gated Hugging Face repo such as `black-forest-labs/FLUX.*`
- leave FlashInfer's version check on; if it reports a mismatch, align
  `flashinfer-python`, `flashinfer-cubin`, and `flashinfer-jit-cache` to the
  version pinned in `python/pyproject.toml` instead of disabling the check,
  because mismatched packages can change which kernels run
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

## Section 1: Output-Preserving Options

These options are intended to preserve output quality. They are deployment choices and are separate from the request `quality` level (`exact`, `lossless`, `high`). In practice, some paths (most notably `torch.compile`) can still introduce small floating-point drift, so validate on your target model when numerical parity matters.

| Option | CLI Flag / Env Var | What It Does | Speedup | Limitations / Notes |
|---|---|---|---|---|
| **Performance Mode** | `--performance-mode auto\|speed\|memory\|manual` (`--mode` alias) | Applies model-aware residency, FSDP/CFG, and compile defaults without overriding explicit flags. `auto` is the safe default; `speed` favors GPU residency; `memory` favors offload; `manual` leaves performance args explicit. | Fastest way to establish a sensible deployment baseline | `speed` may OOM and enables `torch.compile` only when the model deployment config allows it. Explicit offload/FSDP/parallelism/compile flags win. Use `manual` for controlled A/B benchmarks. |
| **torch.compile** | `--enable-torch-compile` | Applies `torch.compile` to the DiT forward pass. Treat it as a measured comparator, not an assumed upgrade. | Model- and shape-dependent; measure it against eager | First request is slow and some models time out or drift numerically. Keep eager as the ground truth, use a warmup watchdog, and validate the target model. |
| **Breakable CUDA Graph** | `--enable-breakable-cuda-graph` plus optional `--warmup-resolutions <WxH...>` and `--bcg-text-buckets ...` | Captures fixed-resolution DiT segments while leaving attention/collectives eager, reducing launch overhead on supported pipelines. | Largest where launch overhead dominates the step (small models, short sequences); none where the GPU is already saturated | Mutually exclusive with `torch.compile` and Cache-DiT; BCG takes priority. The model's default resolution is captured automatically; declare every additional production resolution. The validated models are listed in `BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS` (`runtime/server_args/server_args.py`); a model outside the list stays enabled with a warning, and a listed model is not automatically a validated recipe for your shape. A valid run must log capture and no disable/failure/signature miss. `--warmup-resolutions` covers only `WxH`; video frame/conditioning mismatches can still fall back to Eager. |
| **Warmup** | `--warmup-mode off\|request\|server` | `server` runs a synthetic request at startup, before traffic; `request` warms on the first real request and is the benchmark aid for `sglang generate`; `off` skips warmup unless a warmup shape flag below is set. | Removes first-request latency spike | `sglang serve` defaults to `server`. `--enable-torch-compile` selects `server` when no mode is set, and monolithic BCG serving uses `server` because it captures its graphs there. With warmup on, read the line ending with `(with warmup excluded)`. |
| **Warmup Shapes** | `--warmup-resolutions 256x256 720x720`, `--warmup-num-frames N`, `--warmup-sampling-params <json>` | Warms the listed resolutions (and, for video, the frame count) instead of only the model default shape. | Faster first request per shape | Each shape adds startup time. Under BCG only warmed shapes are captured, at the warmup request's `quality` (`lossless` unless `--warmup-sampling-params` sets it); every other shape or level runs eager. `--warmup-resolutions` sets width and height only. |
| **Multi-GPU (SP)** | `--num-gpus N --ulysses-degree N` | Sequence parallelism across GPUs. Shards sequence tokens (not frames) to minimize padding. | Near-linear scaling with N GPUs | Requires NCCL; inter-GPU bandwidth matters. `ulysses_degree * ring_degree = sp_degree`. For Wan2.2 video, start by benchmarking pure Ulysses before assuming a mixed Ulysses/Ring layout is fastest. |
| **Cross-node SP** | `--nnodes`, `--node-rank`, `--dist-init-addr` with total `--num-gpus`; combine node-local Ulysses with cross-node Ring | Extends sequence parallel groups across multiple nodes. | Capacity and long-sequence scaling beyond one host | Prefer Ulysses within a node and Ring across nodes; all-to-all is usually the least cross-node-friendly. Use `--encoder-parallel replicate` today and verify the model's Ring admission and determinism. MiniMax-H3 is the current end-to-end validated recipe. |
| **CFG Parallel** | `--enable-cfg-parallel` | Runs conditional and unconditional CFG branches in parallel across GPUs. For CFG models on multi-GPU, benchmark this against pure Ulysses on your topology instead of assuming one always wins. | Often faster than pure SP for CFG models | Requires `num_gpus >= 2`. Halves the Ulysses group size (e.g. 8 GPU → two 4-GPU groups). Only for models that use CFG. Nightly coverage configs may intentionally use smaller Ulysses groups to keep ring behavior exercised; that does not automatically make them the lowest-latency choice. |
| **Layerwise Offload** | `--dit-layerwise-offload` | Async layer-by-layer H2D prefetch with compute overlap. Only ~2 DiT layers reside on GPU at a time, dramatically reducing VRAM. For some video models the copy stream can be almost fully hidden behind compute. | Saves VRAM; the speed cost can be near zero when the copy hides behind compute | Whether a model streams its DiT by default depends on `--performance-mode` and the model's deployment config. Compatible with Cache-DiT (skipped blocks are not streamed). For **image models** or highly parallelized setups (many GPUs, small per-GPU compute), the copy stream may not be fully hidden and can cause slowdown. |
| **Offload Prefetch Size** | `--dit-offload-prefetch-size F` | Fine-grained control over layerwise offload: how many layers to prefetch ahead. `0.0` = 1 layer (min VRAM), `0.1` = 10% of layers, `≥1` = absolute layer count. | Tune for cases where default offload has copy stream interference (e.g. image models). Sweep it on the target configuration. | Values ≥ 0.5 approach no-offload VRAM with worse performance. Use lower values when copy overlap is weak; disable offload when memory allows and latency dominates. |
| **FSDP Inference** | `--use-fsdp-inference` | Uses PyTorch FSDP to shard model weights across GPUs with prefetch. Low latency, low VRAM. | Reduces per-GPU VRAM | Mutually exclusive with `--dit-layerwise-offload`. More overhead than SP on high-bandwidth interconnects. |
| **CPU Offload (components)** | `--text-encoder-cpu-offload`, `--image-encoder-cpu-offload`, `--vae-cpu-offload`, `--dit-cpu-offload` | Offloads specific pipeline components to CPU when not in use. | Reduces peak VRAM | Adds H2D transfer latency when the component is needed. `auto` mode chooses placement from GPU headroom. **Tip:** after the first request completes, the console prints the peak GPU memory and a `Components that can remain on GPU` hint for the offload flags that can be turned off. |
| **Pin CPU Memory** | `--pin-cpu-memory` | Uses pinned (page-locked) memory for CPU offload transfers. | Faster H2D transfers | Slightly higher host memory usage. Enabled by default; disable only as workaround for CUDA errors. |
| **Attention Backend (exact kernels)** | `--attention-backend fa` | Selects an exact attention kernel for SGLang-native pipelines: `fa` (FlashAttention 2/3/4 alias) or `torch_sdpa`. | FA is usually faster than SDPA on long sequences | FA requires compatible GPU (Ampere+). For `--backend diffusers`, valid backend names differ; use the names documented in `docs/docs/sglang-diffusion/attention_backends.mdx`. |
| **Parallel Folding** | *(automatic when SP > 1)* | Reuses the SP process group as TP for the T5 text encoder, so text encoding is parallelized "for free". | Faster text encoding on multi-GPU | Automatic; no user action needed. Only applies to T5-based pipelines. |

`--component-residency COMPONENT=MODE` sets the same placements per component
in one flag, and the per-component flags above remain supported. See
"Component Residency" in `docs/docs/sglang-diffusion/api/cli.mdx`.

### Choosing what goes in `--layerwise-offload-components`

Layerwise offload pays one H2D of a component's weights **per pass over that component**, overlapped with that pass's compute. So the question is not how big the component is, it is **how many passes per request it makes** — counted from the code, not from the pipeline diagram:

| Component | Passes per request | Placement |
|---|---|---|
| DiT | one per denoising step | Stream it. The transfer amortizes over every step and hides behind attention/FFN. |
| Video VAE | one per **temporal chunk**, not one | Declare it streamed and keep its blocks resident. |
| Text / image encoder | one | Resident if it fits; otherwise streamed with its blocks resident. |
| Vocab table | a gather, one row per token | Neither. Declare it in `host_resident_table_names` and leave it in host memory. |

"One-shot" is a property of the code, not of the diagram. `_decode_temporal_streaming` in `runtime/models/vaes/minimax_h3_video_vae/klvae.py` calls the whole video decoder once per temporal chunk, so streaming its blocks pays one transfer of every block per chunk.

**There are three placements, not two.** A component can be declared streamed and still hold its blocks:

| Placement | How | Transfers |
|---|---|---|
| Resident | leave it out of the list | once at load; the VRAM is held for the whole process |
| Streamed, blocks resident | in the list **plus** `--layerwise-resident-layers <component>=<all its blocks>`, for example `video_vae=36` | once, not per pass, and the VRAM comes back when the component finishes |
| Streamed | in the list, resident layers 0 | every pass — worth it only for a component that makes many passes, i.e. the DiT |

So do not read "drop it from the list" as the fix for a one-shot component: that keeps it resident for the whole process, which is exactly the VRAM a small card does not have. The middle placement gets the fast decode without holding the VRAM.

### Prefetch depth has a knee

`--dit-offload-prefetch-size` is not monotonic. Deeper prefetch hides more of the copy, but its staging buffers crowd out activations. Sweep it on the target configuration: the best value depends on model, resolution and card, so a value carried over from another model means nothing. `--dit-layerwise-residency-policy strided` is the other knob on the same bytes — same VRAM, same volume, spread over the step instead of crammed into its tail.

### Host memory is part of the placement decision

Per-component placement is not independent, for two reasons:

- **One host budget.** Pinning a one-shot component's weights takes host RAM that the page cache needs to serve a streamed component's mapped weights. Pinning something that runs once can slow down the thing that runs every step.
- **Pinned is asynchronous, mapped is not.** A pinned source overlaps its transfer with compute. An unpinned or mapped source is synchronous whatever the code requests, because the driver stages it through its own buffer. Same bytes, different wall clock.

When a streamed run is inexplicably slow, check the host side before touching prefetch or residency: whether the weights are pinned or served from a mapping, and whether host memory pressure pushed them onto one. On a shared machine, host load alone moves these timings, so repeat a run before reading a small difference.

### When the transfer knobs do nothing

Before tuning residency or prefetch, measure whether the transfer is exposed at all. Bytes over bandwidth is an upper bound on what *could* be exposed, not what is; prefetch exists to hide exactly that. A small DiT can show no change across resident-layer counts because its copies already overlap with compute, while a large one needs the same flags to run at all. Sweep two or three values and keep the measured winner instead of reasoning from the checkpoint size.

Residency changes do not change the output.

### A single card that holds the whole model

- **Breakable CUDA Graph pays where the step is launch-bound.** Small models and short sequences gain; a model whose profile is already GEMM and attention gains nothing from BCG or `torch.compile`. For video, the warmup frame count has to match the served frame count, or the request misses the captured graph and runs eager.
- **Component CPU offload is overhead when everything fits.** Presets that enable `--text-encoder-cpu-offload` target memory-bound cards. Re-test each `*-cpu-offload` flag on a large-VRAM card before keeping it.

---

## Section 2: Approximate Options

These options **trade output quality** for speed or VRAM savings. Results will differ from the baseline.

| Option | CLI Flag / Env Var | What It Does | Speedup | Quality Impact / Limitations |
|---|---|---|---|---|
| **Request Quality Fast Paths** | `--quality {exact,lossless,high}` (`lossless` is default) | `lossless` mounts only request-gated DiT/VAE fusions. `high` includes that complete set and may add model-owned approximate paths such as Cache-DiT or lower-precision decode. | Model- and shape-specific | Support is per model and may be a no-op. Keep `--quality exact` as the A/B ground truth, then compare `lossless` before `high` to isolate fusion wins. Report aggregate and worst-frame SSIM/PSNR for every non-bit-exact path; defaults are 0.95/28 dB for images and 0.92/24 dB for video unless checked-in model metadata overrides them. Do not confuse this with `--output-quality`, which controls file compression. |
| **Approximate Attention** | Server-wide: `--attention-backend sage_attn` / `sage_attn_3` / `sliding_tile_attn` / `video_sparse_attn` / `sparse_video_gen_2_attn` / `vmoba_attn` / `sla_attn` / `sage_sla_attn`. Per-request (dense drop-ins only): `--attention-backend-override sage_attn` sampling param / API `extra_body` — valid values `fa`, `torch_sdpa`, `sage_attn`, `sage_attn_3`; rejected (with a log) under BCG, torch.compile, sparse server backends, or a non-ring-capable target with ring parallelism. | Replaces exact attention with approximate or sparse variants. `sage_attn`: INT8/FP8 quantized Q·K; `sliding_tile_attn`: spatial-temporal tile skipping; others: model-specific sparse patterns. | ~1.5–2x on attention (varies by backend) | Quality degradation varies by backend and model. `sage_attn` is the most general; sparse backends (`sliding_tile_attn`, `video_sparse_attn`, etc.) are video-model-specific, may require config files (e.g. `--attention-backend-config mask_strategy_file_path=...` for STA), and are server-level only. Requires corresponding packages installed. |
| **Cache-DiT** | Native: per-request `--enable-cache-dit true\|false` + `--cache-dit-params <json>` (sampling params; also via API `extra_body`). `SGLANG_CACHE_DIT_ENABLED` / `SGLANG_CACHE_DIT_*` env vars are the server-wide defaults for requests that leave them unset. Diffusers backend: `--backend diffusers --cache-dit-config <yaml-or-json>` | Caches intermediate residuals across denoising steps and skips redundant computations via DBCache, TaylorSeer, and optional SCM. | ~1.5-2x on supported models | Quality depends on cache policy. Compatible with `--dit-layerwise-offload`: skipped blocks are not streamed, and the first layer after a skip may sync-load. Models that touch every layer before the block loop (for example a full-stack AdaLN prepass) must keep that prepass off while caching. Do not pass `--cache-dit-config` for native SGLang tuning unless you are intentionally using the diffusers backend flow. |
| **CFG Gating** | Per-request `--cfg-gate-step 0.5` (sampling param; also via API `extra_body`). `SGLANG_DIFFUSION_CFG_GATE_STEP` is the server-wide default (1.0 = off). | After the given fraction of denoising steps, reuses the cached cond-uncond residual instead of running the unconditional branch each step. | Up to ~2x on the gated tail of CFG models (skips one of two branches) | Lossy; no-op without classifier-free guidance or with `--enable-cfg-parallel`. Lower fractions gate earlier and drift more. |
| **TeaCache** | `--enable-teacache` (uses model sampling presets) | Reuses residuals when adjacent denoising steps are sufficiently similar. | Model- and threshold-dependent | Approximate and model-specific. Mutually exclusive with Spectrum. Fix prompt/seed/shape/steps and validate temporal consistency, not only single frames. |
| **Spectrum** | `--enable-spectrum` plus optional `--spectrum-*` controls | Forecasts DiT features and skips selected denoising steps. | Defaults target an accuracy/speed tradeoff; aggressive windows can be much faster | Native `sglang generate` for FLUX.1, Wan, HunyuanVideo, SD3, and MiniMax-H3; not FLUX.2 or server requests. Mutually exclusive with TeaCache. `--debug` is not representative latency. |
| **Progressive Resolution** | `--progressive-mode dct_rewind --progressive-levels N --progressive-delta D` | Runs early denoising at lower latent resolution, then spectrally upsamples and switches to the target resolution. | Model- and schedule-dependent | Approximate and pipeline-specific. Keep the switch schedule fixed and compare detail, composition, and temporal stability. |
| **Causal KV-Cache Quantization** | `--kv-cache-quant int4\|int2` plus optional `--kv-cache-quant-*` controls | Compresses completed causal KV-cache chunks with Quant-VideoGen PRQ while keeping the mutable/current chunk and recent chunks in BF16. | Primarily a long-session memory saving | Currently limited to LingBot World realtime causal serving; requires `quant-videogen`. INT4 is the starting point; INT2 saves more memory with more error. It quantizes cache state, not checkpoint weights. |
| **Quantized Models (Nunchaku / SVDQuant)** | `--enable-svdquant --transformer-weights-path <path>` + optional `--quantization-precision int4\|nvfp4`, `--quantization-rank 32` | W4A4-style quantization via [Nunchaku](https://nunchaku.tech). Reduces DiT weight memory by ~4x. Precision/rank can be auto-inferred from weight filename or set explicitly. | ~1.5–2x compute speedup | Lossy quantization; quality depends on rank and precision. Requires pre-quantized weights. Ampere (SM8x) or SM12x only (no Hopper SM90). Higher rank = better quality but more memory. |
| **GGUF Transformer** | `--transformer-weights-path <file.gguf\|owner/repo:QUANT>` | Loads a community-quantized DiT from one `.gguf`; other components stay on the base model. **Shrinks the checkpoint, not the peak VRAM** — offload already bounds peak, so reach for this when the *download* or the host RAM offload pins is the problem (MiniMax-H3 17.5 vs 61.7 GiB), not when VRAM is. For a 24 GB card `convrot_int8` is the faster option if you can afford the full BF16 checkpoint on disk. | None; expect a small slowdown from per-step dequantization | Lossy (4-bit families ~0.997 cosine vs BF16). CUDA only, `--tp-size 1`, no FSDP, no LoRA, no `--quantization`, no `--enable-svdquant`, and mutually exclusive with the H3 AdaLN cache/online flags — each rejected at startup. Validated on MiniMax-H3 `fl2va` Q4_K_M, 1 GPU. |
| **Pre-quantized Transformer Override** | `--transformer-path <dir-or-repo>` / `--transformer-weights-path <path>` | Load a quantized transformer component or raw transformer weights. For converted ModelOpt FP8/NVFP4 directories, prefer `--transformer-path`; use `--transformer-weights-path` for weight-only artifacts the model loader expects. | ~1.3–1.5x compute (dtype dependent) | Requires a validated quantized transformer override, such as one produced by the ModelOpt helper tools. Quality is usually slightly worse than BF16 and depends on the format, fallback layers, and calibration scope. |
| **Component Precision Override** | `--dit-precision fp16`, `--vae-precision fp16\|bf16` | On-the-fly dtype conversion for individual components. E.g. convert a BF16 model to FP16 at load time, or run VAE in BF16 instead of FP32. | Reduces memory; FP16 can be faster on some GPUs | May affect numerical stability. VAE is FP32 by default for accuracy; lowering it is lossy. DiT defaults to BF16. |
| **Fewer Inference Steps** | `--num-inference-steps N` (sampling param) | Reduces the number of denoising steps. Fewer steps = faster. | Linear speedup | Quality degrades with too few steps. Model-dependent optimal range. |

---

## Quick Recipes

Per-model commands live in the cookbook, which is what users follow. Start from
the model's page and apply the options above to it:

| Model | Cookbook page |
|---|---|
| MiniMax-H3 | `docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx` |
| Wan2.1 / Wan2.2 | `docs/cookbook/diffusion/Wan/Wan2.1.mdx`, `docs/cookbook/diffusion/Wan/Wan2.2.mdx` |
| LTX-2 / LTX-2.3 / LTX-2.5 | `docs/cookbook/diffusion/LTX/LTX2 & LTX2.3.mdx`, `docs/cookbook/diffusion/LTX/LTX2.5.mdx` |
| Other families | `docs/cookbook/diffusion/<Family>/` |

"Per-model tuning starting points" in
`docs/docs/sglang-diffusion/performance-optimization.mdx` maps model families
to the lever worth trying first.

The recipes below are patterns for a class of model, plus baselines for models
that have no cookbook page yet.

### Maximum speed, image model, single GPU

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
exact, lossless, and high Eager/BCG ABBA pairs on one GPU set, then deletes the
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
  --quality exact --prompt "..." --seed 42 \
  --perf-dump-path baseline.json --save-output

sglang generate --model-path <MODEL> \
  --quality lossless --prompt "..." --seed 42 \
  --perf-dump-path quality-lossless.json --save-output

sglang generate --model-path <MODEL> \
  --quality high --prompt "..." --seed 42 \
  --perf-dump-path quality-high.json --save-output
```

Keep every other flag fixed and compare the generated artifact as well as the
perf dumps. `high` must retain every fusion observed under `lossless`. If the
model has no registered request-gated or high-only sites, either tier may be a
no-op.

### Image-edit baselines: JoyAI and FireRed

`${ASSET_DIR}`, `$ENV_PY`, and the `cat.png` download come from "Prerequisites"
in the benchmark/profile skill's `benchmark-and-profile.md`.

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
seed, and output size fixed when comparing denoise numbers. Compare 2-GPU CFG
parallel against the otherwise matching 2-GPU Ulysses command on your hardware.

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

### Maximum speed, approximate native path (SageAttention + Cache-DiT)

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
- **Offload tuning**: after the first request, the runtime logs peak GPU memory and a `Components that can remain on GPU` hint. Use it to decide which `--*-cpu-offload` flags to disable.
- **Backend selection**: `--backend sglang` (default, auto-detected) enables native optimizations (fused kernels, SP, native Cache-DiT env knobs, etc.). `--backend diffusers` falls back to Diffusers pipelines and is the path that accepts `--cache-dit-config` plus diffusers attention backend names.
- **Wan2.2-I2V sizing**: explicit `--width/--height` on `Wan2.2-I2V-A14B` control the target area while preserving the condition-image aspect ratio.
- **Mainline diffusion fast paths**: before proposing a new kernel or overlap scheme, check `sglang-diffusion-benchmark-profile/existing-fast-paths.md`. It covers H3 indexed modulation, fused QK norm + RoPE, packed Ulysses QKV/USP relayout and batched TP AdaLN; FLUX/GLM/SANA bit-exact LayerNorm+modulate; request-scoped quality gates; Wan causal-VAE data movement; GroupNorm+SiLU, Z-Image bf16-native norm modulation, LTX2 split RoPE/residual-gate add, varlen USP pack/scatter, packed QKV/NVFP4, breakable CUDA graph, and existing distributed overlap families.
- **NVFP4 trace interpretation**: on FLUX.2 NVFP4 and Nunchaku-style checkpoints, packed QKV is expected. SGLang intentionally uses fused projection modules such as `to_qkv` / `to_added_qkv` instead of separate `to_q` / `to_k` / `to_v`, so a split-QKV trace usually means the quantized path did not engage rather than a brand new fusion opportunity.
- **Hotspot workflow split**: use `sglang-diffusion-benchmark-profile` to prove and classify a slowdown with perf dumps plus `torch.profiler`; hand concrete kernel work off with the perf/profile evidence attached instead of expanding the benchmark skill.
