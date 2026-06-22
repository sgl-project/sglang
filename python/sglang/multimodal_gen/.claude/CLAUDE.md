# CLAUDE.md — sglang-diffusion (multimodal_gen)

## What is this?

SGLang's diffusion/multimodal generation subsystem. Separate from the LLM runtime (`srt`). Supports 20+ image/video diffusion models (Wan, FLUX, HunyuanVideo, LTX, Qwen-Image, etc.) with distributed inference, LoRA, and multiple attention backends.

## Quick Start

```bash
# One-shot generation
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "A curious raccoon" --save-output

# Start server
sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --num-gpus 4

# Python API
from sglang import DiffGenerator
gen = DiffGenerator.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
result = gen.generate(sampling_params_kwargs={"prompt": "A curious raccoon"})
```

## Architecture

```
CLI / Python API / HTTP Server (FastAPI + OpenAI-compatible)
    ↓ ZMQ
Scheduler (request queue, batching, dispatch)
    ↓ multiprocessing pipes
GPU Worker(s) → ComposedPipeline (stages: TextEncode → Denoise → Decode)
```

### Key Directories

```
runtime/
├── entrypoints/        # CLI (generate/serve), HTTP server, Python API (DiffGenerator)
├── managers/           # scheduler.py, gpu_worker.py
├── pipelines_core/     # ComposedPipelineBase, stages/, schedule_batch.py (Req/OutputBatch)
├── pipelines/          # Model-specific pipelines (wan, flux, hunyuan, ltx, qwen_image, ...)
├── models/             # encoders/, dits/, vaes/, schedulers/
├── layers/             # attention/, lora/, quantization/
├── loader/             # Model loading, weight utils
├── server_args.py      # ServerArgs (all CLI/config params)
└── distributed/        # Multi-GPU (TP, SP: ulysses/ring)
configs/
├── pipeline_configs/    # Per-model pipeline configs
├── sample/             # SamplingParams
└── models/             # DiT, VAE, Encoder configs
native/
└── omnidreams_singleview/  # Vendored FlashDreams native FP8 extension (C++/CUDA + Python shims)
```

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `DiffGenerator` | `runtime/entrypoints/diffusion_generator.py` | Python API entry point |
| `ComposedPipelineBase` | `runtime/pipelines_core/composed_pipeline_base.py` | Pipeline orchestrator (stages) |
| `Scheduler` | `runtime/managers/scheduler.py` | ZMQ event loop, request dispatch |
| `GPUWorker` | `runtime/managers/gpu_worker.py` | GPU inference worker |
| `Req` / `OutputBatch` | `runtime/pipelines_core/schedule_batch.py` | Request/output containers |
| `ServerArgs` | `runtime/server_args.py` | All config params |
| `SamplingParams` | `configs/sample/sampling_params.py` | Generation params |
| `PipelineConfig` | `configs/pipeline_configs/base.py` | Model structure config |

### Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `build_pipeline()` | `runtime/pipelines_core/__init__.py` | Instantiate pipeline from model_path |
| `get_model_info()` | `registry.py` | Resolve pipeline + config classes |
| `launch_server()` | `runtime/launch_server.py` | Start multi-process server |

## Adding a New Model

1. Create pipeline in `runtime/pipelines/` extending `ComposedPipelineBase`
2. Define stages via `create_pipeline_stages()` (TextEncoding → Denoising → Decoding)
3. Add config in `configs/pipeline_configs/`
4. Register in `registry.py` via `register_configs()`

## Multi-GPU

```bash
# Sequence parallelism (video frames across GPUs)
sglang serve --model-path ... --num-gpus 4 --ulysses-degree 2 --ring-degree 2

# Tensor parallelism (model layers across GPUs)
sglang serve --model-path ... --num-gpus 2 --tp-size 2
```


## OmniDreams — Autoregressive Video World Model

NVIDIA OmniDreams (FlashDreams port): a 2.06B-parameter Cosmos-Predict2.5-based autoregressive
video diffusion model for autonomous driving simulation. It takes a text prompt, a first-frame
reference image, and optional per-frame HD-map rasters, then generates a driving video via
chunked autoregressive rollout.

**Checkpoint**: `single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt` — flat bf16, 570 keys, DiT-only
(3.9 GB). The Wan 2.1 VAE and Cosmos-Reason1-7B (Qwen2.5-VL) text encoder are separate downloads.

**Model layout** (all three components must be pre-placed at `model_path`):
```
model_path/
  single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt   # nvidia/omni-dreams-models (gated, ~4 GB)
  text_encoder/          # nvidia/Cosmos-Reason1-7B (public, ~16 GB)
    config.json, model-*.safetensors, tokenizer.json, ...
  wan_vae/               # Wan 2.1 diffusers-format VAE (public, ~485 MB)
    config.json, diffusion_pytorch_model.safetensors
```
The pipeline resolves each component via `_resolve_ckpt_path()`, `_resolve_vae_path()`, and
`_resolve_text_encoder_src()` respectively. The VAE must be in **diffusers format** (`*.safetensors`
+ `config.json`), not the flat `Wan2.1_VAE.pth` (different key naming). The VAE is NOT a Cosmos
model — `nvidia/Cosmos-Reason1-7B-VAE` does not exist. Source: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
(just the `vae/` subdirectory).

### Quick Start

```bash
# Serve
sglang serve --model-path /path/to/omni-dreams-checkpoint \
  --pipeline-class-name OmniDreamsPipeline \
  --enable-torch-compile --warmup \
  --text-encoder-cpu-offload --vae-cpu-offload

# Generate
sglang generate --model-path /path/to/omni-dreams-checkpoint \
  --pipeline-class-name OmniDreamsPipeline \
  --prompt "A car driving down a city street" \
  --image-path first_frame.png \
  --hdmap-path hdmap.mp4 \
  --enable-torch-compile --warmup \
  --seed 42 --save-output

# With native FP8 DiT (sm_120 / Blackwell only — Python API)
from sglang import DiffGenerator
gen = DiffGenerator.from_pretrained(
    "/path/to/omni-dreams-checkpoint",
    pipeline_class_name="OmniDreamsPipeline",
    pipeline_config_kwargs={"native_dit_acceleration": "auto"},
    server_kwargs={"text_encoder_cpu_offload": True, "vae_cpu_offload": True},
)
result = gen.generate(sampling_params_kwargs={
    "prompt": "A car driving down a city street",
    "image_path": "first_frame.png",
    "seed": 42,
})

# Force native FP8 (fails if extension unavailable):
SGLANG_OMNIDREAMS_FP8_DIT=1 sglang serve --model-path /path/to/omni-dreams-checkpoint \
  --pipeline-class-name OmniDreamsPipeline \
  --text-encoder-cpu-offload --vae-cpu-offload
```

`native_dit_acceleration` is NOT a CLI flag — use `pipeline_config_kwargs` in the Python API,
`SGLANG_OMNIDREAMS_FP8_DIT=1` env var, or a server config file. The native C++ sources under
`native/omnidreams_singleview/` are JIT-compiled via `torch.utils.cpp_extension.load()` on
first use; a prebuilt `.so` is reused across restarts.

### Architecture

```
OmniDreamsBeforeDenoisingStage
  ├─ _encode_text()          : prompt -> Cosmos-Reason1-7B -> [1,512,100352] full_concat
  │                            CRITICAL: NO attention_mask passed (model trained unmasked)
  ├─ _encode_reference_image : first frame -> Wan VAE encode -> normalized latent (frame-0 pin)
  └─ _encode_hdmap()         : HD-map video -> per-frame VAE encode -> per-chunk tokens

OmniDreamsDenoisingStage (autoregressive rollout)
  └─ Per chunk:
        shift_t(ar_idx) RoPE -> 2-step self-forcing denoise (predict_flow)
        context-noise=128 clean re-forward -> write authoritative K/V into cache
        unpatchify -> [B, C, len_t, H, W]

DecodingStage (standard single-pass VAE decode)
  └─ Concatenated AR latents -> Wan VAE causal temporal decode -> pixels
```

### Key Files

| File | Purpose |
|------|---------|
| `runtime/pipelines/omnidreams_pipeline.py` | Pipeline loader: non-Diffusers flat `.pt`, VAE/text/scheduler resolution, stage wiring |
| `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py` | BeforeDenoisingStage + DenoisingStage (AR rollout loop) |
| `runtime/models/dits/omnidreams.py` | OmniDreamsDiT (2.06B, 28 blocks), OmniDreamsBlock, OmniDreamsAttention |
| `runtime/models/dits/omnidreams_fp8.py` | OmniDreamsFP8DiT: native FP8 dispatch wrapper, `_SGLTransformerAdapter` |
| `runtime/models/dits/omnidreams_kvcache.py` | BlockKVCache: windowed KV cache with sink + split-copy roll |
| `runtime/models/dits/omnidreams_rope.py` | RotaryPositionEmbedding3D (NeoX 44:42:42), apply_rope_freqs, to_cos_sin_cache |
| `runtime/models/encoders/omnidreams_text.py` | full_concat_embeddings: 28-layer hidden-state mean-normalize -> 100352-dim |
| `configs/models/dits/omnidreams.py` | OmniDreamsDiTArchConfig (2048-d, 28 blocks, 16 heads, HDMap 16ch) |
| `configs/pipeline_configs/omnidreams.py` | OmniDreamsPipelineConfig (I2V, bf16 DiT, fp32 VAE, 2-step warp sigma) |
| `configs/sample/omnidreams.py` | OmniDreamsSamplingParams (720p, 2-step, len_t=2, window_size_t=6) |
| `native/omnidreams_singleview/python/optimized_dit.py` | Vendored OptimizedDiTExecutor: weight snapshot, `_ensure_weights_snapshot`, `_predict_flow_ext_impl` |
| `native/omnidreams_singleview/python/cosmos_weights.py` | Vendored BF16 streaming weight prep (QKV fuse + `_prepared` transpose) |
| `native/omnidreams_singleview/python/cosmos_fp8_utils.py` | Vendored FP8 quantization: `prepare_cosmos_quantized_streaming_weights` |
| `native/singleview_loader.py` | JIT build + prebuilt `.so` loader for the native CUDA extension |
| `native/omnidreams_singleview/src/omnidreams_singleview_ext.cpp` | Unified PYBIND11_MODULE entry (DiT + VAE + VAE-streaming bindings) |
| `native/omnidreams_singleview/src/dit_streaming/pyext/streaming_dit_bridge.cu` | C++ bridge: `optimized_dit_forward()`, `get_w()`, quantized weight extraction |
| `native/omnidreams_singleview/src/dit_streaming/kernels/cosmos_block.cuh` | `CosmosBlockWeights/Buffers/Params` structs, all native kernel declarations |
| `native/omnidreams_singleview/src/dit_streaming/kernels/cosmos_block.cu` | Per-block orchestrator: AdaLN split, ln_modulate, FP8/BF16 dispatch, cuDNN FMHA |

### Non-Negotiable Architecture Facts

- **Dim:** hidden 2048, 16 heads x 128 head-dim, 28 blocks, MLP ratio 4 (8192), AdaLN-LoRA rank 256
- **No bias** except crossattn_proj.0.bias. No GQA. No CFG (guidance_scale=1.0).
- **RoPE:** NeoX-style, T:H:W = 44:42:42 split, h/w extrap 3.0, t 1.0. Q/K RMSNorm(128-d) before RoPE.
  Self-attn has RoPE + KV-cache; cross-attn has neither.
- **Channels:** x_embedder 72-in = (16 latent + 1 cond_mask + 1 pad_mask) x patch(1x2x2).
  additional_patch_embedding 64-in = 16 HDMap-ch x patch. Pad-mask channel fused away at load time.
- **Final layer:** 2-chunk AdaLN (shift+scale, no gate) -> linear 2048->64.
  Cosmos channel-shuffle (kt kh kw c)->(c kt kh kw) fused into last-linear weight in post_load_weights.
- **Text:** Cosmos-Reason1-7B (Qwen2.5-VL, 7B, 28 x 3584 = 100352 full_concat).
  Padded to 512 tokens. NO attention_mask — adding one causes ~99 abs diff in padding embeddings
  and severe blur (see `omnidreams_blur_diagnosis_report.md`).
- **Schedule:** 2-step flow-match, warp s = shift*s/(1+(shift-1)*s) with shift=5, sigma_min=0
  -> sigmas [1.0, 0.8036, 0.0]. Custom OmniDreamsFlowMatchScheduler (not framework default).
- **AR rollout:** len_t=2 latent frames per chunk, KV-cache window=6 sink=0, context_noise=128.
  RoPE shift_t advances per chunk. Chunk0 = 1+(len_t-1)*4 pixel frames, chunk>=1 = len_t*4 frames.
- **HD-map:** Per-frame rasters VAE-encoded as causal clip, sliced into per-chunk tokens.
- **Cross-view:** Gated by enable_cross_view_attn=False (single-view checkpoint). Raises NotImplementedError when on.
- **TP: yes, SP: no.** ColumnParallelLinear/RowParallelLinear for TP projection layers.
  SP (ulysses/ring) rejected with clear error in AR rollout stage.
- **Native FP8 DiT (P4a):** Vendored from FlashDreams under `native/omnidreams_singleview/`.
  Requires sm_120 (Blackwell). Activated via `pipeline_config_kwargs={"native_dit_acceleration": "auto"}`
  or env `SGLANG_OMNIDREAMS_FP8_DIT=1`. All C++/CUDA + `cosmos_weights.py` + `cosmos_fp8_utils.py`
  are byte-identical to the FlashDreams source. Only `optimized_dit.py` differs (5 FlashDreams
  import stubs). The DenoisingStage wraps the vendored `OptimizedDiTExecutor` with
  `_SGLTransformerAdapter` (in `omnidreams_fp8.py`), which presents SGLang's `OmniDreamsDiT`
  as FlashDreams' `CosmosTransformer`. Weight pipeline: `state_dict()` → QKV fuse →
  per-out-channel FP8 quant (8 per-block linears → uint8 E4M3 + FP16 scale) → `_fp8_prepared`
  aliases. norm/adaln/embedder/final_layer stay BF16. C++ bridge resolves quantized weights
  via `cosmos_quantized_prepared=True` two-level alias resolution. Both eager and native paths
  receive raw patch tokens `[B, L, 64]` from the DenoisingStage and run the real `x_embedder`
  projection through the checkpoint weight `[2048, 68]`.

  **Fallback chain (three-layer graceful degradation):**
  1. `singleview_loader.py`: tries prebuilt `.so` first, then JIT-compiles. Returns `None` if both fail.
  2. `_load_native()`: if `mode="auto"` and `load_extension()` returned `None`, returns `None`;
     if `mode="required"`, raises `NativeAccelerationUnavailable`.
  3. `OmniDreamsDenoisingStage`: if `build_fp8_dit()` returns `None`, falls back to eager
     `OmniDreamsDiT.forward()`. The eager CUDA-graph runner is **disabled** when native is
     attempted (mutual exclusion) and **not re-enabled** on fallback. Runtime C++ failures
     after successful init propagate as Python exceptions and terminate the request.

  **Known tech debt:**
  - `x_embedder` fixup at `omnidreams_fp8.py:198-204` already removed (was dead code).
  - `prepare_fp8_dit_weights()` at `omnidreams_fp8.py:65` defined but never called in
    production — only useful for offline preprocessing or tests.
  - `_load_prebuilt_extension()` mutates `os.environ["LD_LIBRARY_PATH"]` permanently on
    first load. Harmless but messy.

### Performance Optimizations (Active)

| # | Optimization | Where | What |
|---|---|---|---|
| T1 | AdaLN fusion | OmniDreamsBlock | LayerNormScaleShift replaces norm+scale+shift (CuTe DSL fused kernel on CUDA) |
| T2 | RoPE kernel | omnidreams_rope.py + DiT | to_cos_sin_cache + _apply_rotary_emb dispatch (FlashInfer/Triton fallback) |
| T3 | KV-cache split-copy | BlockKVCache | Split-copy eliminates per-block .clone() during window roll |
| T4 | Text encoder cache | BeforeDenoisingStage | OrderedDict LRU (max 32) keyed on prompt, embeddings stored on CPU |
| T5 | Cross-attn KV precompute | OmniDreamsDiT | K/V projected once per prompt, reused across all AR chunks |
| T6 | torch.compile on blocks | OmniDreamsDiT | _compile_conditions = [isinstance(m, OmniDreamsBlock)] |
| T7 | Native FP8 DiT | native/omnidreams_singleview/ | CUTLASS FP8 tensor-core GEMMs + cuDNN FP8 FMHA + native fused kernels |

### Tests

```bash
# CPU component tests (run anywhere, no GPU)
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_components.py -x -v
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_scaffold.py -x -v

# Regression + HD-map (needs GPU + checkpoint)
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_regression.py -x -v
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_hdmap_validation.py -x -v

# FP8 DiT tests (needs GPU + native extension build)
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_fp8.py -x -v
```

### Diagnostics

```bash
# Per-frame tensor stats
SGLANG_OMNIDREAMS_DIAGNOSTICS=1 sglang generate ...

# Stage-level timing
SGLANG_DIFFUSION_STAGE_LOGGING=1 sglang generate ...
```

### Related Docs

- `omni.md` — Original implementation plan (Phases 0–6)
- `OmniDreamsOptimizationFindings.md` — Deep-dive on all 12 optimization targets
- `omnidreams_blur_diagnosis_report.md` — Root cause of the attention_mask blur bug
- `progress.md` — HD-map validation results + GPU memory observations
- `docs/superpowers/p4a_optimized_dit_refactor_to_sglang_native.md` — Plan to refactor vendored OptimizedDiTExecutor into SGLang-native implementation

For detailed file paths, architecture facts, and step-by-step build instructions, invoke
the `sglang-diffusion-omnidreams` skill.

## Testing

```bash
# Tests live in test/ subdirectory
python -m pytest python/sglang/multimodal_gen/test/

# No need to pre-download models — auto-downloaded at runtime
# Dependencies assumed already installed via `pip install -e "python[diffusion]"`
```

## Performance Tuning

For questions about optimal performance, fastest commands, VRAM reduction, or best flag combinations for a given model/GPU setup, **read the [sglang-diffusion-performance skill](skills/sglang-diffusion-performance/SKILL.md)**. It contains a complete table of lossless and lossy optimization flags with trade-offs, quick recipes, and tuning tips.

### Perf Measurement

Look for `Pixel data generated successfully in xxxx seconds` in console output. With warmup enabled, use the line containing `warmup excluded` for accurate timing.

## Env Vars

Defined in `envs.py` (300+ vars). Key ones:
- `SGLANG_DIFFUSION_ATTENTION_BACKEND` — attention backend override
- `SGLANG_CACHE_DIT_ENABLED` — enable Cache-DiT acceleration
- `SGLANG_CLOUD_STORAGE_TYPE` — cloud output storage (s3, etc.)
- `SGLANG_OMNIDREAMS_FP8_DIT` — force `native_dit_acceleration="required"` (sm_120 only)
