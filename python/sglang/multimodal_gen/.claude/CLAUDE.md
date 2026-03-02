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

## Testing

```bash
# Tests live in test/ subdirectory
python -m pytest python/sglang/multimodal_gen/test/

# No need to pre-download models — auto-downloaded at runtime
# Dependencies assumed already installed via `pip install -e "python[diffusion]"`
```

## Perf Measurement

Look for `Pixel data generated successfully in xxxx seconds` in console output. With warmup enabled, use the line containing `warmup excluded` for accurate timing.

## Env Vars

Defined in `envs.py` (300+ vars). Key ones:
- `SGLANG_DIFFUSION_ATTENTION_BACKEND` — attention backend override
- `SGLANG_CACHE_DIT_ENABLED` — enable Cache-DiT acceleration
- `SGLANG_CLOUD_STORAGE_TYPE` — cloud output storage (s3, etc.)
