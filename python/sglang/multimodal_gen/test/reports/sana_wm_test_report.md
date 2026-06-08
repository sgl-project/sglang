# SANA-WM Test Report

Date: 2026-06-09

Status: draft, GPU execution pending.

## Scope

This report covers SANA-WM validation for:

- Unit and syntax tests.
- Single-GPU correctness and performance comparisons with optimizations off/on.
- Multi-GPU correctness and performance comparisons for supported parallel modes.

The production-quality reference command uses:

- model: `Efficient-Large-Model/SANA-WM_bidirectional`
- pipeline: `SanaWMTwoStagePipeline`
- input image: `/workspace/Sana/asset/sana_wm/demo_0.png`
- prompt file: `/workspace/Sana/asset/sana_wm/demo_0.txt`
- output: `704x1280`, `321` frames, `16` fps, `60` denoising steps
- action: `w-80,jw-40,w-40,lw-60,w-100`

## Current Result Summary

| Area | Status | Notes |
| --- | --- | --- |
| Python syntax check | PASS | `compileall` passed for SANA-WM config, model, stages, pipeline, refiner, and unit test file. |
| Unit test import | BLOCKED | Local Python environment is missing `numpy`; test import fails before SANA-WM tests run. |
| Single-GPU baseline | PENDING | Requires GPU machine and model weights. |
| Single-GPU optimization matrix | PENDING | Requires GPU machine and model weights. |
| Single-GPU combo optimization matrix | PENDING | Cache-DiT + regional `torch.compile` should be tested as experimental coverage. |
| Multi-GPU TP/CFG matrix | PENDING | Requires multi-GPU machine and model weights. |

## Unit Tests

### Commands

Run from repository root:

```bash
python3 -m compileall -q \
  python/sglang/multimodal_gen/configs/pipeline_configs/sana_wm.py \
  python/sglang/multimodal_gen/configs/models/dits/sana_wm.py \
  python/sglang/multimodal_gen/runtime/models/dits/sana_wm.py \
  python/sglang/multimodal_gen/runtime/models/dits/sana_wm_refiner_transformer.py \
  python/sglang/multimodal_gen/runtime/pipelines/sana_wm_pipeline.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/sana_wm/stages.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/sana_wm/refiner.py \
  python/sglang/multimodal_gen/test/unit/test_sana_wm_pipeline_config.py
```

```bash
PYTHONPATH=python \
python3 -m unittest sglang.multimodal_gen.test.unit.test_sana_wm_pipeline_config
```

Optional full unit suite entrypoint:

```bash
PYTHONPATH=python \
python3 python/sglang/multimodal_gen/test/run_suite.py --suite unit -k sana_wm
```

### Result

| Command | Result | Detail |
| --- | --- | --- |
| `python3 -m compileall -q ...` | PASS | No syntax errors. |
| `PYTHONPATH=python python3 -m unittest ...test_sana_wm_pipeline_config` | BLOCKED | Fails at import: `ModuleNotFoundError: No module named 'numpy'`. |

Action item:

- Install the normal SGLang test/runtime dependencies in the test environment, then rerun the unittest command.

## Shared GPU Test Setup

### Environment

Fill before running GPU tests:

| Field | Value |
| --- | --- |
| Commit SHA | TBD |
| GPU model | TBD |
| GPU count | TBD |
| Driver version | TBD |
| CUDA version | TBD |
| PyTorch version | TBD |
| SGLang install mode | TBD |
| Model path | `Efficient-Large-Model/SANA-WM_bidirectional` |
| Reference asset path | `/workspace/Sana/asset/sana_wm` |
| Output root | `/workspace/outputs/sana_wm` |

### Common Base Command

Use diagnostics for the first correctness run. Disable diagnostics for performance comparisons unless investigating tensor values.

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_SANA_WM_DIAGNOSTICS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name action_321f_with_refiner.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_baseline_1gpu.json
```

### Pipeline Config Files

Use `--pipeline-config-path` for SANA-WM-specific pipeline settings.

Default conservative config:

```json
{
  "sana_wm_skip_refiner": false,
  "sana_wm_torch_compile_scope": "off"
}
```

Stage-1 only config:

```json
{
  "sana_wm_skip_refiner": true,
  "sana_wm_torch_compile_scope": "off"
}
```

Experimental regional compile config:

```json
{
  "sana_wm_skip_refiner": false,
  "sana_wm_torch_compile_scope": "regional",
  "sana_wm_torch_compile_mode": null,
  "sana_wm_torch_compile_cache_size_limit": 128
}
```

Stage-1 only regional compile config:

```json
{
  "sana_wm_skip_refiner": true,
  "sana_wm_torch_compile_scope": "regional",
  "sana_wm_torch_compile_mode": null,
  "sana_wm_torch_compile_cache_size_limit": 128
}
```

## Correctness Criteria

Each GPU test should satisfy:

- Command exits with status 0.
- Output file exists and is playable.
- Output has expected resolution and frame count.
- No NaN/Inf diagnostics in SANA-WM tensors.
- No stage/runtime exception during refiner or VAE decode.
- Visual motion roughly follows the action DSL:
  - `w`: forward motion
  - `jw`: yaw-left plus forward motion
  - `lw`: yaw-right plus forward motion
- For performance tests, diagnostics should be disabled and the same prompt, seed, shape, step count, action, precision, and model path must be reused.

Suggested media validation:

```bash
ffprobe -hide_banner -show_streams \
  /workspace/outputs/sana_wm/action_321f_with_refiner.mp4
```

## Single-GPU Tests

Use `CUDA_VISIBLE_DEVICES=0`.

For performance runs, prefer:

```bash
SGLANG_SANA_WM_DIAGNOSTICS=0
```

### Single-GPU Matrix

| Case ID | Purpose | Extra env / args | Expected |
| --- | --- | --- | --- |
| `1gpu_baseline_two_stage` | Full two-stage baseline, optimizations off | `--pipeline-config-path sana_wm_default.json` | Correct output, stable memory. |
| `1gpu_stage1_only` | Measure stage-1 without refiner cost | `--pipeline-config-path sana_wm_skip_refiner.json` | Faster than two-stage, lower memory, lower final quality. |
| `1gpu_cache_dit` | Cache-DiT acceleration experiment | `SGLANG_CACHE_DIT_ENABLED=true` | May improve denoise latency; validate quality drift. |
| `1gpu_torch_compile_regional` | Experimental regional compile | `--enable-torch-compile true --pipeline-config-path sana_wm_compile_regional.json` | Must not crash; inspect graph breaks/recompiles. |
| `1gpu_cache_dit_torch_compile_regional` | Experimental Cache-DiT + regional compile | `SGLANG_CACHE_DIT_ENABLED=true --enable-torch-compile true --pipeline-config-path sana_wm_compile_regional.json` | Must not crash; inspect quality drift, graph breaks, and recompiles. |
| `1gpu_layerwise_offload` | Memory reduction path | `--dit-layerwise-offload true --dit-offload-prefetch-size 0.25` | Lower peak memory, may be slower. |
| `1gpu_fa_backend` | Explicit FlashAttention backend | `--component-attention-backends transformer=fa` | Correct output; compare latency with default backend. |

### Single-GPU Commands

Baseline, optimizations off:

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_SANA_WM_DIAGNOSTICS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --pipeline-config-path /workspace/configs/sana_wm_default.json \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name 1gpu_baseline_two_stage.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_1gpu_baseline_two_stage.json
```

Stage-1 only, refiner skipped:

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_SANA_WM_DIAGNOSTICS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --pipeline-config-path /workspace/configs/sana_wm_skip_refiner.json \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name 1gpu_stage1_only.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_1gpu_stage1_only.json
```

Cache-DiT experiment:

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_SANA_WM_DIAGNOSTICS=0 \
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_WARMUP=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --pipeline-config-path /workspace/configs/sana_wm_default.json \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name 1gpu_cache_dit.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_1gpu_cache_dit.json
```

Regional `torch.compile` experiment:

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_SANA_WM_DIAGNOSTICS=0 \
TORCH_LOGS="+recompiles,graph_breaks" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --pipeline-config-path /workspace/configs/sana_wm_compile_regional.json \
  --enable-torch-compile true \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name 1gpu_torch_compile_regional.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_1gpu_torch_compile_regional.json
```

Cache-DiT + regional `torch.compile` experiment:

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_SANA_WM_DIAGNOSTICS=0 \
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_WARMUP=4 \
TORCH_LOGS="+recompiles,graph_breaks" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --pipeline-config-path /workspace/configs/sana_wm_compile_regional.json \
  --enable-torch-compile true \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name 1gpu_cache_dit_torch_compile_regional.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_1gpu_cache_dit_torch_compile_regional.json
```

Layerwise offload memory test:

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_SANA_WM_DIAGNOSTICS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --pipeline-config-path /workspace/configs/sana_wm_default.json \
  --dit-layerwise-offload true \
  --dit-offload-prefetch-size 0.25 \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name 1gpu_layerwise_offload.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_1gpu_layerwise_offload.json
```

Do not combine `--dit-layerwise-offload` with `SGLANG_CACHE_DIT_ENABLED=true`; server validation rejects this combination.

`Cache-DiT + torch.compile` is allowed as an experiment. The denoising stage enables Cache-DiT before calling `torch.compile`, and SANA-WM compile is explicitly opt-in through `sana_wm_torch_compile_scope="regional"`. Treat this as higher risk than either optimization alone: inspect `TORCH_LOGS`, first-token compile overhead, quality drift, and whether repeated same-shape runs avoid recompilation.

### Single-GPU Results

| Case ID | Pass/Fail | Total latency ms | Warmup-excluded latency ms | Avg denoise step ms | Peak allocated MB | Peak reserved MB | Output file | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `1gpu_baseline_two_stage` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `1gpu_stage1_only` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `1gpu_cache_dit` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `1gpu_torch_compile_regional` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Recompile/graph-break count: TBD |
| `1gpu_cache_dit_torch_compile_regional` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Quality drift and recompile/graph-break count: TBD |
| `1gpu_layerwise_offload` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `1gpu_fa_backend` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Additional Coverage To Add

These tests are not replacements for the production-shape benchmark above; they are smaller targeted checks that reduce review risk.

### Unit / Integration Coverage

| Test area | Purpose | Suggested coverage |
| --- | --- | --- |
| Action DSL validation | Reject malformed camera action input early. | Invalid keys, bad counts, empty segments, mixed case normalization. |
| Camera condition selection | Ensure precedence and shape validation are stable. | `action`, `camera_to_world`, `camera_conditions`, and missing camera input. |
| Request runtime cache cleanup | Prevent stale per-request tensors across requests/sessions. | Same session repeated request, exception path, no-session `batch.extra` path. |
| Refiner controls | Verify two-stage control plane. | `sana_wm_skip_refiner`, request `skip_refiner`, `refiner_seed`, `sink_size`, prompt batch mismatch. |
| Torch compile config | Keep conservative default stable. | Default `off`, alias normalization, explicit `regional`, explicit `full` warning path. |
| TP validation | Fail early on unsupported topology. | Invalid `tp_size`, `sp_degree != 1`, TP + CFG-refiner constraints. |

### GPU Smoke Coverage

| Case ID | Shape | Purpose |
| --- | --- | --- |
| `gpu_smoke_17f_two_stage` | `384x640`, `17` frames, `12` steps | Fast end-to-end sanity, mirrors existing CI case. |
| `gpu_smoke_17f_stage1_only` | `384x640`, `17` frames, `12` steps | Isolate stage-1 without refiner. |
| `gpu_smoke_invalid_action` | small | Confirm invalid action returns a clear error. |
| `gpu_smoke_no_action` | small | Confirm default/no-camera path works or fails intentionally with a clear message. |
| `gpu_smoke_repeated_same_shape` | small, run twice | Check request runtime cache cleanup and `torch.compile` recompile behavior when enabled. |

### Optimization Combination Coverage

Run combinations only after the baseline and each individual optimization pass.

| Combination | Should test? | Notes |
| --- | --- | --- |
| Cache-DiT + regional `torch.compile` | Yes, experimental | Allowed by current code. Verify quality drift and recompile behavior. |
| Cache-DiT + layerwise offload | No | Server validation rejects this combination. |
| Regional `torch.compile` + layerwise offload | Optional | Higher risk; test only if memory pressure requires offload and regional compile is already stable. |
| TP + Cache-DiT | Optional after 1-GPU Cache-DiT passes | Useful for stage-1 TP performance; keep SP disabled. |
| TP + regional `torch.compile` | Optional after 1-GPU compile passes | Watch compile time and rank-local graph breaks. |

## Multi-GPU Tests

SANA-WM currently supports tensor parallelism for the stage-1 DiT. Sequence parallelism is intentionally not covered because SANA-WM has frame-wise recurrent GDN, temporal convolution, camera UCPE, and Plucker conditioning that are not SP-safe yet.

### Multi-GPU Matrix

| Case ID | GPUs | Purpose | Extra args | Expected |
| --- | ---: | --- | --- | --- |
| `2gpu_tp2_two_stage` | 2 | Stage-1 TP=2, full refiner | `--tp-size 2 --enable-cfg-parallel false` | Correct output, lower per-GPU memory. |
| `4gpu_tp4_two_stage` | 4 | Stage-1 TP=4, full refiner | `--tp-size 4 --enable-cfg-parallel false` | Correct output, lower per-GPU memory than TP=2. |
| `2gpu_cfg_parallel` | 2 | CFG branch parallelism, TP=1 | `--cfg-parallel-size 2` | Correct output; compare denoise speed. |
| `2gpu_tp2_stage1_only` | 2 | TP=2 without refiner | `--tp-size 2 --enable-cfg-parallel false --pipeline-config-path sana_wm_skip_refiner.json` | Isolate stage-1 TP performance. |
| `2gpu_tp2_cache_dit_stage1_only` | 2 | TP=2 with Cache-DiT, refiner skipped | `SGLANG_CACHE_DIT_ENABLED=true --tp-size 2 --enable-cfg-parallel false --pipeline-config-path sana_wm_skip_refiner.json` | Optional after 1-GPU Cache-DiT passes. |

### Multi-GPU Commands

TP=2 full two-stage:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_SANA_WM_DIAGNOSTICS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --pipeline-config-path /workspace/configs/sana_wm_default.json \
  --tp-size 2 \
  --enable-cfg-parallel false \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name 2gpu_tp2_two_stage.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_2gpu_tp2_two_stage.json
```

TP=4 full two-stage:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_SANA_WM_DIAGNOSTICS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --pipeline-config-path /workspace/configs/sana_wm_default.json \
  --tp-size 4 \
  --enable-cfg-parallel false \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name 4gpu_tp4_two_stage.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_4gpu_tp4_two_stage.json
```

CFG parallel, TP=1:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_SANA_WM_DIAGNOSTICS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --pipeline-config-path /workspace/configs/sana_wm_default.json \
  --cfg-parallel-size 2 \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name 2gpu_cfg_parallel.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_2gpu_cfg_parallel.json
```

TP=2 stage-1 only:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_SANA_WM_DIAGNOSTICS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Efficient-Large-Model/SANA-WM_bidirectional \
  --pipeline-class-name SanaWMTwoStagePipeline \
  --pipeline-config-path /workspace/configs/sana_wm_skip_refiner.json \
  --tp-size 2 \
  --enable-cfg-parallel false \
  --image-path /workspace/Sana/asset/sana_wm/demo_0.png \
  --prompt-path /workspace/Sana/asset/sana_wm/demo_0.txt \
  --height 704 \
  --width 1280 \
  --num-frames 321 \
  --fps 16 \
  --num-inference-steps 60 \
  --guidance-scale 5.0 \
  --flow-shift 9.8 \
  --seed 42 \
  --negative-prompt '' \
  --action 'w-80,jw-40,w-40,lw-60,w-100' \
  --save-output \
  --output-path /workspace/outputs/sana_wm \
  --output-file-name 2gpu_tp2_stage1_only.mp4 \
  --perf-dump-path /workspace/outputs/sana_wm/perf_2gpu_tp2_stage1_only.json
```

### Multi-GPU Results

| Case ID | Pass/Fail | Total latency ms | Warmup-excluded latency ms | Avg denoise step ms | Peak allocated MB per GPU | Peak reserved MB per GPU | Output file | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `2gpu_tp2_two_stage` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `4gpu_tp4_two_stage` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `2gpu_cfg_parallel` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `2gpu_tp2_stage1_only` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `2gpu_tp2_cache_dit_stage1_only` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Optional. |

## Existing CI GPU Cases

Current GPU case coverage includes:

| Case ID | Topology | Notes |
| --- | --- | --- |
| `sana_wm_ti2v` | 1 GPU | Reduced CI size: `384x640`, `17` frames, `12` steps. Perf and consistency checks disabled. |
| `sana_wm_ti2v_tp2` | 2 GPU TP | Reduced CI size: `384x640`, `17` frames, `12` steps. Perf and consistency checks disabled. |

Useful suite entrypoints:

```bash
PYTHONPATH=python \
python3 python/sglang/multimodal_gen/test/run_suite.py \
  --suite 1-gpu \
  --total-partitions 1 \
  --partition-id 0 \
  -k sana_wm_ti2v
```

```bash
PYTHONPATH=python \
python3 python/sglang/multimodal_gen/test/run_suite.py \
  --suite 2-gpu \
  --total-partitions 1 \
  --partition-id 0 \
  -k sana_wm_ti2v_tp2
```

## Risk Notes

- `torch.compile` is currently SANA-WM opt-in. Default `sana_wm_torch_compile_scope` is `off`.
- Regional compile only targets stage-1 `SanaWMBlock` modules. It can still graph-break or recompile when request shape, camera conditioning, chunk settings, or Python-side arguments vary.
- Cache-DiT + regional `torch.compile` can be tested, but should not be a release blocker unless it is a documented supported optimization target.
- Cache-DiT can change numerical behavior. Always inspect output quality and compare against the baseline video.
- Layerwise offload is a memory optimization path, not necessarily a speed path.
- SANA-WM sequence parallelism is not tested because it is not currently supported.
- For TP with the native refiner, disable CFG parallel unless specifically testing CFG parallel with `tp_size=1`.

## Final Sign-off Checklist

Before marking SANA-WM testing complete:

- Unit tests pass in an environment with full dependencies.
- Baseline two-stage single-GPU run succeeds at production shape.
- Stage-1-only run succeeds and confirms refiner overhead.
- At least one memory-pressure mitigation path is validated if needed.
- TP=2 multi-GPU run succeeds.
- CFG parallel is either validated or explicitly documented as not used.
- Cache-DiT and regional `torch.compile` each pass individually before the combined experiment is considered.
- Any `torch.compile` run includes `TORCH_LOGS="+recompiles,graph_breaks"` notes.
- Output videos are archived with matching perf JSON files.
- Any failures include exact command, log path, and first traceback.
