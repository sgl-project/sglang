---
name: sglang-diffusion-modelopt-quant
description: Use when quantizing a diffusion DiT with NVIDIA ModelOpt and making the resulting FP8 or NVFP4 checkpoint loadable, verifiable, and benchmarkable in SGLang Diffusion.
---

# SGLang Diffusion ModelOpt Quant

Use this skill when the task is:
- quantize a diffusion transformer with NVIDIA ModelOpt
- adapt the exported checkpoint to SGLang Diffusion
- validate image correctness against the BF16 reference
- benchmark whether the quantized checkpoint is actually faster

This skill owns the ModelOpt-to-SGLang bridge. It is not a generic kernel-tuning skill.
For denoise benchmarking and profiler workflows, pair it with
[../sglang-diffusion-benchmark-profile/SKILL.md](../sglang-diffusion-benchmark-profile/SKILL.md).

## Read First

Before changing code or exporting a checkpoint, read these sources in order:
- NVIDIA ModelOpt diffusers guide: `examples/diffusers/README.md`
- ModelOpt export entrypoint: `examples/diffusers/quantization/quantize.py`
- ModelOpt diffusers quant config presets: `examples/diffusers/quantization/config.py`
- ModelOpt HF export path: `modelopt/torch/export/unified_export_hf.py`
- Existing SGLang FLUX.2 NVFP4 support: `https://github.com/sgl-project/sglang/pull/20137`

The important implementation fact is:
- ModelOpt diffusers `--format fp8 --hf-ckpt-dir ...` writes float8 weights plus `quantization_config`
- in the validated FLUX.2 flow, the exported diffusers weights still match the original FP16/BF16 tensors, so they must be explicitly converted into `float8_e4m3fn` values for SGLang's native FP8 loader
- it also does not currently materialize the `weight_scale` and `input_scale` tensors that SGLang's diffusion FP8 runtime expects
- the authoritative FP8 scales still exist in `backbone.pt` as `*.weight_quantizer._amax` and `*.input_quantizer._amax`

That means:
- NVFP4 can often use the official diffusers export directly, because the export
  already contains the packed FP4 weights, the scale tensors, and enough
  safetensors metadata for SGLang to reconstruct the quant config without an
  extra conversion pass
- FP8 currently needs one extra materialization step before SGLang can load it natively

## What SGLang Supports Here

This repo now contains:
- diffusion-side `modelopt_fp8` quant config detection
- flat `quant_method: modelopt` + `quant_algo: FP8/NVFP4` resolution
- FLUX.2 NVFP4 packed-QKV detection that distinguishes packed mixed checkpoints from standard diffusers exports
- automatic `dit_cpu_offload/dit_layerwise_offload -> false` protection for diffusion ModelOpt FP8 runs
- a conversion tool for ModelOpt diffusers FP8 exports:
  [`python/sglang/multimodal_gen/tools/convert_modelopt_fp8_checkpoint.py`](../../../tools/convert_modelopt_fp8_checkpoint.py)
- public SGLang-ready FP8 checkpoints for direct consumption:
  - `BBuf/flux2-dev-modelopt-fp8-sglang-transformer`
  - `BBuf/wan22-t2v-a14b-modelopt-fp8-sglang-transformer`

## Generic Vs Model-Specific

The skill is meant to stay generic, but not every step is generic.

Generic pieces:
- use ModelOpt's official `quantize.py` as the source of truth
- convert ModelOpt FP8 exports with `convert_modelopt_fp8_checkpoint.py`
- detect `quant_method=modelopt` + `quant_algo=FP8/NVFP4` inside SGLang diffusion
- validate correctness and performance with fixed prompts, seeds, and perf dumps

Model-specific pieces:
- BF16 fallback patterns in the FP8 converter
- multi-transformer wiring, such as `transformer` + `transformer_2`
- model-specific auto-offload defaults in SGLang
- external runtime dependencies outside diffusers, such as `ltx_core`

Treat FLUX.2 as the validated reference example, not as the only supported shape.

## Preflight

1. Use the H100 or another Hopper+ machine.
2. Export `FLASHINFER_DISABLE_VERSION_CHECK=1`.
3. Verify the base BF16 model already runs in SGLang before quantizing it.
4. Keep a fixed prompt file and seed for all correctness checks.

For gated FLUX models, also export `HF_TOKEN`.

## Clone And Inspect ModelOpt

Clone ModelOpt locally or on the remote validation host and inspect the exact revision you will use.

Validated reference:
- repo: `https://github.com/NVIDIA/Model-Optimizer`
- tested commit: `f7557221e382dbd4d2d0eae35b09887add034624`

Single-H100 FLUX.2 quantization at that commit needed three small local fixes in the ModelOpt clone:
- `pipeline_manager.py`: prefer `enable_sequential_cpu_offload()` when available
- `calibration.py`: merge `extra_params` into the pipeline call kwargs for standard models
- `quantize.py`: when exporting backbone-only checkpoints with CPU offload, remove accelerate hooks and export backbones one by one

If upstream has already fixed those issues in the revision you are using, do not re-apply them.

## FP8 Workflow

### 1. Quantize With Official ModelOpt Script

Use ModelOpt's own `quantize.py`.

Generic single-backbone template:

```bash
python quantize.py \
  --model <model-name> \
  --override-model-path <local-model-snapshot> \
  --model-dtype <Half|BFloat16> \
  --format fp8 \
  --batch-size 1 \
  --calib-size <small-smoke-size> \
  --n-steps <small-smoke-steps> \
  --quantize-mha \
  --prompts-file <fixed-prompt-file> \
  --quantized-torch-ckpt-save-path <out>/ckpt \
  --hf-ckpt-dir <out>/hf
```

For FLUX.2 on a single H100, the smallest validated smoke recipe was:

```bash
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/tmp/modelopt_min
cd /tmp/Model-Optimizer/examples/diffusers/quantization

python quantize.py \
  --model flux2-dev \
  --override-model-path /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
  --model-dtype Half \
  --format fp8 \
  --batch-size 1 \
  --calib-size 1 \
  --n-steps 2 \
  --quantize-mha \
  --cpu-offloading \
  --extra-param height=512 \
  --extra-param width=512 \
  --extra-param max_sequence_length=256 \
  --quantized-torch-ckpt-save-path /tmp/modelopt_flux2_fp8/ckpt \
  --hf-ckpt-dir /tmp/modelopt_flux2_fp8/hf \
  --verbose
```

Outputs to keep:
- `/tmp/modelopt_flux2_fp8/ckpt/backbone.pt`
- `/tmp/modelopt_flux2_fp8/hf/transformer`

### 2. Materialize The FP8 Scales For SGLang

Convert the official export into an SGLang-loadable transformer directory.
This step does two things:
- quantizes each kept FP8 linear weight to `float8_e4m3fn` using ModelOpt's saved `amax / 448` scale
- injects `weight_scale` and `input_scale` tensors into the output shards

This converter currently expects a single quantized backbone per `backbone.pt`.
For multi-transformer models, do separate ModelOpt exports per backbone and run
the converter once per exported component directory.

Command:

```bash
PYTHONPATH=python python3 -m sglang.multimodal_gen.tools.convert_modelopt_fp8_checkpoint \
  --modelopt-hf-dir /tmp/modelopt_flux2_fp8/hf \
  --modelopt-backbone-ckpt /tmp/modelopt_flux2_fp8/ckpt/backbone.pt \
  --base-transformer-dir /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c/transformer \
  --output-dir /tmp/modelopt_flux2_fp8/sglang_transformer \
  --overwrite
```

For FLUX.2, the converter intentionally restores these modules to BF16 by default:
- `time_guidance_embed.*`
- `double_stream_modulation_img.linear`
- `double_stream_modulation_txt.linear`
- `single_stream_modulation.linear`
- `x_embedder`
- `context_embedder`
- `norm_out.linear`

Why:
- the current SGLang diffusion FP8 path was validated with the main transformer blocks quantized
- these top-level FLUX.2 layers should stay BF16 until they are explicitly validated on the same path

### 3. Run The FP8 Checkpoint In SGLang

For a single-transformer model, use `--transformer-weights-path`.

```bash
sglang generate \
  --backend sglang \
  --model-path /root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c \
  --transformer-weights-path /tmp/modelopt_flux2_fp8/sglang_transformer \
  --prompt-path /tmp/modelopt_flux2_fp8/prompt.txt \
  --width 512 \
  --height 512 \
  --num-inference-steps 8 \
  --guidance-scale 4.0 \
  --seed 1234 \
  --save-output \
  --output-path /tmp/modelopt_flux2_fp8/out_fp8 \
  --perf-dump-path /tmp/modelopt_flux2_fp8/perf_fp8.json \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --pin-cpu-memory true
```

Current FP8 caveat:
- diffusion ModelOpt FP8 needs `dit_cpu_offload=false` and
  `dit_layerwise_offload=false`
- the loader now flips both automatically when it detects `modelopt_fp8`
- for multi-transformer models, do not rely on the global `--transformer-weights-path`
  if different components need different checkpoints; use
  `--component_paths.transformer=...` and `--component_paths.transformer_2=...`

If you just want to run the validated checkpoints instead of reproducing the
export/conversion flow, use the published Hugging Face repos directly:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --transformer-path BBuf/flux2-dev-modelopt-fp8-sglang-transformer \
  ...
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --transformer-path BBuf/wan22-t2v-a14b-modelopt-fp8-sglang-transformer \
  ...
```

WAN scope note:
- the published WAN2.2 checkpoint overrides only the primary `transformer`
- `transformer_2` is still taken from the base BF16 model

## NVFP4 Workflow

### 1. Quantize With Official ModelOpt Script

Start from the same BF16 model and export with `--format nvfp4`.

### 2. Use The Export In SGLang

For FLUX.2 there are two checkpoint families:
- packed-QKV mixed checkpoints, such as the existing FLUX.2 NVFP4 path from PR `#20137`
- standard diffusers exports from ModelOpt

SGLang now distinguishes them automatically:
- packed checkpoints enable the fused FLUX.2 QKV loader path
- standard diffusers exports keep the original `to_q/to_k/to_v` layout

That means the skill for NVFP4 is:
- export with ModelOpt
- point `--transformer-weights-path` at the exported NVFP4 transformer directory
- benchmark and validate normally

## Model Family Notes

### FLUX.2

FLUX.2 is the validated FP8 and NVFP4 reference in this skill.
Use it as the first end-to-end example when extending the workflow.

Published SGLang-ready FP8 override:
- `https://huggingface.co/BBuf/flux2-dev-modelopt-fp8-sglang-transformer`

### WAN2.2

Official ModelOpt status:
- `wan2.2-t2v-14b` and `wan2.2-t2v-5b` are listed in the README FP8 script matrix

Important topology fact:
- WAN2.2 A14B is a dual-transformer model in SGLang
- the default ModelOpt command quantizes only `backbone=transformer`
- for full dual-transformer quantization, quantize `transformer` and `transformer_2`
  separately and load them with per-component overrides:
  `--component_paths.transformer=...`
  `--component_paths.transformer_2=...`

Important SGLang fact:
- if `--dit-layerwise-offload` is not passed explicitly, SGLang may auto-enable it
  for WAN2.2
- for benchmark parity, and for FP8 debugging, pin
  `--dit-layerwise-offload false` explicitly

H100 status on `2026-04-07`:
- primary `transformer` ModelOpt FP8 quantization succeeded with the official script
- BF16 SGLang smoke succeeded at `256x256`, `5` frames, `1` step
- converted FP8 checkpoint ran successfully in SGLang when loaded only into
  `transformer` via `--component_paths.transformer=...` while keeping
  `transformer_2` in BF16, with `--dit-layerwise-offload false`
- enabling `--dit-layerwise-offload true` reproduced
  `RuntimeError: mat_b must be a column major tensor` from
  `sgl_kernel.fp8_scaled_mm`; the layerwise offload manager flattens and restores
  FP8 weights in a way that breaks the CUTLASS column-major layout requirement
- a separate `transformer_2` FP8 quantization run completed calibration and saved
  `backbone.pt`, but ModelOpt HF export failed with
  `AttributeError: 'TensorQuantizer' object has no attribute '_amax'`

Current conclusion:
- WAN2.2 FP8 is validated as a single-transformer isolation recipe on H100:
  quantize/export/convert `transformer`, override it with
  `--component_paths.transformer=...`, keep `transformer_2` in BF16, and pin
  `--dit-layerwise-offload false`
- do not present WAN2.2 as a fully quantized dual-transformer FP8 recipe until the
  `transformer_2` export failure is resolved

Published SGLang-ready FP8 override for the benchmarked path:
- `https://huggingface.co/BBuf/wan22-t2v-a14b-modelopt-fp8-sglang-transformer`
- this repo contains only the primary `transformer`
- `transformer_2` remains BF16 from the base WAN2.2 checkpoint

Benchmark snapshot with the same command shape on H100:

| Variant | E2E latency | TextEncoding | Denoising | Decoding | Notes |
| --- | --- | --- | --- | --- | --- |
| BF16 baseline | `6947.04 ms` | `2143.39 ms` | `3800.63 ms` | `985.69 ms` | `256x256`, `5` frames, `1` step, `--dit-layerwise-offload false` |
| FP8 primary `transformer` only | `4415.50 ms` | `1638.98 ms` | `1965.23 ms` | `799.73 ms` | same command shape, `transformer_2` kept BF16 |

Observed deltas from `compare_perf.py`:
- E2E: `-36.4%`
- TextEncoding: `-23.5%`
- Denoising: `-48.3%`
- Decoding: `-18.9%`

Operational note:
- the auto-disable path for `--dit-layerwise-offload true` was validated separately
  with `/tmp/modelopt_wan22_fp8_smoke/fp8_t1_only_layeroffload_autodisable_perf.json`
  to prove the loader-side guard works, but the benchmark table above uses the
  explicit no-layerwise-offload command for parity

### LTX-2

Official ModelOpt status:
- the README documents `ltx-2` only in the FP4 section
- `ltx-2` still appears in `quantize.py --model` choices, so the code path exists,
  but it is not an officially documented FP8 recipe in the README

Environment fact:
- `ltx-2` requires the Lightricks runtime packages such as `ltx_core`
- the stock H100 container used for this skill does not include those packages

H100 status on `2026-04-07`:
- SGLang one-stage `sglang generate` succeeded on H100 at `768x512`, `9` frames,
  `2` steps, so the earlier LTX-2 issue is not caused by the SGLang codebase being
  too old
- SGLang two-stage `LTX2TwoStagePipeline` loaded successfully on H100, but a
  `1`-step smoke run produced NaN timesteps during scheduler preparation; avoid
  `num_inference_steps=1` for LTX-2 smoke checks
- after installing the Lightricks runtime packages and ModelOpt's missing Python
  extras on H100, the `quantize.py --model ltx-2 --format fp8` path advanced
  through pipeline creation, backbone extraction, quantizer insertion, and
  calibration startup
- the current ModelOpt LTX-2 path required several temporary compatibility fixes
  in the ModelOpt clone:
  `pipeline_manager.py` unsupported-kwargs filtering for `TI2VidTwoStagesPipeline`,
  `pipeline_manager.py` bridging the old `stage_1_model_ledger` access to the
  current `DiffusionStage` API,
  `calibration.py` switching from removed
  `DEFAULT_AUDIO_GUIDER_PARAMS/DEFAULT_VIDEO_GUIDER_PARAMS` constants to
  `detect_params(checkpoint_path)`,
  and `calibration.py` forwarding `streaming_prefetch_count` to reduce prompt
  encoder memory pressure
- without `streaming_prefetch_count`, prompt encoding on a single H100 ran out of
  memory while loading Gemma
- with `streaming_prefetch_count=1`, calibration progressed further but the full
  two-stage path still failed when stage 2 fused the distilled LoRA into the FP8
  base checkpoint:
  `RuntimeError: The size of tensor a (8192) must match the size of tensor b (2048)`

Current conclusion:
- SGLang's native LTX-2 runtime path is healthy on H100; the blocker is the
  ModelOpt LTX-2 FP8 path, not the SGLang runtime
- do not claim LTX-2 ModelOpt FP8 as validated in the PR
- the likely next step is a stage-1-only calibration flow, because the current
  quantized backbone under test is the stage-1 transformer and the stage-2
  distilled LoRA path introduces a separate compatibility problem

## Correctness Validation

Never accept a quantized diffusion checkpoint on speed alone.

Minimum validation loop:

1. Fix prompt, seed, resolution, inference steps, guidance scale.
2. Generate a BF16 reference image.
3. Generate the quantized image.
4. Compare both numerically and visually.

A minimal numeric check:

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

ref = np.asarray(Image.open(Path("/tmp/bf16_ref.png")).convert("RGB"), dtype=np.float32)
new = np.asarray(Image.open(Path("/tmp/fp8.png")).convert("RGB"), dtype=np.float32)
mae = np.abs(ref - new).mean()
mse = np.square(ref - new).mean()
psnr = float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)
print({"mae": float(mae), "psnr": float(psnr)})
PY
```

What to require:
- image is visually normal
- no broken composition, collapsed color, or obvious structure loss
- MAE stays low and PSNR stays high enough that the drift is visually negligible

### Trajectory-Latent Cosine Validation

Final-image PSNR is useful, but it only checks the decoded endpoint.
For diffusion quantization, also compare an intermediate denoising step directly.

This repo now includes:
- `python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py`

It runs a BF16 reference and a quantized candidate with identical sampling args,
captures `return_trajectory_latents=True`, and reports:
- per-step cosine similarity on the denoising latents
- per-step MAE / RMSE / max-abs
- frame-level MAE / PSNR for the final output

Validated smoke-style usage pattern:

```bash
PYTHONPATH=python python3 -m sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity \
  --model-path <base-model> \
  --prompt "<fixed prompt>" \
  --width <w> --height <h> \
  --num-inference-steps <small deterministic step count> \
  --guidance-scale <scale> \
  --seed <seed> \
  --candidate-transformer-path <sglang-fp8-transformer> \
  --output-json /tmp/modelopt_similarity.json
```

Recommended reduced configs for trajectory capture:
- FLUX.2: `512x512`, `8` steps, fixed seed, same offload flags on both sides
- WAN2.2: `704x512`, `21` frames, `2` steps, fixed seed, same `--enable-cfg-parallel`, `--ulysses-degree`, and offload flags on both sides

Why reduced configs:
- `return_trajectory_latents` stores every denoising-step latent tensor
- that makes exact-nightly video settings unnecessarily heavy for accuracy smoke checks
- the goal here is deterministic quant drift measurement, not throughput benchmarking

Publish a recipe only after both checks look healthy:
- decoded output remains visually normal
- selected denoising-step cosine stays very high and per-step errors remain small

Validated H100 smoke numbers from the new tool:

| Model | Reduced config | Selected step | Latent cosine | Latent MAE | Frame-0 PSNR | Frame-0 MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FLUX.2 | `512x512`, `8` steps | `7` | `0.9971` | `0.0405` | `34.67 dB` | `2.43` |
| WAN2.2 | `704x512`, `21` frames, `2` steps | `1` | `0.9755` | `0.2310` | `21.28 dB` | `18.05` |

Do not rely on only one prompt before publishing a recipe. Use several prompts with:
- simple object composition
- text-heavy / detailed prompts
- high-frequency texture prompts

## Performance Validation

Follow the perf-dump workflow from
[../sglang-diffusion-benchmark-profile/SKILL.md](../sglang-diffusion-benchmark-profile/SKILL.md).

For every baseline / quantized pair:

```bash
sglang generate ... --perf-dump-path /tmp/baseline.json
sglang generate ... --perf-dump-path /tmp/quantized.json

python3 python/sglang/multimodal_gen/benchmarks/compare_perf.py \
  /tmp/baseline.json \
  /tmp/quantized.json
```

Important interpretation rule:
- a benchmark comparison is only valid when the BF16 and FP8 commands use the same offload configuration
- keep `dit_cpu_offload`, `text_encoder_cpu_offload`, `vae_cpu_offload`, `dit_layerwise_offload`, GPU count, TP size, prompt, seed, resolution, and step count identical between the two runs
- if one side needs a different offload setup to fit, do not present that result as a benchmark; label it explicitly as a deployability or practical latency comparison instead
- for WAN2.2, explicitly pass `--dit-layerwise-offload false` if that is the intended benchmark condition; otherwise SGLang may auto-enable it

For apples-to-apples compute benchmarking:
- use a setup where both BF16 and FP8 fit under the exact same command shape
- if single-GPU BF16 does not fit, move both BF16 and FP8 to the same multi-GPU topology and keep the offload flags unchanged between them

Validated benchmark tables to cite:

### FLUX.2, 2x H100 TP2, same BF16/FP8 command shape

| Variant | Total | Text | Denoising | Decoding | Total delta | Denoise delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | `13.75 s` | `0.51 s` | `12.84 s` | `0.01 s` | - | - |
| FP8 | `9.84 s` | `0.51 s` | `8.93 s` | `0.01 s` | `-28.4%` | `-30.5%` |

Profiler rerun on the same topology:

| Variant | Total | Text | Denoising | Decoding | Total delta | Denoise delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 profile | `27.67 s` | `0.51 s` | `26.72 s` | `0.01 s` | - | - |
| FP8 profile | `21.29 s` | `0.51 s` | `20.34 s` | `0.01 s` | `-23.1%` | `-23.9%` |

### WAN2.2 A14B, 4x H100 exact-nightly payload, no torch.compile

The exact-nightly BF16/FP8 runs were pinned to:
- same `720p`, `81` frames, `40` steps, seed, prompt, GPU topology
- same `--dit-cpu-offload false`
- same `--dit-layerwise-offload false`
- same allocator environment:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

| Variant | Total | Text | Denoising | Decoding | Total delta | Denoise delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | `212.19 s` | `1.00 s` | `204.09 s` | `6.91 s` | - | - |
| FP8 | `204.38 s` | `1.00 s` | `196.28 s` | `6.92 s` | `-3.68%` | `-3.83%` |

5-step profiler rerun on the same no-compile setup:

| Variant | Total | Text | Denoising | Decoding | Total delta | Denoise delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 profile | `227.77 s` | `1.00 s` | `219.62 s` | `6.96 s` | - | - |
| FP8 profile | `223.82 s` | `1.00 s` | `215.69 s` | `7.02 s` | `-1.74%` | `-1.79%` |

## PR-Ready Change Summary

Code areas to highlight in the PR:

| File | Why it changed |
| --- | --- |
| `runtime/layers/quantization/__init__.py` | registers `modelopt` / `modelopt_fp8` diffusion quantization methods |
| `runtime/layers/quantization/modelopt_quant.py` | adds `ModelOptFp8Config`, the diffusion FP8 linear path, and shared ModelOpt config handling |
| `runtime/utils/quantization_utils.py` | resolves flat ModelOpt FP8 configs and distinguishes standard diffusers exports from packed NVFP4 cases |
| `runtime/loader/transformer_load_utils.py` | disables incompatible DiT offload modes for ModelOpt FP8 loads |
| `runtime/models/dits/flux_2.py` | keeps packed-QKV handling specific to checkpoints that actually use the packed NVFP4 layout |
| `tools/convert_modelopt_fp8_checkpoint.py` | materializes SGLang-loadable FP8 checkpoints from ModelOpt exports |
| `test/unit/test_diffusion_modelopt_quant.py` | covers FP8 quant-config resolution and the offload guard |
| `test/unit/test_convert_modelopt_fp8_checkpoint.py` | covers FP8 checkpoint conversion and BF16 fallback behavior |

## Validated H100 Reference

Key validated H100 paths:
- local SGLang validation tree: `/tmp/sglang_local_validate_wan`
- raw FLUX.2 ModelOpt FP8 export: `/tmp/modelopt_flux2_fp8_half_smoke5/hf/transformer`
- raw FLUX.2 ModelOpt backbone checkpoint: `/tmp/modelopt_flux2_fp8_half_smoke5/ckpt/backbone.pt`
- converted FLUX.2 SGLang transformer: `/tmp/modelopt_flux2_fp8_half_smoke5/sglang_transformer_v2`
- raw WAN2.2 ModelOpt FP8 export: `/tmp/modelopt_wan22_fp8_smoke/hf/transformer`
- raw WAN2.2 ModelOpt backbone checkpoint: `/tmp/modelopt_wan22_fp8_smoke/ckpt/backbone.pt`
- converted WAN2.2 SGLang transformer with the current local converter: `/tmp/modelopt_wan22_h100_current/sglang_transformer`

Desktop artifact roots from the validated reruns:
- FLUX.2 benchmark/profile artifacts:
  `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408`
- WAN2.2 benchmark/profile artifacts:
  `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22`
- trajectory similarity JSONs copied from H100 can be stored alongside those benchmark roots if they are needed for PR review

Important files worth attaching to PR notes:
- FLUX.2 images:
  `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408/flux2_bf16.png`
  `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408/flux2_fp8.png`
- FLUX.2 traces:
  `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408/profile/bf16/trace.json.gz`
  `/Users/bbuf/Desktop/flux2_h100_2gpu_20260408/profile/fp8/trace.json.gz`
- WAN2.2 videos:
  `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/bf16_nocompile/wan22_bf16_nocompile.mp4`
  `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/fp8_nocompile/wan22_fp8_nocompile.mp4`
- WAN2.2 traces:
  `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/profile/bf16/trace.json.gz`
  `/Users/bbuf/Desktop/modelopt_nightly_rerun_20260408/wan22/profile/fp8/trace.json.gz`
- H100 trajectory similarity summaries:
  `/tmp/modelopt_accuracy/flux2_similarity.json`
  `/tmp/modelopt_accuracy/wan22_similarity.json`

## Hand-off Rule

If the quantized checkpoint is correct but not faster:
- stop
- collect perf dumps and a profiler trace
- hand the bottleneck to
  [../sglang-diffusion-benchmark-profile/SKILL.md](../sglang-diffusion-benchmark-profile/SKILL.md)
  or a lower-level optimization skill
