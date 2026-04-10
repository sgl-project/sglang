---
name: sglang-diffusion-modelopt-quant
description: Use when quantizing a diffusion DiT with NVIDIA ModelOpt and making the resulting FP8 or NVFP4 checkpoint loadable, verifiable, and benchmarkable in SGLang Diffusion.
---

# SGLang Diffusion ModelOpt Quant

Use this skill when the task is:
- quantize a diffusion transformer with NVIDIA ModelOpt
- adapt the exported checkpoint to SGLang Diffusion
- validate that quality remains acceptable
- benchmark whether the quantized checkpoint is actually faster

This skill owns the ModelOpt-to-SGLang bridge. It is not a generic kernel-tuning skill.

## Core Rules

- Use ModelOpt's official `quantize.py` as the source of truth for PTQ.
- Keep the workflow generic. Put model-specific constraints in short notes, not in the main procedure.
- Benchmark only when BF16 and quantized commands are identical except for the checkpoint override being tested.
- For diffusion FP8, pin `dit_cpu_offload=false` and `dit_layerwise_offload=false`.
- For multi-transformer pipelines, use per-component overrides if different components need different checkpoints.

## Read First

Read these sources before changing code:
- NVIDIA ModelOpt diffusers guide: `examples/diffusers/README.md`
- ModelOpt quantization entrypoint: `examples/diffusers/quantization/quantize.py`
- ModelOpt diffusers quant presets: `examples/diffusers/quantization/config.py`
- SGLang diffusion quant runtime:
  - `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py`
  - `python/sglang/multimodal_gen/runtime/utils/quantization_utils.py`
  - `python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py`
- Existing FLUX.2 NVFP4 reference: `https://github.com/sgl-project/sglang/pull/20137`

## What SGLang Supports Here

This repo now contains:
- flat `quant_method=modelopt` plus `quant_algo=FP8/NVFP4` resolution
- diffusion-side ModelOpt FP8 linear loading
- diffusion-side NVFP4 loading from ModelOpt exports
- FLUX.2 packed-QKV detection that distinguishes packed NVFP4 checkpoints from standard diffusers exports
- automatic protection against incompatible FP8 offload modes
- FP8 export conversion:
  [`python/sglang/multimodal_gen/tools/convert_modelopt_fp8_checkpoint.py`](../../../tools/convert_modelopt_fp8_checkpoint.py)
- trajectory similarity validation:
  [`python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py`](../../../tools/compare_diffusion_trajectory_similarity.py)

## FP8 Vs NVFP4

FP8 and NVFP4 are not wired into SGLang in exactly the same way.

FP8:
- the validated ModelOpt diffusers FP8 export still needs an extra SGLang-side conversion step
- SGLang expects explicit `weight_scale` and `input_scale`
- the validated path also materializes SGLang-native `float8_e4m3fn` weights from `backbone.pt`

NVFP4:
- the official diffusers export often already contains the packed FP4 weights, scale tensors, and enough safetensors metadata for SGLang to rebuild the quant config
- in that case SGLang mainly needs to detect the checkpoint family and rearrange tensors into the runtime layout
- this is why NVFP4 often does not need an extra offline conversion pass like FP8 does

Important caveat:
- "often" does not mean "always"
- the exact load path still depends on the checkpoint family, especially whether a model uses a packed-QKV layout

## Generic Workflow

### 1. Verify The BF16 Baseline First

Before quantizing anything:
- run the original BF16 model in SGLang
- fix the prompt, seed, size, steps, and GPU topology
- save the output and `perf.json`

Do not start quantization work until the BF16 path is already healthy.

### 2. Quantize With Official ModelOpt

Use ModelOpt's official script. Generic template:

```bash
python quantize.py \
  --model <model-name> \
  --override-model-path <hf-repo-or-local-model> \
  --model-dtype <Half|BFloat16> \
  --format <fp8|nvfp4> \
  --batch-size 1 \
  --calib-size <calib-size> \
  --n-steps <calib-steps> \
  --quantize-mha \
  --prompts-file <prompt-file> \
  --quantized-torch-ckpt-save-path <out>/ckpt \
  --hf-ckpt-dir <out>/hf
```

For multi-transformer models, quantize each backbone deliberately:
- if ModelOpt supports quantizing all target backbones in one run, use that
- otherwise quantize/export per backbone and keep each output directory separate

Keep these outputs:
- `backbone.pt`
- `hf/<component>`

### 3. Convert FP8 Exports For SGLang

FP8 requires an extra conversion step:

```bash
PYTHONPATH=python python3 -m sglang.multimodal_gen.tools.convert_modelopt_fp8_checkpoint \
  --modelopt-hf-dir <out>/hf \
  --modelopt-backbone-ckpt <out>/ckpt/backbone.pt \
  --base-transformer-dir <base-model-transformer-dir> \
  --output-dir <out>/sglang_transformer \
  --overwrite
```

What the converter does:
- reads `weight_quantizer._amax` and `input_quantizer._amax` from `backbone.pt`
- writes `weight_scale` and `input_scale`
- materializes eligible FP8 weights as `float8_e4m3fn`
- preserves ModelOpt `ignore` layers as BF16

For multi-transformer models:
- run the converter once per exported backbone
- keep each converted component in its own output directory

### 4. Load The Quantized Checkpoint In SGLang

Single-transformer example:

```bash
sglang generate \
  --model-path <base-model> \
  --transformer-path <quantized-transformer> \
  --prompt "<prompt>" \
  --seed <seed> \
  --save-output
```

Multi-transformer example:

```bash
sglang generate \
  --model-path <base-model> \
  --transformer-path <quantized-transformer> \
  --transformer-2-path <another-transformer-or-bf16-override> \
  --prompt "<prompt>" \
  --seed <seed> \
  --save-output
```

Guideline:
- use the global `--transformer-path` only when the model effectively has one transformer override to apply
- use per-component overrides when different backbones need different checkpoints
- the preferred CLI form is `--<component>-path`
- config-expanded forms such as `--component_paths.transformer_2=...` also resolve to the same internal override map

### 5. Validate Accuracy

Use two levels of validation.

Reduced deterministic validation:
- use a small fixed configuration first
- compare BF16 and quantized runs with identical prompt and seed
- capture denoising trajectories
- compare:
  - per-step latent cosine similarity
  - latent MAE / RMSE
  - final image or frame PSNR / MAE

Tool:

```bash
PYTHONPATH=python python3 -m sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity \
  ...
```

Full-output validation:
- run the same user-facing generation config in BF16 and quantized mode
- inspect the output visually
- for images, compare the final image directly
- for videos, compare representative frames and, when practical, aggregate all-frame metrics

### 6. Benchmark Correctly

Benchmark only when these match between BF16 and quantized:
- prompt
- seed
- width and height
- number of frames
- number of inference steps
- GPU count and topology
- offload flags
- compile settings
- profiler settings

Only the quantized checkpoint path should differ.

Interpretation rule:
- the primary expected gain is in denoising
- text-encoding and VAE differences are secondary and should not be over-attributed unless they were quantized too

## FP8 Offload Constraint

Current diffusion ModelOpt FP8 support requires:
- `dit_cpu_offload=false`
- `dit_layerwise_offload=false`

Reason:
- the FP8 linear path depends on a CUTLASS-compatible weight layout after loading
- the offload and restore path does not preserve that layout
- in particular, layerwise offload can flatten and rebuild FP8 weights in a way that breaks the column-major requirement used by the FP8 GEMM path

Runtime behavior:
- SGLang currently force-disables these two flags when it detects `modelopt_fp8`
- benchmark commands should still pin them explicitly so the command line itself makes the comparison rule obvious

## Reference Examples

Keep examples short and treat them as references, not as the main workflow.

### FLUX.2

Validated reference for:
- ModelOpt FP8
- ModelOpt NVFP4

Base model:
- `black-forest-labs/FLUX.2-dev`

Published SGLang-ready FP8 override:
- `https://huggingface.co/BBuf/flux2-dev-modelopt-fp8-sglang-transformer`

Typical load pattern:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --transformer-path BBuf/flux2-dev-modelopt-fp8-sglang-transformer \
  ...
```

### WAN2.2

Validated benchmarked path:
- quantized primary `transformer`
- BF16 `transformer_2`

Base model:
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers`

Published SGLang-ready FP8 override:
- `https://huggingface.co/BBuf/wan22-t2v-a14b-modelopt-fp8-sglang-transformer`

Typical load pattern:

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --transformer-path BBuf/wan22-t2v-a14b-modelopt-fp8-sglang-transformer \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  ...
```

If you later validate dual-transformer quantization:
- keep `transformer` and `transformer_2` outputs separate
- load them with per-component overrides
- do not describe that as equivalent to the current single-transformer benchmarked path unless it was benchmarked separately

## Claim Discipline

When documenting results:
- claim only scopes that were actually validated end-to-end
- do not collapse "primary-transformer FP8" into "full model FP8"
- do not call a practical deployment comparison a benchmark if BF16 and quantized commands used different offload behavior

## Current Code Areas

| File | Role |
| --- | --- |
| `runtime/layers/quantization/__init__.py` | registers diffusion quant methods |
| `runtime/layers/quantization/modelopt_quant.py` | ModelOpt FP8 and NVFP4 runtime loading |
| `runtime/utils/quantization_utils.py` | resolves flat ModelOpt configs and reconstructs NVFP4 config from metadata |
| `runtime/loader/transformer_load_utils.py` | guards incompatible FP8 offload modes |
| `runtime/models/dits/flux_2.py` | packed-QKV handling for the packed FLUX.2 NVFP4 family |
| `tools/convert_modelopt_fp8_checkpoint.py` | FP8 offline conversion into SGLang-native layout |
| `tools/compare_diffusion_trajectory_similarity.py` | reduced deterministic BF16-vs-quantized validation |
