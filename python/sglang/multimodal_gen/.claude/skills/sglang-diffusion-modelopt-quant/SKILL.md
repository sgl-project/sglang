---
name: sglang-diffusion-modelopt-quant
description: Use when quantizing a diffusion DiT with NVIDIA ModelOpt and making the resulting FP8 or NVFP4 checkpoint loadable, verifiable, and benchmarkable in SGLang Diffusion.
---

# SGLang Diffusion ModelOpt Quant

## Overview

Use this skill when the task is to take a diffusion transformer through the full ModelOpt workflow:

- quantize it with NVIDIA ModelOpt
- adapt the exported checkpoint to SGLang Diffusion
- verify that quality holds up
- benchmark whether the quantized checkpoint is actually faster

This skill owns the ModelOpt-to-SGLang bridge. It is not a generic kernel-tuning skill.

## Core Rules

- Use ModelOpt's official `quantize.py` as the PTQ source of truth.
- Keep the workflow generic. Put model-specific fallback logic in small isolated branches, not in the main conversion path.
- Benchmark only when BF16 and quantized commands are identical except for the checkpoint override being tested.
- For diffusion FP8, keep `dit_cpu_offload=false`. `dit_layerwise_offload=true` is valid on the fixed path when you want lower DiT residency.
- For multi-transformer pipelines, use per-component overrides when different components need different checkpoints.
- For B200 NVFP4 validation, keep backend-sensitive environment variables explicit. Wan2.2 NVFP4 is commonly validated with `SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn`; benchmark the default CUTLASS path separately if that is what you are evaluating.
- When a branch is missing the validated helper tools, refresh `python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py`, `python/sglang/multimodal_gen/tools/build_modelopt_nvfp4_transformer.py`, and `python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py` instead of inventing one-off scripts elsewhere.
- After validating a new ModelOpt quant path, update the ModelOpt support matrix in `docs/diffusion/quantization.md` before closing the task.

## Read First

Read these sources before changing code:

- NVIDIA ModelOpt diffusers guide: `examples/diffusers/README.md`
- ModelOpt quantization entrypoint: `examples/diffusers/quantization/quantize.py`
- ModelOpt diffusers quant presets: `examples/diffusers/quantization/config.py`
- SGLang diffusion quant runtime:
  - `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py`
  - `python/sglang/multimodal_gen/runtime/utils/quantization_utils.py`
  - `python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py`
- Helper tools in this repo:
  - [`python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py`](../../../tools/build_modelopt_fp8_transformer.py)
  - [`python/sglang/multimodal_gen/tools/build_modelopt_nvfp4_transformer.py`](../../../tools/build_modelopt_nvfp4_transformer.py)
  - [`python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py`](../../../tools/compare_diffusion_trajectory_similarity.py)

If you are working on a new model family, inspect the transformer's config and tensor naming before changing the generic converter.

## What SGLang Supports Here

This repo now contains:

- flat `quant_method=modelopt` plus `quant_algo=FP8/NVFP4` resolution
- diffusion-side ModelOpt FP8 linear loading
- diffusion-side NVFP4 loading from ModelOpt exports
- FLUX.2 packed-QKV detection that distinguishes packed NVFP4 checkpoints from standard diffusers exports
- automatic protection against incompatible FP8 CPU offload while keeping layerwise DiT offload available
- FP8 transformer build:
  [`python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py`](../../../tools/build_modelopt_fp8_transformer.py)
- NVFP4 mixed transformer build:
  [`python/sglang/multimodal_gen/tools/build_modelopt_nvfp4_transformer.py`](../../../tools/build_modelopt_nvfp4_transformer.py)
- trajectory similarity validation:
  [`python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py`](../../../tools/compare_diffusion_trajectory_similarity.py)

Validated documentation and CI coverage currently center on these ModelOpt diffusion transformer override families:

- FP8: FLUX.1-dev, FLUX.2-dev, Wan2.2, HunyuanVideo, Qwen Image, Qwen Image Edit
- NVFP4: FLUX.1-dev, FLUX.2-dev, Wan2.2

Treat a new family, a new precision, or a new checkpoint layout as unsupported until it has a documented matrix row and a matching validation story.
Before writing CLI examples, re-read the active branch's `docs/diffusion/quantization.md`: FLUX.2 NVFP4 is an official `black-forest-labs/*` repo rather than a `lmsys/*` converted repo, and its preferred flag depends on the current documented loader flow. Use `--transformer-path` for a component override directory with `config.json`; use `--transformer-weights-path` when the repo or path should be probed as raw weights.

B200 CI coverage can include loose BF16-vs-quantized quality checks. Inspect the active branch's `run_suite.py` before assuming they are part of the suite; mainline and feature branches may differ. Those checks are intended to catch blank, corrupted, or obviously divergent images, not exact image parity.

Mainline documentation now uses `lmsys/*` for the eight converted ModelOpt
checkpoint repos; the FLUX.2 NVFP4 raw export remains
`black-forest-labs/FLUX.2-dev-NVFP4`. Do not use older `BBuf/*` examples unless
you are explicitly testing a historical branch.

## Related PR Watchlist

As of 2026-05-04, these related SGLang PRs are relevant to ModelOpt diffusion
support. Treat unmerged items as future support or migration work until the
docs/CI matrix is updated.

- #23155 added Qwen Image ModelOpt FP8 support.
- #23199 adds HunyuanVideo ModelOpt FP8 support.
- #23373 adds a runtime quantization flag; keep PTQ/export workflows separate from runtime quant examples until the CLI behavior is merged.
- #24024 adds transformer FP8-cast compatibility mode.
- #24186 re-enables B200 multimodal CI with NVFP4 fixes for FLUX.2 and Wan2.2.

Do not expand the validated matrix beyond the documented rows solely because a
related PR exists. Add a row only after the exact checkpoint, loader path,
accuracy check, and benchmark scope are validated on the active branch.

## Documentation Maintenance

- Keep the validated ModelOpt support matrix in `docs/diffusion/quantization.md`.
- Each row should record the validated scope, the Hugging Face repo or path for the quantized DiT weights, and the key caveats.
- If the quantized DiT weights are not published yet, write `unpublished` explicitly instead of leaving the field blank.

## FP8 Vs NVFP4

FP8 and NVFP4 are not wired into SGLang in exactly the same way.

FP8:

- the validated ModelOpt diffusers FP8 export still needs an extra SGLang-side conversion step
- SGLang expects explicit `weight_scale` and `input_scale`
- the validated path also materializes SGLang-native `float8_e4m3fn` weights from `backbone.pt`

NVFP4:

- the official diffusers export often already contains packed FP4 weights, scale tensors, and enough safetensors metadata for SGLang to rebuild the quant config
- in that case SGLang mainly needs to detect the checkpoint family and rearrange tensors into the runtime layout
- this is why NVFP4 often does not need an extra offline conversion pass like FP8 does
- backend choice matters on B200; record whether the run used the default CUTLASS path or a cuDNN-backed FlashInfer FP4 GEMM path

Important caveat:

- "often" does not mean "always"
- the exact load path still depends on the checkpoint family, especially whether a model uses a packed-QKV layout

## Generic Workflow

### 1. Verify The BF16 Baseline First

Before quantizing anything:

- run the original BF16 model in SGLang
- fix the prompt, seed, size, step count, and GPU topology
- save the output and `perf.json`

Do not start quantization work until the BF16 path is already healthy.

### 2. Quantize With Official ModelOpt

Use ModelOpt's official script. Generic template:

```bash
python quantize.py \
  --model <model-name> \
  --override-model-path <hf-repo-or-local-model> \
  --model-dtype <Half|BFloat16> \
  --format <fp8|fp4> \
  --batch-size 1 \
  --calib-size <calib-size> \
  --n-steps <calib-steps> \
  --quantize-mha \
  --prompts-file <prompt-file> \
  --quantized-torch-ckpt-save-path <out>/ckpt \
  --hf-ckpt-dir <out>/hf
```

For current ModelOpt diffusion examples, use `--format fp4` for NVFP4 exports.
Do not assume the checked-out ModelOpt version accepts a literal `nvfp4` format string unless you verified it locally.

For multi-transformer models:

- quantize each backbone deliberately
- keep each output directory separate
- save both `backbone.pt` and the matching `hf/<component>` export

### 3. Convert FP8 Exports For SGLang

FP8 requires an extra conversion step:

```bash
PYTHONPATH=python python3 -m sglang.multimodal_gen.tools.build_modelopt_fp8_transformer \
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
- strips stale `_quantizer.*` tensors and fallback-layer scales that should not survive into the SGLang-native checkpoint

For `FLUX.1-dev`, the validated fallback set currently keeps these modules in BF16:

- `transformer_blocks.*.norm1.linear`
- `transformer_blocks.*.norm1_context.linear`
- `transformer_blocks.*.ff.net.0.proj`
- `transformer_blocks.*.ff.net.2`
- `transformer_blocks.*.ff_context.net.0.proj`
- `transformer_blocks.*.ff_context.net.2`
- `single_transformer_blocks.*.norm.linear`
- `single_transformer_blocks.*.proj_mlp`

Use `--model-type flux1` to force that profile, or rely on `--model-type auto` when the export config identifies `FluxTransformer2DModel`.

HunyuanVideo uses `HunyuanVideoTransformer3DModel`, so the validated
HunyuanVideo FP8 fallback preset keeps these modules in BF16:

- `context_embedder.*`
- `x_embedder.proj`
- `time_text_embed.(timestep_embedder|guidance_embedder|text_embedder).linear_[12]`
- `norm_out.linear`
- `proj_out`
- `transformer_blocks.*.norm1.linear`
- `transformer_blocks.*.norm1_context.linear`
- `single_transformer_blocks.*.norm.linear`

Use `--model-type hunyuan-video` to force that profile, or rely on
`--model-type auto` when the export config identifies
`HunyuanVideoTransformer3DModel`.

HunyuanVideo ModelOpt exports use diffusers module names that differ from
SGLang runtime names for fused QKV and fused QKV+MLP layers. Keep the
diffusers-to-runtime mapping in `build_modelopt_fp8_transformer.py` in sync
with `runtime/models/dits/hunyuanvideo.py` before trusting converted scale
tensors.

Qwen Image and Qwen Image Edit share `QwenImageTransformer2DModel`, so one
ModelOpt FP8 fallback preset covers both. The validated Qwen Image fallback set
keeps these modules in BF16:

- `img_in`
- `txt_in`
- `time_text_embed.timestep_embedder.linear_1`
- `time_text_embed.timestep_embedder.linear_2`
- `norm_out.linear`
- `proj_out`
- `transformer_blocks.*.img_mlp.net.2`
- `transformer_blocks.*.img_mod`
- `transformer_blocks.*.txt_mod`

Use `--model-type qwen-image` to force that profile, or rely on
`--model-type auto` when the export config identifies
`QwenImageTransformer2DModel`.

Qwen modulation weights can appear in safetensors as `.img_mod.1.weight` and
`.txt_mod.1.weight`. Canonicalize those module names to `.img_mod` and
`.txt_mod` before fallback matching.

For Qwen Image FP8, explicit BF16 fallback tensors must be written before
honoring ModelOpt ignored weights. Otherwise converter stats can report a
fallback while the output checkpoint still retains the source FP8 tensor, which
causes severe image-quality regressions.

For FLUX.1-dev NVFP4 model families that need a mixed BF16+NVFP4 checkpoint, build the merged transformer explicitly:

```bash
PYTHONPATH=python python3 -m sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer \
  --base-transformer-dir <base-model-transformer-dir> \
  --modelopt-hf-dir <out>/hf/transformer \
  --output-dir <out>/transformer-mixed \
  --pattern-preset flux1-nvfp4
```

The validated FLUX.1-dev mixed builder also needs to preserve:

- `quant_type: NVFP4` in `config.json`
- `swap_weight_nibbles: false` for the validated diffusers export

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

- keep prompt, seed, resolution, and step count fixed
- compare BF16 and quantized runs
- capture denoising trajectories
- inspect per-step latent cosine similarity plus MAE or RMSE
- compare final frames with image metrics such as PSNR or MAE

Tool:

```bash
PYTHONPATH=python python3 -m sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity \
  --model-path <base-model> \
  --model-id <optional-native-model-id> \
  --prompt "<prompt>" \
  --width <w> \
  --height <h> \
  --num-inference-steps <steps> \
  --guidance-scale <cfg> \
  --seed <seed> \
  --candidate-transformer-path <quantized-transformer> \
  --output-json <report.json>
```

Use `--model-id FLUX.1-dev` when `--model-path` points to a local directory but the runtime still needs the native FLUX.1 model registration.

Full-output validation:

- run the same user-facing generation config in BF16 and quantized mode
- inspect the output visually
- only claim "quality preserved" for the exact scope you actually checked

### 6. Benchmark Correctly

Benchmark only when these match between BF16 and quantized:

- prompt
- seed
- width and height
- frame count
- inference step count
- GPU count and topology
- offload flags
- compile settings
- profiler settings

Only the quantized checkpoint path should differ.

Interpretation rule:

- the primary expected gain is in denoising
- text-encoding and VAE differences are secondary and should not be over-attributed unless they were quantized too

### 7. Add Model-Specific Fallbacks Only When Needed

If the generic FP8 path fails on a new model family:

- inspect which modules are numerically sensitive or loader-incompatible
- keep fallback patterns small and explicit
- isolate them in the converter instead of scattering ad-hoc exceptions
- re-run deterministic trajectory checks after every fallback change

Do not turn one validated model quirk into a generic rule unless another family also needs it.

## FP8 Offload Constraint

Current diffusion ModelOpt FP8 support requires:

- `dit_cpu_offload=false`
- `dit_layerwise_offload` may be enabled when you want lower DiT residency

Reason:

- the FP8 linear path depends on a CUTLASS-compatible weight layout after loading
- `dit_cpu_offload` is still treated conservatively
- the fixed layerwise offload path now preserves non-contiguous tensor strides instead of flattening and rebuilding FP8 weights into a contiguous layout

Runtime behavior:

- SGLang still force-disables `dit_cpu_offload` when it detects `modelopt_fp8`
- benchmark commands should still pin offload flags explicitly so the command line itself makes the comparison rule obvious

## Claim Discipline

When documenting results:

- claim only scopes that were actually validated end to end
- do not collapse "single-transformer FP8 override" into "full-model FP8"
- do not call a practical deployment comparison a benchmark if BF16 and quantized commands used different offload behavior

## Current Code Areas

| File | Role |
| --- | --- |
| `runtime/layers/quantization/__init__.py` | registers diffusion quant methods |
| `runtime/layers/quantization/modelopt_quant.py` | ModelOpt FP8 and NVFP4 runtime loading |
| `runtime/utils/quantization_utils.py` | resolves flat ModelOpt configs and reconstructs NVFP4 config from metadata |
| `runtime/loader/transformer_load_utils.py` | guards incompatible FP8 offload modes |
| `runtime/models/dits/flux_2.py` | packed-QKV handling for the packed FLUX.2 NVFP4 family |
| `tools/build_modelopt_fp8_transformer.py` | Build an SGLang-loadable FP8 transformer from a ModelOpt export |
| `tools/build_modelopt_nvfp4_transformer.py` | Build mixed BF16+NVFP4 transformer directories when a family needs preserved BF16 layers |
| `tools/compare_diffusion_trajectory_similarity.py` | reduced deterministic BF16-vs-quantized validation |
| `docs/diffusion/quantization.md` | public ModelOpt support matrix and CLI examples |
| `test/server/testcase_configs.py` | reusable ModelOpt testcase constants, thresholds, and helpers |
| `test/server/gpu_cases.py` | concrete GPU and B200 ModelOpt CI case lists |
