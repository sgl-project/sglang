# Quantization

SGLang-Diffusion supports quantized transformer checkpoints. In most cases, keep
the base model and the quantized transformer override separate.

## Quick Reference

Use these paths:

- `--model-path`: the base or original model
- `--transformer-path`: a quantized transformers-style transformer component directory that already contains its own `config.json`
- `--transformer-weights-path`: quantized transformer weights provided as a single safetensors file, a sharded safetensors directory, a local path, or a Hugging Face repo ID

Recommended example:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --transformer-weights-path black-forest-labs/FLUX.2-dev-NVFP4 \
  --prompt "a curious pikachu"
```

For quantized transformers-style transformer component folders:

```bash
sglang generate \
  --model-path /path/to/base-model \
  --transformer-path /path/to/quantized-transformer \
  --prompt "A Logo With Bold Large Text: SGL Diffusion"
```

NOTE: Some model-specific integrations also accept a quantized repo or local
directory directly as `--model-path`, but that is a compatibility path. If a
repo contains multiple candidate checkpoints, pass
`--transformer-weights-path` explicitly.

## Quant Families

Here, `quant_family` means a checkpoint and loading family with shared CLI
usage and loader behavior. It is not just the numeric precision or a kernel
backend.

| quant_family      | checkpoint form                                                                            | canonical CLI                                                          | supported models                        | extra dependency                      | platform / notes                                                                                                                       |
|-------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------|-----------------------------------------|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `fp8`             | Quantized transformer component folder, or safetensors with `quantization_config` metadata | `--transformer-path` or `--transformer-weights-path`                   | ALL                                     | None                                  | Component-folder and single-file flows are both supported                                                                              |
| `modelopt-fp8`    | Converted ModelOpt FP8 transformer directory or repo with `config.json`                    | `--transformer-path`                                                    | FLUX.1, FLUX.2, Wan2.2, HunyuanVideo, Qwen Image, Qwen Image Edit | None                                  | Serialized config stays `quant_method=modelopt` with `quant_algo=FP8`; `dit_layerwise_offload` is supported and `dit_cpu_offload` stays disabled |
| `modelopt-nvfp4`  | Mixed transformer directory/repo with `config.json`, or raw NVFP4 safetensors export/repo | `--transformer-path` for mixed overrides; `--transformer-weights-path` for raw exports | FLUX.1, FLUX.2, Wan2.2                  | None                                  | Mixed override repos keep the base model separate; raw exports such as `black-forest-labs/FLUX.2-dev-NVFP4` still use the weights-path flow |
| `nunchaku-svdq`   | Pre-quantized Nunchaku transformer weights, usually named `svdq-{int4\|fp4}_r{rank}-...`   | `--transformer-weights-path`                                           | Model-specific support such as Qwen-Image, FLUX, and Z-Image | `nunchaku`                            | SGLang can infer precision and rank from the filename and supports both `int4` and `nvfp4`                                             |
| `msmodelslim`     | Pre-quantized msmodelslim transformer weights                                              | `--model-path`                                                         | Wan2.2 family                           | None                                  | Currently only compatible with the Ascend NPU family and supports both `w8a8` and `w4a4`                                               |

## Validated ModelOpt Checkpoints

This section is the canonical support matrix for the nine diffusion ModelOpt
checkpoints currently wired up in SGLang docs and validation coverage.

Published checkpoints keep the serialized quantization config as
`quant_method=modelopt`; the FP8 vs NVFP4 split below is a documentation label
derived from `quant_algo`.

Eight of the nine repos live under `lmsys/*`. The FLUX.2 NVFP4 entry keeps the
official `black-forest-labs/FLUX.2-dev-NVFP4` repo.

| Quant Algo | Base Model | Preferred CLI | HF Repo | Current Scope | Notes |
| --- | --- | --- | --- | --- | --- |
| `FP8` | `black-forest-labs/FLUX.1-dev` | `--transformer-path` | `lmsys/flux1-dev-modelopt-fp8-sglang-transformer` | single-transformer override, deterministic latent/image comparison, H100 benchmark, torch-profiler trace | SGLang converter keeps a validated BF16 fallback set for modulation and FF projection layers; use `--model-id FLUX.1-dev` for local mirrors |
| `FP8` | `black-forest-labs/FLUX.2-dev` | `--transformer-path` | `lmsys/flux2-dev-modelopt-fp8-sglang-transformer` | single-transformer override load and generation path | published SGLang-ready transformer override |
| `FP8` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `--transformer-path` | `lmsys/wan22-t2v-a14b-modelopt-fp8-sglang-transformer` | primary `transformer` quantized, `transformer_2` kept BF16 | primary-transformer-only path; keep `transformer_2` on the base checkpoint, and do not describe this as dual-transformer full-model FP8 unless that path is validated separately |
| `FP8` | `hunyuanvideo-community/HunyuanVideo` | `--transformer-path` | `lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer` | single-transformer override, BF16-vs-FP8 video comparison, H100 benchmark, torch-profiler trace | HunyuanVideo uses different ModelOpt/diffusers and SGLang runtime module names; the converter maps those names before writing FP8 scale tensors and BF16 fallback ignores |
| `FP8` | `Qwen/Qwen-Image` | `--transformer-path` | `lmsys/qwen-image-modelopt-fp8-sglang-transformer` | single-transformer override, BF16-vs-FP8 image comparison, H100 benchmark, torch-profiler trace | shares the Qwen Image FP8 fallback preset; keep `img_in`, `txt_in`, timestep embedder, `norm_out.linear`, `proj_out`, `img_mod`/`txt_mod`, and `img_mlp.net.2` in BF16 |
| `FP8` | `Qwen/Qwen-Image-Edit-2511` | `--transformer-path` | `lmsys/qwen-image-edit-modelopt-fp8-sglang-transformer` | TI2I edit path, BF16-vs-FP8 image comparison, H100 benchmark | shares `QwenImageTransformer2DModel` with Qwen Image and uses the same Qwen Image FP8 fallback preset |
| `NVFP4` | `black-forest-labs/FLUX.1-dev` | `--transformer-path` | `lmsys/flux1-dev-modelopt-nvfp4-sglang-transformer` | mixed BF16+NVFP4 transformer override, correctness validation, 4x RTX 5090 benchmark, torch-profiler trace | use `build_modelopt_nvfp4_transformer.py`; validated builder keeps selected FLUX.1 modules in BF16 and sets `swap_weight_nibbles=false` |
| `NVFP4` | `black-forest-labs/FLUX.2-dev` | `--transformer-weights-path` | `black-forest-labs/FLUX.2-dev-NVFP4` | packed-QKV load path | official raw export repo; validated packed export detection and runtime layout handling |
| `NVFP4` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `--transformer-path` | `lmsys/wan22-t2v-a14b-modelopt-nvfp4-sglang-transformer` | primary `transformer` quantized with ModelOpt NVFP4, `transformer_2` kept BF16 | primary-transformer-only path; keep `transformer_2` on the base checkpoint, and current B200/Blackwell bring-up uses `SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn` |

These nine checkpoints are also the intended case set for the B200 diffusion
CI job (`multimodal-gen-test-1-b200`).

## ModelOpt FP8

### Usage Examples

Converted ModelOpt FP8 checkpoints should be loaded as transformer component
overrides. If the repo or local directory already contains `config.json`, use
`--transformer-path`.

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --transformer-path lmsys/flux2-dev-modelopt-fp8-sglang-transformer \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output
```

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --transformer-path lmsys/wan22-t2v-a14b-modelopt-fp8-sglang-transformer \
  --prompt "a fox walking through neon rain" \
  --save-output
```

```bash
sglang generate \
  --model-path hunyuanvideo-community/HunyuanVideo \
  --transformer-path lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer \
  --height 544 --width 960 --num-frames 17 \
  --prompt "A cinematic shot of a red sports car driving through rain at night" \
  --save-output
```

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --transformer-path lmsys/qwen-image-modelopt-fp8-sglang-transformer \
  --prompt "A tiny astronaut reading a book under a glass greenhouse" \
  --save-output
```

```bash
sglang generate \
  --model-path Qwen/Qwen-Image-Edit-2511 \
  --transformer-path lmsys/qwen-image-edit-modelopt-fp8-sglang-transformer \
  --image-path /path/to/input.png \
  --prompt "Turn the scene into a warm watercolor illustration" \
  --save-output
```

### Notes

- `--transformer-path` is the canonical flag for converted ModelOpt FP8
  transformer component repos or directories that already carry `config.json`.
- If the override repo or local directory contains its own `config.json`,
  SGLang reads the quantization config from that override instead of relying on
  the base model config.
- `--transformer-weights-path` still works when you intentionally point at raw
  weight files or a directory that should be metadata-probed as weights first.
- `dit_layerwise_offload` is supported for ModelOpt FP8 checkpoints.
- `dit_cpu_offload` still stays disabled for ModelOpt FP8 checkpoints.
- The layerwise offload path now preserves the non-contiguous FP8 weight stride
  expected by the runtime FP8 GEMM path.
- On disk, the quantization config stays `quant_method=modelopt` with
  `quant_algo=FP8`; the `modelopt-fp8` label in this document is a support
  family name, not a serialized config key.
- `hunyuanvideo-community/HunyuanVideo` uses the `hunyuan-video` converter
  preset. Use `--model-type hunyuan-video` to force it, or rely on
  auto-detection from `_class_name=HunyuanVideoTransformer3DModel`.
- The validated HunyuanVideo FP8 fallback preset keeps `context_embedder`,
  `x_embedder.proj`, timestep/guidance/text embedder linear layers,
  `norm_out.linear`, `proj_out`, double-block modulation linear layers, and
  single-block modulation linear layers in BF16.
- HunyuanVideo ModelOpt exports use diffusers module names that do not match
  SGLang runtime module names for fused QKV and fused QKV+MLP layers. The
  converter maps the names before selecting scale tensors and before writing
  the runtime ignore list.
- `Qwen/Qwen-Image` and `Qwen/Qwen-Image-Edit-2511` share the `qwen-image`
  converter preset. Use `--model-type qwen-image` to force it, or rely on
  auto-detection from `_class_name=QwenImageTransformer2DModel`.
- The validated Qwen Image FP8 fallback preset keeps `img_in`, `txt_in`,
  timestep embedder linear layers, `norm_out.linear`, `proj_out`,
  `transformer_blocks.*.(img_mod|txt_mod)`, and
  `transformer_blocks.*.img_mlp.net.2` in BF16.
- For Qwen Image FP8 conversion, write explicit BF16 fallback tensors before
  honoring ModelOpt ignored weights. Otherwise converter stats can report a
  fallback while the output checkpoint still retains the source FP8 tensor.
- To build the converted checkpoint yourself from a ModelOpt diffusers export,
  use `python -m sglang.multimodal_gen.tools.build_modelopt_fp8_transformer`.

## ModelOpt NVFP4

### Usage Examples

For mixed ModelOpt NVFP4 transformer overrides that already contain
`config.json`, keep the base model and quantized transformer separate and use
`--transformer-path`:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.1-dev \
  --transformer-path lmsys/flux1-dev-modelopt-nvfp4-sglang-transformer \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output
```

For raw NVFP4 exports such as the official FLUX.2 release, use
`--transformer-weights-path`:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --transformer-weights-path black-forest-labs/FLUX.2-dev-NVFP4 \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output
```

SGLang also supports passing the NVFP4 repo or local directory directly as
`--model-path`:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev-NVFP4 \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output
```

For a dual-transformer Wan2.2 export where only the primary `transformer`
was quantized:

```bash
SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn \
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --transformer-path lmsys/wan22-t2v-a14b-modelopt-nvfp4-sglang-transformer \
  --prompt "a fox walking through neon rain" \
  --save-output
```

### Notes

- Use `--transformer-path` for mixed ModelOpt NVFP4 transformer repos or local
  directories that already include `config.json`.
- Use `--transformer-weights-path` for raw NVFP4 exports, individual
  safetensors files, or repo layouts that should be treated as weights first.
- For dual-transformer pipelines such as `Wan2.2-T2V-A14B-Diffusers`, the
  primary `--transformer-path` override targets only `transformer`. Use a
  per-component override such as `--transformer-2-path` only when you
  intentionally want a non-default `transformer_2`.
- On Blackwell, the validated Wan2.2 ModelOpt NVFP4 path currently prefers
  FlashInfer FP4 GEMM via
  `SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=cudnn`.
- This environment-variable override is a current workaround for NVFP4 cases
  where the default sglang JIT/CUTLASS `sm100` path rejects a large-M shape at
  `can_implement()`. The intended long-term fix is to add a validated CUTLASS
  fallback for those shapes rather than rely on the override.
- Direct `--model-path` loading is a compatibility path for FLUX.2 NVFP4-style
  repos or local directories.
- If `--transformer-weights-path` is provided explicitly, it takes precedence
  over the compatibility `--model-path` flow.
- For local directories, SGLang first looks for `*-mixed.safetensors`, then
  falls back to loading from the directory.
- To force the generic diffusion ModelOpt FP4 path onto a specific FlashInfer
  backend, set `SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND`. Supported values
  include `flashinfer_cudnn`, `flashinfer_cutlass`, and `flashinfer_trtllm`.
- On disk, the quantization config stays `quant_method=modelopt` with
  `quant_algo=NVFP4`; the `modelopt-nvfp4` label here is again a documentation
  family name rather than a serialized config key.

## Nunchaku (SVDQuant)

### Install

Install the runtime dependency first:

```bash
pip install nunchaku
```

For platform-specific installation methods and troubleshooting, see the
[Nunchaku installation guide](https://nunchaku.tech/docs/nunchaku/installation/installation.html).

### File Naming and Auto-Detection

For Nunchaku checkpoints, `--model-path` should still point to the original
base model, while `--transformer-weights-path` points to the quantized
transformer weights.

If the basename of `--transformer-weights-path` contains the pattern
`svdq-(int4|fp4)_r{rank}`, SGLang will automatically:
- enable SVDQuant
- infer `--quantization-precision`
- infer `--quantization-rank`

Examples:

| checkpoint name fragment | inferred precision | inferred rank | notes |
|--------------------------|--------------------|---------------|-------|
| `svdq-int4_r32`          | `int4`             | `32`          | Standard INT4 checkpoint |
| `svdq-int4_r128`         | `int4`             | `128`         | Higher-quality INT4 checkpoint |
| `svdq-fp4_r32`           | `nvfp4`            | `32`          | `fp4` in the filename maps to CLI value `nvfp4` |
| `svdq-fp4_r128`          | `nvfp4`            | `128`         | Higher-quality NVFP4 checkpoint |

Common filenames:

| filename | precision | rank | typical use |
|----------|-----------|------|-------------|
| `svdq-int4_r32-qwen-image.safetensors` | `int4` | `32` | Balanced default |
| `svdq-int4_r128-qwen-image.safetensors` | `int4` | `128` | Quality-focused |
| `svdq-fp4_r32-qwen-image.safetensors` | `nvfp4` | `32` | RTX 50-series / NVFP4 path |
| `svdq-fp4_r128-qwen-image.safetensors` | `nvfp4` | `128` | Quality-focused NVFP4 |
| `svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors` | `int4` | `32` | Lightning 4-step |
| `svdq-int4_r128-qwen-image-lightningv1.1-8steps.safetensors` | `int4` | `128` | Lightning 8-step |

If your checkpoint name does not follow this convention, pass
`--enable-svdquant`, `--quantization-precision`, and `--quantization-rank`
explicitly.

### Usage Examples

Recommended auto-detected flow:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --transformer-weights-path /path/to/svdq-int4_r32-qwen-image.safetensors \
  --prompt "a beautiful sunset" \
  --save-output
```

Manual override when the filename does not encode the quant settings:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --transformer-weights-path /path/to/custom_nunchaku_checkpoint.safetensors \
  --enable-svdquant \
  --quantization-precision int4 \
  --quantization-rank 128 \
  --prompt "a beautiful sunset" \
  --save-output
```

### Notes

- `--transformer-weights-path` is the canonical flag for Nunchaku checkpoints.
  Older config names such as `quantized_model_path` are treated as
  compatibility aliases.
- Auto-detection only happens when the checkpoint basename matches
  `svdq-(int4|fp4)_r{rank}`.
- The CLI values are `int4` and `nvfp4`. In filenames, the NVFP4 variant is
  written as `fp4`.
- Lightning checkpoints usually expect matching `--num-inference-steps`, such
  as `4` or `8`.
- Current runtime validation only allows Nunchaku on NVIDIA CUDA Ampere (SM8x)
  or SM12x GPUs. Hopper (SM90) is currently rejected.

## [ModelSlim](https://gitcode.com/Ascend/msmodelslim)
MindStudio-ModelSlim (msModelSlim) is a model offline quantization compression tool launched by MindStudio and optimized for Ascend hardware.

- **Installation**

    ```bash
    # Clone repo and install msmodelslim:
    git clone https://gitcode.com/Ascend/msmodelslim.git
    cd msmodelslim
    bash install.sh
    ```

- **Multimodal_sd quantization**

    Download the original floating-point weights of the large model. Taking Wan2.2-T2V-A14B as an example, you can go to [Wan2.2-T2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B) to obtain the original model weights. Then install other dependencies (related to the model, refer to the modelscope model card).
    > Note: You can find pre-quantized validated models on [modelscope/Eco-Tech](https://modelscope.cn/models/Eco-Tech).

  Run quantization using one-click quantization (recommended):

  ```bash
  msmodelslim quant \
    --model_path /path/to/wan2_2_float_weights \
    --save_path /path/to/wan2_2_quantized_weights \
    --device npu \
    --model_type Wan2_2 \
    --quant_type w8a8 \
    --trust_remote_code True
  ```

  For more detailed examples of quantization of models, as well as information about their support, see the [examples](https://gitcode.com/Ascend/msmodelslim/blob/master/example/multimodal_sd/README.md) section in ModelSLim repo.

  > Note: SGLang does not support quantized embeddings, please disable this option when quantizing using msmodelslim.

- **Auto-Detection and different formats**

    For msmodelslim checkpoints, it's enough to specify only ```--model-path```, the detection of quantization occurs automatically for each layer using parsing of      `quant_model_description.json` config.

    In the case of `Wan2.2` only `Diffusers` weights storage format are supported, whereas modelslim saves the quantized model in the original `Wan2.2` format,
    for conversion in use `python/sglang/multimodal_gen/tools/wan_repack.py` script:

    ```bash
    python wan_repack.py \
      --input-path {path_to_quantized_model} \
      --output-path {path_to_converted_model}
    ```

    After that, please copy all files from original `Diffusers` checkpoint (instead of `transformer`/`tranfsormer_2` folders)

- **Usage Example**

    With auto-detected flow:

    ```bash
    sglang generate \
      --model-path Eco-Tech/Wan2.2-T2V-A14B-Diffusers-w8a8 \
      --prompt "a beautiful sunset" \
      --save-output
    ```

- **Available Quantization Methods**:
    - [x]  ```W4A4_DYNAMIC``` linear with online quantization of activations
    - [x]  ```W8A8``` linear with offline quantization of activations
    - [x]  ```W8A8_DYNAMIC``` linear with online quantization of activations
    - [x]  ```mxfp8``` linear with online/offline MXFP8 quantization (Ascend A5, CANN ≥ 8.0.RC3; see [Ascend NPU quantization](../platforms/ascend/ascend_npu_quantization.md#diffusion-model-quantization-on-ascend-npu))
