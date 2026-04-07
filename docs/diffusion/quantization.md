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

| quant_family     | checkpoint form                                                                            | canonical CLI                                        | supported models                                             | extra dependency                      | platform / notes                                                                                                      |
|------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `fp8`            | Quantized transformer component folder, or safetensors with `quantization_config` metadata | `--transformer-path` or `--transformer-weights-path` | ALL                                                          | None                                  | Component-folder and single-file flows are both supported                                                             |
| `nvfp4-modelopt` | NVFP4 safetensors file, sharded directory, or repo providing transformer weights           | `--transformer-weights-path`                         | FLUX.2                                                       | `comfy-kitchen` optional on Blackwell | Blackwell can use a best-performance kit when available; otherwise SGLang falls back to the generic ModelOpt FP4 path |
| `nunchaku-svdq`  | Pre-quantized Nunchaku transformer weights, usually named `svdq-{int4\|fp4}_r{rank}-...`   | `--transformer-weights-path`                         | Model-specific support such as Qwen-Image, FLUX, and Z-Image | `nunchaku`                            | SGLang can infer precision and rank from the filename and supports both `int4` and `nvfp4`                            |
| `msmodelslim`    | Pre-quantized msmodelslim transformer weights                                              | `--model-path`                                       | Wan2.2 family                                                | None                                  | Currently only compatible with the Ascend NPU family and supports both `w8a8` and `w4a4`                              |

## NVFP4

### Usage Examples

Recommended usage keeps the base model and quantized transformer override
separate:

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

### Notes

- `--transformer-weights-path` is still the canonical CLI for NVFP4
  transformer checkpoints.
- Direct `--model-path` loading is a compatibility path for FLUX.2 NVFP4-style
  repos or local directories.
- If `--transformer-weights-path` is provided explicitly, it takes precedence
  over the compatibility `--model-path` flow.
- For local directories, SGLang first looks for `*-mixed.safetensors`, then
  falls back to loading from the directory.
- On Blackwell, `comfy-kitchen` can provide the best-performance path when
  available; otherwise SGLang falls back to the generic ModelOpt FP4 path.

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
    - [ ]  ```mxfp8``` linear in progress
