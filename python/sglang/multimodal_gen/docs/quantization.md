# Quantization

This document introduces the model quantization schemes supported in SGLang and how to use them to reduce memory usage and accelerate inference.

## Nunchaku (SVDQuant)

### Introduction

**SVDQuant** is a Post-Training Quantization (PTQ) technique for diffusion models that quantizes model weights and activations to 4-bit precision (W4A4) while maintaining high visual quality. This method uses Singular Value Decomposition (SVD) to decompose the weight matrix into low-rank components and residuals, effectively absorbing outliers in activations, making 4-bit quantization possible.

**Nunchaku** is a high-performance inference engine that implements SVDQuant, optimized for low-bit neural networks. It is not Quantization-Aware Training (QAT), but directly quantizes pre-trained models.

Paper: [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007) (ICLR 2025 Spotlight)

### Key Features

SVDQuant significantly reduces memory usage and accelerates inference while maintaining visual quality:

- **Memory Optimization**: Reduces memory usage by **3.6×** compared to BF16 models.
- **Inference Acceleration**:
  - **3.0×** faster than the NF4 (W4A16) baseline on desktop/laptop RTX 4090 GPUs.
  - **8.7×** speedup on laptop RTX 4090 by eliminating CPU offloading compared to 16-bit models.
  - **3.1×** faster than BF16 and NF4 models on RTX 5090 GPUs with NVFP4.

### Supported Precisions

Nunchaku supports two quantization precisions:

- **INT4**: Standard INT4 quantization, supported on NVIDIA GPUs with Compute Capability 7.0+ (RTX 20 series and above).
- **NVFP4**: FP4 quantization, providing better image quality on newer cards like the RTX 5090.

### Usage

#### 1. Install Nunchaku

```bash
pip install nunchaku
```

For more installation information, please refer to the [Nunchaku Official Documentation](https://nunchaku.tech/docs/nunchaku/installation/installation.html).

#### 2. Download Quantized Models

Nunchaku provides pre-quantized model weights available on Hugging Face:

- [nunchaku-ai/nunchaku-qwen-image](https://huggingface.co/nunchaku-ai/nunchaku-qwen-image)
- [nunchaku-ai/nunchaku-flux](https://huggingface.co/nunchaku-ai/nunchaku-flux)

Taking Qwen-Image as an example, several quantized models with different configurations are provided:

| Filename | Precision | Rank | Usage |
|----------|-----------|------|-------|
| `svdq-int4_r32-qwen-image.safetensors` | INT4 | 32 | Standard Version |
| `svdq-int4_r128-qwen-image.safetensors` | INT4 | 128 | High-Quality Version |
| `svdq-fp4_r32-qwen-image.safetensors` | NVFP4 | 32 | RTX 5090 Standard Version |
| `svdq-fp4_r128-qwen-image.safetensors` | NVFP4 | 128 | RTX 5090 High-Quality Version |
| `svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors` | INT4 | 32 | Lightning 4-Step Version |
| `svdq-int4_r128-qwen-image-lightningv1.1-8steps.safetensors` | INT4 | 128 | Lightning 8-Step Version |

> **Note**: Higher Rank usually means better image quality, but with slightly increased memory usage and computation.

#### 3. Run Quantized Models

SGLang features **smart auto-detection** for Nunchaku models. In most cases, you only need to provide the path to the quantized weights, and the precision and rank will be automatically inferred from the filename.

**Simplified Command (Recommended):**

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "change the raccoon to a cute cat" \
  --save-output \
  --transformer-weights-path /path/to/svdq-int4_r32-qwen-image.safetensors
```

**Manual Override (If needed):**

If your filename doesn't follow the standard naming convention, or you want to force specific settings:

- `--enable-svdquant`: Manually enable SVDQuant.
- `--quantization-precision`: Set to `int4` or `nvfp4`.
- `--quantization-rank`: Set the SVD rank (e.g., 32, 128).
- `--quantization-act-unsigned` (Optional): Use unsigned activation quantization.

Example with manual overrides:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "a beautiful sunset" \
  --enable-svdquant \
  --transformer-weights-path /path/to/custom_model.safetensors \
  --quantization-precision int4 \
  --quantization-rank 128
```

#### 4. Configuration Recommendations

Choose the appropriate configuration based on your hardware and requirements:

| Scenario | Recommended Config | Description |
|----------|-------------------|-------------|
| Standard Use (20/30/40 Series GPU) | INT4 + Rank 32 | Balanced performance and quality |
| Quality Focus (Sufficient VRAM) | INT4 + Rank 128 | Better image quality |
| RTX 5090 Standard Use | NVFP4 + Rank 32 | Utilizes FP4 hardware acceleration |
| RTX 5090 Quality Focus | NVFP4 + Rank 128 | Best image quality |
| Fast Prototyping/Preview | Lightning 4-Step Version | Extremely fast generation, slightly reduced quality |

### Notes

1.  Model Path Correspondence: `--model-path` should point to the original non-quantized model (for loading config and tokenizer, etc.), while `--transformer-weights-path` points to the quantized weight file / folder / Huggingface Repo ID.

2.  Auto-Detection Requirements: For auto-detection to work, the filename must contain the pattern `svdq-{precision}_r{rank}` (e.g., `svdq-int4_r32`).

3.  GPU Compatibility:
    -   INT4: Supports NVIDIA GPUs with Compute Capability 7.0+ (RTX 20 series and above).
    -   NVFP4: Optimized mainly for newer cards like the RTX 50 series that support FP4.

4.  Lightning Models: When using Lightning versions, adjust `--num-inference-steps` accordingly (usually 4 or 8 steps).

### Custom Model Quantization

If you want to quantize your own models, you can use the [DeepCompressor](https://github.com/mit-han-lab/deepcompressor) tool. For detailed instructions, please refer to the Nunchaku official documentation.

## Quantization

### Usage

#### Option 1: Pre-quantized folder (has `config.json`)

For quantized checkpoints that include a `config.json` with a `quantization_config` field (e.g., models converted via `convert_hf_to_fp8.py`), where the transformer's `config.json` already encodes the `quantization_config`, use the component override:

```bash
sglang generate \
  --model-path /path/to/FLUX.1-dev \
  --transformer-path /path/to/FLUX.1-dev/transformer-FP8 \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output
```


If you need to convert a model to FP8 format yourself, use the provided conversion script:

```bash
# convert transformer to FP8 with block quantization
python -m sglang.multimodal_gen.tools.convert_hf_to_fp8 \
  --model-dir /path/to/FLUX.1-dev/transformer \
  --save-dir /path/to/FLUX.1-dev/transformer-FP8 \
  --strategy block \
  --block-size 128 128
```

#### Option 2: Pre-quantized single-file checkpoint (no `config.json`)



Some providers (e.g., [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8)) distribute a single `.safetensors` file without a companion `config.json`. Use `--transformer-weights-path` to point to this file (or HuggingFace repo ID) while keeping `--model-path` for the base model:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-klein-9B \
  --transformer-weights-path black-forest-labs/FLUX.2-klein-9b-fp8 \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --save-output
```

SGLang-Diffusion will automatically read the `quantization_config` metadata embedded in the safetensors file header (if present). For the quant config to be auto-detected, the file's metadata must contain a JSON-encoded `quantization_config` key with at least a `quant_method` field (e.g. `"fp8"`).

Note: this feature is a WIP
