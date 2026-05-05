# Compatibility Matrix

The table below shows every supported model and the optimizations supported for them.

The symbols used have the following meanings:

- ✅ = Full compatibility
- ❌ = No compatibility
- ⭕ = Does not apply to this model

## Models x Optimization

The `HuggingFace Model ID` can be passed directly to `from_pretrained()` methods, and sglang-diffusion will use the
optimal
default parameters when initializing and generating videos.

### Video Generation Models

| Model Name                   | Hugging Face Model ID                             | Resolutions          | TeaCache | Sliding Tile Attn | Sage Attn | Video Sparse Attention (VSA) | Sparse Linear Attention (SLA) | Sage Sparse Linear Attention (SageSLA) | Sparse Video Gen 2 (SVG2) |
|:-----------------------------|:--------------------------------------------------|:---------------------|:--------:|:-----------------:|:---------:|:----------------------------:|:-----------------------------:|:--------------------------------------:|:-------------------------:|
| FastWan2.1 T2V 1.3B          | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`         | 480p                 |    ⭕     |         ⭕         |     ⭕     |              ✅               |               ❌               |                   ❌                    |             ❌             |
| FastWan2.2 TI2V 5B Full Attn | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 720p                 |    ⭕     |         ⭕         |     ⭕     |              ✅               |               ❌               |                   ❌                    |             ❌             |
| Wan2.2 TI2V 5B               | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`                 | 720p                 |    ⭕     |         ⭕         |     ✅     |              ⭕               |               ❌               |                   ❌                    |             ❌             |
| Wan2.2 T2V A14B              | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`                | 480p<br>720p         |    ❌     |         ❌         |     ✅     |              ⭕               |               ❌               |                   ❌                    |             ❌             |
| Wan2.2 I2V A14B              | `Wan-AI/Wan2.2-I2V-A14B-Diffusers`                | 480p<br>720p         |    ❌     |         ❌         |     ✅     |              ⭕               |               ❌               |                   ❌                    |             ❌             |
| HunyuanVideo                 | `hunyuanvideo-community/HunyuanVideo`             | 720×1280<br>544×960  |    ❌     |         ✅         |     ✅     |              ⭕               |               ❌               |                   ❌                    |             ✅             |
| FastHunyuan                  | `FastVideo/FastHunyuan-diffusers`                 | 720×1280<br>544×960  |    ❌     |         ✅         |     ✅     |              ⭕               |               ❌               |                   ❌                    |             ✅             |
| Wan2.1 T2V 1.3B              | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`                | 480p                 |    ✅     |         ✅         |     ✅     |              ⭕               |               ❌               |                   ❌                    |             ✅             |
| Wan2.1 T2V 14B               | `Wan-AI/Wan2.1-T2V-14B-Diffusers`                 | 480p, 720p           |    ✅     |         ✅         |     ✅     |              ⭕               |               ❌               |                   ❌                    |             ✅             |
| Wan2.1 I2V 480P              | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`            | 480p                 |    ✅     |         ✅         |     ✅     |              ⭕               |               ❌               |                   ❌                    |             ✅             |
| Wan2.1 I2V 720P              | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`            | 720p                 |    ✅     |         ✅         |     ✅     |              ⭕               |               ❌               |                   ❌                    |             ✅             |
| TurboWan2.1 T2V 1.3B         | `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers`      | 480p                 |    ✅     |         ❌         |     ❌     |              ❌               |               ✅               |                   ✅                    |             ⭕             |
| TurboWan2.1 T2V 14B          | `IPostYellow/TurboWan2.1-T2V-14B-Diffusers`       | 480p                 |    ✅     |         ❌         |     ❌     |              ❌               |               ✅               |                   ✅                    |             ⭕             |
| TurboWan2.1 T2V 14B 720P     | `IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers`  | 720p                 |    ✅     |         ❌         |     ❌     |              ❌               |               ✅               |                   ✅                    |             ⭕             |
| TurboWan2.2 I2V A14B         | `IPostYellow/TurboWan2.2-I2V-A14B-Diffusers`      | 720p                 |    ✅     |         ❌         |     ❌     |              ❌               |               ✅               |                   ✅                    |             ⭕             |
| Wan2.1 Fun 1.3B InP           | `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers`          | 480p                 |    ✅     |         ✅         |     ✅     |              ⭕               |               ❌               |                   ❌                    |             ✅             |
| Helios Base                   | `BestWishYsh/Helios-Base`                          | 720p                 |    ❌     |         ❌         |     ❌     |              ❌               |               ❌               |                   ❌                    |             ❌             |
| Helios Mid                    | `BestWishYsh/Helios-Mid`                           | 720p                 |    ❌     |         ❌         |     ❌     |              ❌               |               ❌               |                   ❌                    |             ❌             |
| Helios Distilled              | `BestWishYsh/Helios-Distilled`                     | 720p                 |    ❌     |         ❌         |     ❌     |              ❌               |               ❌               |                   ❌                    |             ❌             |
| LTX-2 (one and two stages)   | `Lightricks/LTX-2`                                | 768×512<br>1536×1024 |    ❌     |         ❌         |     ❌     |              ❌               |               ❌               |                   ❌                    |             ❌             |
| LTX-2.3 (one and two stages) | `Lightricks/LTX-2.3`                              | 768×512<br>1536×1024 |    ❌     |         ❌         |     ❌     |              ❌               |               ❌               |                   ❌                    |             ❌             |

**Note**:

1. Wan2.2 TI2V 5B has some quality issues when performing I2V generation. We are working on fixing this issue.
2. SageSLA is based on SpargeAttn. Install it first with `pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation`
3. LTX-2 and LTX-2.3 two-stage generation uses `--pipeline-class-name LTX2TwoStagePipeline`. The spatial upsampler and distilled LoRA are auto-resolved from the model snapshot by default, and can still be overridden with `--spatial-upsampler-path` and `--distilled-lora-path`.
  - For LTX models, the `Resolutions` column uses output video `width×height` semantics, matching `sglang generate --width ... --height ...`.

### Image Generation Models

| Model Name                | HuggingFace Model ID                                     |
|:--------------------------|:---------------------------------------------------------|
| FLUX.1-dev                | `black-forest-labs/FLUX.1-dev`                           |
| FLUX.2-dev                | `black-forest-labs/FLUX.2-dev`                           |
| FLUX.2-dev-NVFP4          | `black-forest-labs/FLUX.2-dev-NVFP4`                     |
| FLUX.2-Klein-4B           | `black-forest-labs/FLUX.2-klein-4B`                      |
| FLUX.2-Klein-9B           | `black-forest-labs/FLUX.2-klein-9B`                      |
| Z-Image                   | `Tongyi-MAI/Z-Image`                                    |
| Z-Image-Turbo             | `Tongyi-MAI/Z-Image-Turbo`                              |
| GLM-Image                 | `zai-org/GLM-Image`                                     |
| Qwen Image                | `Qwen/Qwen-Image`                                       |
| Qwen Image 2512           | `Qwen/Qwen-Image-2512`                                  |
| Qwen Image Edit           | `Qwen/Qwen-Image-Edit`                                  |
| Qwen Image Edit 2509      | `Qwen/Qwen-Image-Edit-2509`                             |
| Qwen Image Edit 2511      | `Qwen/Qwen-Image-Edit-2511`                             |
| Qwen Image Layered        | `Qwen/Qwen-Image-Layered`                               |
| SD3 Medium                | `stabilityai/stable-diffusion-3-medium-diffusers`        |
| SD3.5 Medium              | `stabilityai/stable-diffusion-3.5-medium-diffusers`      |
| SD3.5 Large               | `stabilityai/stable-diffusion-3.5-large-diffusers`       |
| Hunyuan3D-2               | `tencent/Hunyuan3D-2`                                    |
| SANA 1.5 1.6B             | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers`   |
| SANA 1.5 4.8B             | `Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers`   |
| SANA 1600M 1024px         | `Efficient-Large-Model/Sana_1600M_1024px_diffusers`     |
| SANA 600M 1024px          | `Efficient-Large-Model/Sana_600M_1024px_diffusers`      |
| SANA 1600M 512px          | `Efficient-Large-Model/Sana_1600M_512px_diffusers`      |
| SANA 600M 512px           | `Efficient-Large-Model/Sana_600M_512px_diffusers`       |
| FireRed-Image-Edit 1.0    | `FireRedTeam/FireRed-Image-Edit-1.0`                     |
| FireRed-Image-Edit 1.1    | `FireRedTeam/FireRed-Image-Edit-1.1`                     |
| ERNIE-Image               | `baidu/ERNIE-Image`                                      |
| ERNIE-Image-Turbo         | `baidu/ERNIE-Image-Turbo`                                |

## Supported Components

SGLang Diffusion supports overriding individual pipeline components with
`--<component>-path`. The value can be either a Hugging Face repo ID or a local
component directory.

The same overrides can also be provided in config files through
`component_paths.<component>`.

### Common Syntax

CLI:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --vae-path black-forest-labs/FLUX.2-small-decoder \
  --transformer-path /models/flux2/transformer
```

Config file:

```yaml
model_path: black-forest-labs/FLUX.2-dev
component_paths:
  vae: black-forest-labs/FLUX.2-small-decoder
  transformer: /models/flux2/transformer
```

Use the component name from the pipeline's `model_index.json` or the native pipeline's registered module name:

| Component Type    | Supported Keys                                                                                                             | Notes                                                         |
|:------------------|:---------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| VAE               | `vae`, `video_vae`, `audio_vae`                                                                                            | `vae` is the common image-generation override                 |
| Transformer / DiT | `transformer`, `video_dit`, `audio_dit`                                                                                    | `transformer` is the standard override for the main denoiser  |
| Text / Preprocess | `text_encoder`, `text_encoder_2`, `tokenizer`, `processor`, `image_processor`                                              | Replacement encoders often need matching preprocessing assets |
| Auxiliary         | `scheduler`, `spatial_upsampler`, `vocoder`, `connectors`, `dual_tower_bridge`, `image_encoder`, `vision_language_encoder` | Only valid for pipelines that expose these components         |

### Known Component Repos

The table below lists concrete Hugging Face component repos that are already used in SGLang Diffusion docs or tests. It is not an exhaustive catalog of all compatible component repos.

| Base Model                     | Override Key  | Example Repo                             | Notes                                     |
|:-------------------------------|:--------------|:-----------------------------------------|:------------------------------------------|
| `black-forest-labs/FLUX.2-dev` | `vae`         | `black-forest-labs/FLUX.2-small-decoder` | Decoder-only FLUX.2 VAE override          |
| `black-forest-labs/FLUX.2-dev` | `vae`         | `fal/FLUX.2-Tiny-AutoEncoder`            | Existing tested custom VAE path           |

### VAE

- `--vae-path` is the common image-generation override.
- `--video-vae-path` and `--audio-vae-path` are only relevant for pipelines with separate video or audio VAEs.

### Transformer / DiT

- `--transformer-path` is the standard override for the main denoising transformer.
- For quantized transformers, prefer `--transformer-path` or `--transformer-weights-path`; see `quantization.md`.
- `--video-dit-path` and `--audio-dit-path` are only for pipelines that split denoisers by modality.

### Text Encoders and Preprocessors

- `--text-encoder-path` and `--text-encoder-2-path` override primary and secondary text encoders.
- `--tokenizer-path`, `--processor-path`, and `--image-processor-path` are useful when the replacement encoder requires matching preprocessing assets.

### Auxiliary Components

- `--scheduler-path` is only relevant when the pipeline exposes a scheduler component.
- `--spatial-upsampler-path` is mainly for two-stage pipelines such as `LTX2TwoStagePipeline`.
- `--vocoder-path`, `--connectors-path`, `--dual-tower-bridge-path`, `--image-encoder-path`, and `--vision-language-encoder-path` are only valid for pipelines that expose those components.

### Notes

1. Component overrides are only valid when the target pipeline actually uses
   that component.
2. The override key should match the component name in the pipeline's
   `model_index.json` or the native pipeline's registered module name.

## Verified LoRA Examples

This section lists example LoRAs that have been explicitly tested and verified with each base model in the **SGLang Diffusion** pipeline.

> Important:
> LoRAs that are not listed here are not necessarily incompatible.
> In practice, most standard LoRAs are expected to work, especially those following common Diffusers or SD-style conventions.
> The entries below simply reflect configurations that have been manually validated by the SGLang team.

### Verified LoRAs by Base Model

| Base Model      | Supported LoRAs                                                                                                                                    |
|:----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| Wan2.2          | `lightx2v/Wan2.2-Distill-Loras`<br>`Cseti/wan2.2-14B-Arcane_Jinx-lora-v1`                                                                          |
| Wan2.1          | `lightx2v/Wan2.1-Distill-Loras`                                                                                                                    |
| Z-Image-Turbo   | `tarn59/pixel_art_style_lora_z_image_turbo`<br>`wcde/Z-Image-Turbo-DeJPEG-Lora`                                                                    |
| Qwen-Image      | `lightx2v/Qwen-Image-Lightning`<br>`flymy-ai/qwen-image-realism-lora`<br>`prithivMLmods/Qwen-Image-HeadshotX`<br>`starsfriday/Qwen-Image-EVA-LoRA` |
| Qwen-Image-Edit | `ostris/qwen_image_edit_inpainting`<br>`lightx2v/Qwen-Image-Edit-2511-Lightning`                                                                   |
| Flux            | `dvyio/flux-lora-simple-illustration`<br>`XLabs-AI/flux-furry-lora`<br>`XLabs-AI/flux-RealismLora`                                                 |

## Special requirements

### Sliding Tile Attention

- Currently, only Hopper GPUs (H100s) are supported.
