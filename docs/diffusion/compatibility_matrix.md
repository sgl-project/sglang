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

| Model Name                   | Hugging Face Model ID                             | Resolutions         | TeaCache | Sliding Tile Attn | Sage Attn | Video Sparse Attention (VSA) | Sparse Linear Attention (SLA) | Sage Sparse Linear Attention (SageSLA) | Sparse Video Gen 2 (SVG2) |
|:-----------------------------|:--------------------------------------------------|:--------------------|:--------:|:-----------------:|:---------:|:----------------------------:|:----------------------------:|:-----------------------------------------------:|:----------------------------------:|
| FastWan2.1 T2V 1.3B          | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`         | 480p                |    ⭕     |         ⭕         |      ⭕     |              ✅               |              ❌               |              ❌               |    ❌     |
| FastWan2.2 TI2V 5B Full Attn | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 720p                |    ⭕     |         ⭕         |     ⭕     |              ✅               |              ❌               |              ❌               |    ❌     |
| Wan2.2 TI2V 5B               | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`                 | 720p                |    ⭕     |         ⭕         |     ✅     |              ⭕               |              ❌               |              ❌               |    ❌     |
| Wan2.2 T2V A14B              | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`                | 480p<br>720p        |    ❌     |         ❌         |     ✅     |              ⭕               |              ❌               |              ❌               |    ❌     |
| Wan2.2 I2V A14B              | `Wan-AI/Wan2.2-I2V-A14B-Diffusers`                | 480p<br>720p        |    ❌     |         ❌         |     ✅     |              ⭕               |              ❌               |              ❌               |    ❌     |
| HunyuanVideo                 | `hunyuanvideo-community/HunyuanVideo`             | 720×1280<br>544×960 |    ❌     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| FastHunyuan                  | `FastVideo/FastHunyuan-diffusers`                 | 720×1280<br>544×960 |    ❌     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| Wan2.1 T2V 1.3B              | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`                | 480p                |    ✅     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| Wan2.1 T2V 14B               | `Wan-AI/Wan2.1-T2V-14B-Diffusers`                 | 480p, 720p          |    ✅     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| Wan2.1 I2V 480P              | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`            | 480p                |    ✅     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| Wan2.1 I2V 720P              | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`            | 720p                |    ✅     |         ✅         |     ✅     |              ⭕               |              ❌               |              ❌               |    ✅     |
| TurboWan2.1 T2V 1.3B         | `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers`      | 480p                |    ✅     |         ❌         |     ❌     |              ❌               |              ✅               |              ✅               |    ⭕     |
| TurboWan2.1 T2V 14B          | `IPostYellow/TurboWan2.1-T2V-14B-Diffusers`       | 480p                |    ✅     |         ❌         |     ❌     |              ❌               |              ✅               |              ✅               |    ⭕     |
| TurboWan2.1 T2V 14B 720P     | `IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers`  | 720p                |    ✅     |         ❌         |     ❌     |              ❌               |              ✅               |              ✅               |    ⭕     |
| TurboWan2.2 I2V A14B         | `IPostYellow/TurboWan2.2-I2V-A14B-Diffusers`      | 720p                |    ✅     |         ❌         |     ❌     |              ❌               |              ✅               |              ✅               |    ⭕     |
| LTX-2                        | `Lightricks/LTX-2`                                | 768×512<br>1536×1024 |    ❌     |         ❌         |     ❌     |              ❌               |              ❌               |              ❌               |    ❌     |
| LTX-2.3                      | `Lightricks/LTX-2.3`                              | 768×512<br>1536×1024 |    ❌     |         ❌         |     ❌     |              ❌               |              ❌               |              ❌               |    ❌     |

**Note**:
1. Wan2.2 TI2V 5B has some quality issues when performing I2V generation. We are working on fixing this issue.
2. SageSLA is based on SpargeAttn. Install it first with `pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation`
3. LTX-2 two-stage generation uses `--pipeline-class-name LTX2TwoStagePipeline`. The spatial upsampler and distilled LoRA are auto-resolved from the model snapshot by default, and can still be overridden with `--spatial-upsampler-path` and `--distilled-lora-path`.
4. `Lightricks/LTX-2.3` is supported through the bundled native overlay materialization path. One-stage generation uses the default `LTX2Pipeline`; two-stage generation uses `--pipeline-class-name LTX2TwoStagePipeline`.
5. For LTX models, the `Resolutions` column uses output video `width×height` semantics, matching `sglang generate --width ... --height ...`. One-stage generation is validated at `768×512`; two-stage generation is validated at `1536×1024`.

### Image Generation Models

| Model Name           | HuggingFace Model ID                |
|:---------------------|:------------------------------------|
| FLUX.1-dev           | `black-forest-labs/FLUX.1-dev`      |
| FLUX.2-dev           | `black-forest-labs/FLUX.2-dev`      |
| FLUX.2-Klein         | `black-forest-labs/FLUX.2-klein-4B` |
| Z-Image-Turbo        | `Tongyi-MAI/Z-Image-Turbo`          |
| GLM-Image            | `zai-org/GLM-Image`                 |
| Qwen Image           | `Qwen/Qwen-Image`                   |
| Qwen Image 2512      | `Qwen/Qwen-Image-2512`              |
| Qwen Image Edit      | `Qwen/Qwen-Image-Edit`              |
| Qwen Image Edit 2511 | `Qwen/Qwen-Image-Edit-2511`         |

## Verified LoRA Examples

This section lists example LoRAs that have been explicitly tested and verified with each base model in the **SGLang Diffusion** pipeline.

> Important:
> LoRAs that are not listed here are not necessarily incompatible.
> In practice, most standard LoRAs are expected to work, especially those following common Diffusers or SD-style conventions.
> The entries below simply reflect configurations that have been manually validated by the SGLang team.

### Verified LoRAs by Base Model

| Base Model       | Supported LoRAs |
|:-----------------|:----------------|
| Wan2.2           | `lightx2v/Wan2.2-Distill-Loras`<br>`Cseti/wan2.2-14B-Arcane_Jinx-lora-v1` |
| Wan2.1           | `lightx2v/Wan2.1-Distill-Loras` |
| Z-Image-Turbo    | `tarn59/pixel_art_style_lora_z_image_turbo`<br>`wcde/Z-Image-Turbo-DeJPEG-Lora` |
| Qwen-Image       | `lightx2v/Qwen-Image-Lightning`<br>`flymy-ai/qwen-image-realism-lora`<br>`prithivMLmods/Qwen-Image-HeadshotX`<br>`starsfriday/Qwen-Image-EVA-LoRA` |
| Qwen-Image-Edit  | `ostris/qwen_image_edit_inpainting`<br>`lightx2v/Qwen-Image-Edit-2511-Lightning` |
| Flux             | `dvyio/flux-lora-simple-illustration`<br>`XLabs-AI/flux-furry-lora`<br>`XLabs-AI/flux-RealismLora` |

## Special requirements

### Sliding Tile Attention

- Currently, only Hopper GPUs (H100s) are supported.
