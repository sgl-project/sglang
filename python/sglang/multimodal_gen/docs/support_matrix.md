# Compatibility Matrix

The table below shows every supported model and the optimizations supported for them.

The symbols used have the following meanings:

- ✅ = Full compatibility
- ❌ = No compatibility
- ⭕ = Does not apply to this model

## Models x Optimization

The `HuggingFace Model ID` can be passed directly to `from_pretrained()` methods, and sgl-diffusion will use the optimal
default parameters when initializing and generating videos.

### Video Generation Models

| Model Name                   | Hugging Face Model ID                             | Resolutions                                  | TeaCache | Sliding Tile Attn | Sage Attn | Video Sparse Attention (VSA) |
|:-----------------------------|:--------------------------------------------------|:---------------------------------------------|:--------:|:-----------------:|:---------:|:----------------------------:|
| FastWan2.1 T2V 1.3B          | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`         | 480p                                         |    ⭕     |         ⭕         |     ⭕     |              ✅               |
| FastWan2.2 TI2V 5B Full Attn | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 720p                                         |    ⭕     |         ⭕         |     ⭕     |              ✅               |
| Wan2.2 TI2V 5B               | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`                 | 720p                                         |    ⭕     |         ⭕         |     ✅     |              ⭕               |
| Wan2.2 T2V A14B              | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`                | 480p<br>720p                                 |    ❌     |         ❌         |     ✅     |              ⭕               |
| Wan2.2 I2V A14B              | `Wan-AI/Wan2.2-I2V-A14B-Diffusers`                | 480p<br>720p                                 |    ❌     |         ❌         |     ✅     |              ⭕               |
| HunyuanVideo                 | `hunyuanvideo-community/HunyuanVideo`             | 720×1280<br>544×960                          |    ❌     |         ✅         |     ✅     |              ⭕               |
| FastHunyuan                  | `FastVideo/FastHunyuan-diffusers`                 | 720×1280<br>544×960                          |    ❌     |         ✅         |     ✅     |              ⭕               |
| Wan2.1 T2V 1.3B              | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`                | 480p                                         |    ✅     |         ✅         |     ✅     |              ⭕               |
| Wan2.1 T2V 14B               | `Wan-AI/Wan2.1-T2V-14B-Diffusers`                 | 480p, 720p                                   |    ✅     |         ✅         |     ✅     |              ⭕               |
| Wan2.1 I2V 480P              | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`            | 480p                                         |    ✅     |         ✅         |     ✅     |              ⭕               |
| Wan2.1 I2V 720P              | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`            | 720p                                         |    ✅     |         ✅         |     ✅     |              ⭕               |

**Note**: Wan2.2 TI2V 5B has some quality issues when performing I2V generation. We are working on fixing this issue.

### Image Generation Models

| Model Name      | HuggingFace Model ID           | Resolutions    | TeaCache | Sage Attn |
|:----------------|:-------------------------------|:---------------|:--------:|:---------:|
| FLUX.1-dev      | `black-forest-labs/FLUX.1-dev` | Any resolution |    ❌     |     ❌     |
| Qwen Image      | `Qwen/Qwen-Image`              | Any resolution |    ❌     |     ❌     |
| Qwen Image Edit | `Qwen/Qwen-Image-Edit`         | Any resolution |    ❌     |     ❌     |

## Special requirements

### Sliding Tile Attention

- Currently, only Hopper GPUs (H100s) are supported.
