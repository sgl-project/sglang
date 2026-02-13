# Diffusion

SGLang supports two categories of diffusion models for different use cases. This page covers image and video generation; for diffusion LLMs, see [Diffusion LLMs](diffusion_llms.md).

## Image & Video Generation Models

For generating images and videos from text prompts, SGLang supports [many](../diffusion/compatibility_matrix.md) models like:

- **FLUX, Qwen-Image** - High-quality image generation
- **Wan 2.2, HunyuanVideo** - Video generation

```bash
# Example: Launch FLUX for image generation
python3 -m sglang.launch_server \
  --model-path black-forest-labs/FLUX.2-klein-4B \
  --host 0.0.0.0 --port 30000
```

**Full model list:** [Diffusion Models](../diffusion/compatibility_matrix.md)
