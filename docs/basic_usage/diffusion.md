# Diffusion

SGLang supports two categories of diffusion models for different use cases.

## Image & Video Generation Models

For generating images and videos from text prompts, SGLang supports [many](../supported_models/image_generation/diffusion_models.md#image-generation-models) models like:

- **FLUX, Qwen-Image** - High-quality image generation
- **Wan 2.2, HunyuanVideo** - Video generation

```bash
# Example: Launch FLUX for image generation
python3 -m sglang.launch_server \
  --model-path black-forest-labs/FLUX.2-klein-4B \
  --host 0.0.0.0 --port 30000
```

**Full model list:** [Diffusion Models](../supported_models/image_generation/diffusion_models.md)

---

## Diffusion Language Models (dLLMs)

These are text-generation models that use diffusion (denoising) instead of autoregressive decoding:

- **LLaDA** - Large Language Diffusion with mAsking

```bash
# Example: Launch LLaDA for text generation
python3 -m sglang.launch_server \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --host 0.0.0.0 --port 30000
```

**Full model list:** [Diffusion Language Models](../supported_models/text_generation/diffusion_language_models.md)
