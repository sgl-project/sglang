# Diffusion Language Models (dLLMs)

These are text-generation models that use diffusion (denoising) instead of autoregressive decoding:

- **LLaDA** - Large Language Diffusion with mAsking

```bash
# Example: Launch LLaDA for text generation
python3 -m sglang.launch_server \
  --model-path GSAI-ML/LLaDA-8B-Instruct \
  --host 0.0.0.0 --port 30000
```

**Full model list:** [Diffusion Language Models](../supported_models/text_generation/diffusion_language_models.md)
