# Diffusion Language Models

Diffusion language models have shown promise for non-autoregressive text generation with parallel decoding capabilities. Unlike auto-regressive language models, different diffusion language models require different decoding strategies.

## Example Launch Command

SGLang supports different DLLM algorithms such as `LowConfidence` and `JointThreshold`.

```shell
python3 -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \ # example HF/local path
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config ./config.yaml \ # Optional. Uses the algorithm's default if not set.
  --host 0.0.0.0 \
  --port 30000
```

## Example Configuration File

Depending on the algorithm selected, the configuration parameters vary.

LowConfidence Config:

```yaml
# Confidence threshold for accepting predicted tokens
# - Higher values: More conservative, better quality but slower
# - Lower values: More aggressive, faster but potentially lower quality
# Range: 0.0 - 1.0
threshold: 0.95

# Default: 32, for LLaDA2MoeModelLM
block_size: 32
```

JointThreshold Config:

```yaml
# Decoding threshold for Mask-to-Token (M2T) phase
# - Higher values: More conservative, better quality but slower
# - Lower values: More aggressive, faster but potentially lower quality
# Range: 0.0 - 1.0
threshold: 0.5
# Decoding threshold for Token-to-Token (T2T) phase
# Range: 0.0 - 1.0
# Setting to 0.0 allows full editing (recommended for most cases).
edit_threshold: 0.0
# Max extra T2T steps after all masks are removed. Prevents infinite loops.
max_post_edit_steps: 16
# 2-gram repetition penalty (default 0).
# An empirical value of 3 is often sufficient to mitigate most repetitions.
penalty_lambda: 0
```

## DiffusionGemma (uniform-state diffusion, `Gemma4Renoise`)

DiffusionGemma is a uniform-state (renoising) block-diffusion model with no mask
token. Each block is a fixed-length canvas of `canvas_length` tokens that the
`Gemma4Renoise` sampler denoises over
`max_denoising_steps` reverse steps, feeding the previous step's logits back as
self-conditioning.

The required runtime settings are applied automatically for `Gemma4Renoise` (the triton
attention backend, eager mode, and unchunked prefill, needed because the full-attention
head_dim is 512 and the canvas uses bidirectional attention), so a default launch works:

```shell
sglang serve \
  --model-path google/diffusiongemma-26B-A4B-it \
  --dllm-algorithm Gemma4Renoise \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000
```

Gemma4Renoise Config (defaults follow the checkpoint's `generation_config.json`):

```yaml
# Number of reverse denoising steps per canvas.
max_denoising_steps: 48
# Optional. Makes the renoise sampling reproducible (also shared across TP ranks).
seed: 1234
sampler_config:
  # Entropy budget. Accept the lowest-entropy canvas positions within this bound each step (the rest are re-noised).
  entropy_bound: 0.1
# Linear temperature schedule applied over the denoising steps.
temperature_schedule:
  t_min: 0.4
  t_max: 0.8
# Stop early once the canvas is stable and confident.
stopping_config:
  confidence_threshold: 0.005
  stability_threshold: 1
```

Notes and current limitations:

- Sampling is governed by the renoise schedule, so request-level `logprobs`, penalties,
  `logit_bias`, and grammar / structured output (`json_schema` / `regex` / `ebnf` / `structural_tag`) are not
  applied and are rejected with a 400. Core sampling controls (`temperature`, `top_k`,
  `top_p`) are accepted but have no effect.
- Streaming is block-level: one fully-denoised canvas (`canvas_length` tokens) per chunk.
- `max_tokens` truncates the returned text, but the canvas is always fully denoised.

## Example Client Code Snippet

Just like other supported models, diffusion language models can be used via the REST API or Python client.

Python client example for making a generation request to the launched server:

```python
import sglang as sgl

def main():
    llm = sgl.Engine(model_path="inclusionAI/LLaDA2.0-mini",
                     dllm_algorithm="LowConfidence",
                     max_running_requests=1,
                     trust_remote_code=True)

    prompts = [
        "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write a brief introduction of the great wall <|role_end|><role>ASSISTANT</role>"
    ]

    sampling_params = {
        "temperature": 0,
        "max_new_tokens": 1024,
    }

    outputs = llm.generate(prompts, sampling_params)
    print(outputs)

if __name__ == '__main__':
    main()
```

Curl example for making a generation request to the launched server:

```bash
curl -X POST "http://127.0.0.1:30000/generate" \
     -H "Content-Type: application/json" \
     -d '{
        "text": [
            "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write the number from 1 to 128 <|role_end|><role>ASSISTANT</role>",
            "<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role> Write a brief introduction of the great wall <|role_end|><role>ASSISTANT</role>"
        ],
        "stream": true,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1024
        }
    }'
```

## Supported Models

Below the supported models are summarized in a table.

| Model Family               | Example Model                | Description                                                                                          |
| -------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| **LLaDA2.0 (mini, flash)** | `inclusionAI/LLaDA2.0-flash` | LLaDA2.0-flash is a diffusion language model featuring a 100B Mixture-of-Experts (MoE) architecture. |
| **SDAR (JetLM)**           | `JetLM/SDAR-8B-Chat`         | SDAR series diffusion language model (Chat), dense architecture.                                 |
| **SDAR (JetLM)**           | `JetLM/SDAR-30B-A3B-Chat`    | SDAR series diffusion language model (Chat), MoE architecture.                                   |
| **DiffusionGemma**         | `google/diffusiongemma-26B-A4B-it` | Uniform-state (renoising) block-diffusion model, 26B-A4B MoE with a Gemma4 vision tower (text and image input). |
