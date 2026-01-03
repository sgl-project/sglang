# Diffusion Language Models

Diffusion language models have shown promise for non-autoregressive text generation with parallel decoding capabilities. Unlike auto-regressive language models, different diffusion language models require different decoding strategies.

## Example Launch Command

```shell
python3 -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \ # example HF/local path
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config ./config.yaml \ # Optional. Uses the algorithm's default if not set.
  --host 0.0.0.0 \
  --port 30000
```

## Example Configuration File

```yaml
# Confidence threshold for accepting predicted tokens
# - Higher values: More conservative, better quality but slower
# - Lower values: More aggressive, faster but potentially lower quality
# Range: 0.0 - 1.0
threshold: 0.95

# Default: 32, for LLaDA2MoeModelLM
block_size: 32
```
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

| Model Family                               | Example Model                          | Description                                                                 |
| ------------------------------------------ | -------------------------------------- | --------------------------------------------------------------------------- |
| **LLaDA2.0 (mini, flash)** | `inclusionAI/LLaDA2.0-flash` | LLaDA2.0-flash is a diffusion language model featuring a 100B Mixture-of-Experts (MoE) architecture. |
