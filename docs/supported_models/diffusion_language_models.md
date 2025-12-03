# Diffusion Language Models

Diffusion models are extensively used for generating images and videos and are becoming effective for text data generation. Compared to autoregressive models, diffusion models offer the potential for faster generation and enhanced controllability of outputs.

## Example Launch Command

```shell
python3 -m sglang.launch_server \
  --model-path inclusionAI/LLaDA2.0-mini \  # example HF/local path
  --dllm-algorithm LowConfidence \
  --dllm-block-size 32 \
  --host 0.0.0.0 \
  --port 30000
```

## Supported Models

Below the supported models are summarized in a table.

If you are unsure if a specific architecture is implemented, you can search for it via GitHub. For example, to search for `LLaDA2MoeModelLM`, use the expression:

```
repo:sgl-project/sglang path:/^python\/sglang\/srt\/models\// LLaDA2MoeModelLM
```
in the GitHub search bar.

| Model Family                               | Example Model                          | Description                                                                 |
| ------------------------------------------ | -------------------------------------- | --------------------------------------------------------------------------- |
| **LLaDA2.0 (mini, flash)** | `inclusionAI/LLaDA2.0-flash` | LLaDA2.0-flash is a diffusion language model featuring a 100B Mixture-of-Experts (MoE) architecture. |
