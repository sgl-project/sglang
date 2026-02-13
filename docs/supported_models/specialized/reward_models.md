# Reward Models

These models output a scalar reward score or classification result, often used in reinforcement learning or content moderation tasks.

```{important}
They are executed with `--is-embedding` and some may require `--trust-remote-code`.
```

## Example launch Command

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-Math-RM-72B \  # example HF/local path
  --is-embedding \
  --host 0.0.0.0 \
  --tp-size=4 \                          # set for tensor parallelism
  --port 30000 \
```

## Supported models

| Model Family (Reward)                                                     | Example HuggingFace Identifier                              | Description                                                                     |
|---------------------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------|
| **Llama (3.1 Reward / `LlamaForSequenceClassification`)**                   | `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2`            | Reward model (preference classifier) based on Llama 3.1 (8B) for scoring and ranking responses for RLHF.  |
| **Gemma 2 (27B Reward / `Gemma2ForSequenceClassification`)**                | `Skywork/Skywork-Reward-Gemma-2-27B-v0.2`             | Derived from Gemma‑2 (27B), this model provides human preference scoring for RLHF and multilingual tasks.  |
| **InternLM 2 (Reward / `InternLM2ForRewardMode`)**                         | `internlm/internlm2-7b-reward`                       | InternLM 2 (7B)–based reward model used in alignment pipelines to guide outputs toward preferred behavior.  |
| **Qwen2.5 (Reward - Math / `Qwen2ForRewardModel`)**                         | `Qwen/Qwen2.5-Math-RM-72B`                           | A 72B math-specialized RLHF reward model from the Qwen2.5 series, tuned for evaluating and refining responses.  |
| **Qwen2.5 (Reward - Sequence / `Qwen2ForSequenceClassification`)**          | `jason9693/Qwen2.5-1.5B-apeach`                      | A smaller Qwen2.5 variant used for sequence classification, offering an alternative RLHF scoring mechanism.  |
