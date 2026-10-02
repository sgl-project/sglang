---
title: IQuest-Q1
description: "Deploy IQuest-Q1 with SGLang — BF16 recipes for the 320B Mixture-of-Experts model on 8× NVIDIA H200, with optional MTP speculative decoding."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

IQuest-Q1 needs an SGLang build that includes IQuest-Q1 support. The nightly image carries it:

```bash Command
docker pull lmsysorg/sglang:dev
```

For other methods, see the [official SGLang installation guide](../../../docs/get-started/install).

</Accordion>

Download the checkpoint first; the MTP draft lives in its `mtp/` subdirectory:

```bash Command
hf download IQuestLab/IQuest-Q1 --local-dir /model/IQuest-Q1
```

Pick a recipe to generate the launch command. **Low-Latency** enables MTP speculative decoding; **High-Throughput** serves the target model alone.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/IQuestLab/iquest-q1.jsx";

<Deployment config={config} />

## 1. Model Introduction

**IQuest-Q1** is a Mixture-of-Experts language model from IQuestLab with about 320B total parameters and 15B active per token. It has 88 layers that interleave three sliding-window attention layers (4,096-token window) with one full-attention layer, uses learned attention sinks, and routes each token to 8 of 256 experts. It ships with a single-layer MTP draft that SGLang applies recursively for speculative decoding.

**Resources:** [HuggingFace](https://huggingface.co/IQuestLab/IQuest-Q1).

## 2. Configuration Tips

- **Hardware.** IQuest-Q1 requires the FA3 attention backend for both the target and the MTP draft, because the attention sink is applied from the FA3 softmax LSE. SGLang builds FA3 for Hopper, so Blackwell GPUs are not supported. The BF16 weights take about 640 GB, which fits 8× H200 with tensor parallelism 8.
- **Reasoning.** Thinking is on by default. Pass `"chat_template_kwargs": {"thinking": false}` to turn it off for a request. The `iquest_q1` reasoning parser returns the thinking in `reasoning_content`.
- **Tool calling.** The `iquest_q1` tool-call parser supports `tool_choice` values `auto`, `required`, and a named function.
- **MTP depth.** The single draft layer runs 5 sequential steps, one token each, and each round verifies 6 tokens. Keep `--speculative-num-draft-tokens` equal to `--speculative-num-steps` + 1.
