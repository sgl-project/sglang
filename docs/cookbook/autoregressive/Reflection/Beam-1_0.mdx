---
title: Beam 1.0
description: "Deploy Reflection Beam 1.0 with SGLang on B300, GB300, and H200: NVFP4, FP8, and BF16 recipes for low latency and high throughput."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

```bash Command
docker pull lmsysorg/sglang:beam
```

For host setup, see [Install SGLang with Docker](/docs/get-started/install#method-3-using-docker).

</Accordion>

Select your hardware, strategy, and quantization to generate the Docker command.

- **Low-Latency** uses TP with the [DFlash2 drafter](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2) at block size 8.
- **High-Throughput** disables DFlash2. H200 BF16 uses TP8 with an automatic static memory fraction; the other recipes use TP + DPA. Blackwell NVFP4 also enables EP and FlashInfer MegaMoE.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/reflection/beam-1.0.jsx";
import { benchmarks } from "/src/snippets/configs/reflection/beam-1.0-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

Each GSM8K score covers all 1,319 test problems; the benchmark notes identify the evaluation harness.

NVFP4 requires B300 or GB300. **GB300 BF16 uses two four-GPU nodes**: set the head-node address and node rank in **Env**, then run the command on both nodes with rank 0 on the head and rank 1 on the peer.

## TPU Support

Beam 1.0 has day-0 TPU support through SGLang-TorchTPU, a native PyTorch backend for LLM, VLM, and diffusion model inference on TPUs. It combines TPU acceleration with SGLang features such as RadixAttention and HiCache to enable cost-efficient, production-ready serving for agentic workloads.

TPU recipes for Beam 1.0 and access to SGLang-TorchTPU will be available soon. Sign up through this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSe2Y8hzyU3KGB2StA-0huulrVsq79A0tSxcBQwjj8vlhLhYuQ/viewform?usp=publish-editor) to receive the latest updates.

## Usage

All recipes serve the model as `beam` on port 30000. For GB300 BF16, connect to the head node.

<Accordion title="Chat completion (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="beam",
    messages=[{"role": "user", "content": "Explain tensor parallelism briefly."}],
)
message = response.choices[0].message
print("Reasoning:", message.reasoning_content)
print("Answer:", message.content)
```

</Accordion>
