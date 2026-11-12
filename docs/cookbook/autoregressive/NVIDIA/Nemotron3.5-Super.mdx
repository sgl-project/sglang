---
title: NVIDIA Nemotron 3.5 Super VL
description: "Deploy Nemotron 3.5 Super VL 120B-A12B with SGLang on B200, H200, RTX PRO 6000, and DGX Spark, with NVFP4 or BF16 and optional MTP."
tag: NEW
---

## Deployment

The **Docker** commands use `lmsysorg/sglang:nemotron3-5-super`. The command panel selects either `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` or `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`. Set `HF_TOKEN` in the panel's Env control if the checkpoint requires access.

Choose **B200, H200, RTX PRO 6000, or DGX Spark** on one node. B200 offers NVFP4 and BF16. H200 offers BF16; NVFP4 is greyed out. RTX PRO 6000 and DGX Spark offer NVFP4; BF16 is greyed out. Choose `bfloat16` or `float32` in **Mamba SSM DType**; `bfloat16` is the default. **High Throughput** leaves MTP off. **Low Latency** adds EAGLE MTP with top-k 1, three steps, and four draft tokens. These names identify configuration choices; no throughput or latency comparison was measured. Both command tabs use `sglang serve`.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/nvidia/nemotron-3.5-super-vl.jsx";
import { benchmarks } from "/src/snippets/configs/nvidia/nemotron-3.5-super-vl-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

The Playground inherits the selected recipe and lets you toggle the reasoning and tool-call parsers. Parser changes are separate experiments.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model introduction

**NVIDIA Nemotron 3.5 Super VL 120B-A12B** is a vision-language checkpoint in the Nemotron-H hybrid MoE family. The early-access NGC checkpoint configuration identifies `NemotronH_Omni_Reasoning_V3` and includes a vision encoder. The command panel uses the [NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) and [BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) Hugging Face IDs; the source runs used the [NGC model](https://registry.ngc.nvidia.com/orgs/0767305323357365/teams/super3-5-vl-ea-0911/models/nemotron-3.5-super-vl-120b-a12b-ea20261008). Check the checkpoint pages for access terms and model details.

The SGLang model entry point is [`nano_nemotron_vl.py` on the `nemotron3-super-dev-20261015` branch](https://github.com/sgl-project/sglang/blob/nemotron3-super-dev-20261015/python/sglang/srt/models/nano_nemotron_vl.py), with its supporting kernels and runtime code on the same branch.

The validation used text-only GSM8K prompts. It does **not** establish image or video request correctness, long-context behavior, or tool-call accuracy.

## 2. Configuration tips and validation

- **B200, NVFP4 or BF16:** both strategies use one eight-GPU TP8 node with FP8 E4M3 KV cache and 16,384-token configured context. Select the Mamba state dtype in the wizard; SGLang selects the attention backend and memory limits by default.
- **H200, BF16:** both strategies use one eight-GPU TP8 node with FA3 attention, Triton MoE, and `--disable-flashinfer-autotune`.
- **RTX PRO 6000, NVFP4:** both strategies served at TP1 on GPU 1 of an eight-GPU G4 machine. The configured context was 262,144 tokens; only short prompts were tested.
- **DGX Spark, NVFP4:** both strategies completed 300-question text-only GSM8K runs at TP1 with `bfloat16` Mamba state. The `float32` choice has no completed accuracy run.
- **KV precision:** these recipes deliberately use `--kv-cache-dtype fp8_e4m3` to match the validated runs. KV precision can affect output quality; no BF16-KV comparison was measured.

Server information confirmed `speculative_algorithm: null` for High Throughput and `EAGLE` for Low Latency. The runs used installed early-access NGC weights, not the Hugging Face checkpoint IDs in the command panel or the public Docker image.

The Deployment panel’s Benchmark card shows text-only GSM8K accuracy for measured selections and a pending state for unmeasured selections. The existing green Verified status marks a recipe with a completed source-run. These results do not validate the public checkpoint IDs or Docker image.

## 3. Reasoning output

The base recipes enable `--reasoning-parser nemotron_3`. In the tested chat completions, the thinking trace appeared in `message.reasoning_content` and the final answer in `message.content`. The tool-call parser is configured as `qwen3_coder`, but structured tool calls were not part of this accuracy check.

<Accordion title="Reasoning example (Python client)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
model = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": "Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?\nSolve the problem and put the final numeric answer after ####.",
    }],
)
message = response.choices[0].message
print("Reasoning:", message.reasoning_content)
print("Content:", message.content)
```

</Accordion>

<Accordion title="Example output from the B200 NVFP4 smoke run">

```text Output
Reasoning: The user wants me to solve a math problem.

Claire makes a 3 egg omelet every morning.
She eats 3 eggs per day.
4 weeks = 4 × 7 = 28 days.
Total eggs = 3 × 28 = 84 eggs.
Dozens of eggs = 84 / 12 = 7 dozens.

Final answer: 7.

I need to put the final numeric answer after ####.
Content: Claire eats 3 eggs per day.
In 4 weeks, there are 4 × 7 = 28 days.
Total eggs eaten: 3 × 28 = 84 eggs.
Since 1 dozen = 12 eggs: 84 ÷ 12 = 7 dozens.

#### 7
```

</Accordion>
