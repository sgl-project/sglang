---
title: Nemotron3.5-Lightning
description: "Deploy NVIDIA Nemotron 3.5 Lightning with SGLang — NVFP4 serving with MTP, DFlash, and DSpark speculative decoding, reasoning, and tool calling."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install). The two paths below match the **Python / Docker** toggle in the command panel.

<Tabs>

<Tab title="Python (pip / uv)">

```bash Command
pip install --upgrade pip
pip install uv
SGLANG_BUILD_RUST_EXTS=none uv pip install --prerelease=allow 'git+https://github.com/sgl-project/sglang.git#subdirectory=python'
```

Then run the **Python** output of the command panel below in that environment.

</Tab>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:dev-nemotron3-5-lightning
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick your hardware and recipe to generate the launch command. Every platform publishes four operating points: **Balanced** (no speculation) plus three speculative decoders — **MTP**, **DFlash**, and **DSpark**. Use the Playground below to explore knobs beyond them.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/nvidia/nemotron-3.5-lightning.jsx";
import { benchmarks } from "/src/snippets/configs/nvidia/nemotron-3.5-lightning-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations the SGLang team has signed off on; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

**NVIDIA Nemotron 3.5 Lightning** is a 30B-A3B hybrid reasoning LLM. See the Hugging Face model cards below for architecture and evaluation details.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Checkpoint</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Precision</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4">NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4</a></strong></td>
      <td style={{padding: "9px 12px"}}>NVFP4</td>
      <td style={{padding: "9px 12px"}}>Serving — the checkpoint this page deploys</td>
    </tr>
    <tr style={{background: "rgba(255,255,255,0.02)"}}>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16">NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16</a></strong></td>
      <td style={{padding: "9px 12px"}}>BF16</td>
      <td style={{padding: "9px 12px"}}>Full-precision reference</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash">NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash</a></strong></td>
      <td style={{padding: "9px 12px"}}>W4A16</td>
      <td style={{padding: "9px 12px"}}>DFlash speculative draft model</td>
    </tr>
    <tr style={{background: "rgba(255,255,255,0.02)"}}>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark">NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark</a></strong></td>
      <td style={{padding: "9px 12px"}}>W4A16</td>
      <td style={{padding: "9px 12px"}}>DSpark speculative draft model</td>
    </tr>
  </tbody>
</table>

MTP needs no separate download — the draft head is embedded in the target checkpoint.

## 2. Usage

The server speaks the OpenAI API. With `--reasoning-parser nemotron_3` enabled, the thinking trace lands in `message.reasoning_content` and the answer in `message.content`.

```python Example
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="null",
)

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Briefly explain: what is SGLang?"},
    ],
    temperature=1.0,
    top_p=0.95,
    max_tokens=1024,
)
choice = response.choices[0]
print("Reasoning:", choice.message.reasoning_content)
print("Content:", choice.message.content)
```

### 2.1 Tool Calling

With `--tool-call-parser qwen3_coder` enabled, structured tool calls are returned in `message.tool_calls`.

```python Example
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="null",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_tip",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill_total": {"type": "integer", "description": "The total amount of the bill"},
                    "tip_percentage": {"type": "integer", "description": "The percentage of tip to be applied"},
                },
                "required": ["bill_total", "tip_percentage"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    messages=[{"role": "user", "content": "My bill is $50. What will be the amount for 15% tip?"}],
    tools=TOOLS,
    max_tokens=1024,
)
choice = response.choices[0]
print("Content:", choice.message.content)
print("Tool calls:", choice.message.tool_calls)
```
