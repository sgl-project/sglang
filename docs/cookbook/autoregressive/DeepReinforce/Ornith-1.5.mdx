---
title: Ornith-1.5
description: "Deploy Ornith-1.5 with SGLang on NVIDIA H100 and H200 GPUs, with recipes for 9B, 35B-A3B, 122B, and 397B models, FP8, reasoning, and tool calling."
tag: NEW
---

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/ornith-ai/ornith-1.5.jsx";

## Deployment

These recipes target a single Linux node with NVIDIA CUDA GPUs. The Ornith-1.5 model cards recommend SGLang `>=0.5.9`. Both parsers are included in every launch command: `qwen3` separates reasoning from the answer, and `qwen3_coder` parses tool calls.

<Accordion title="Install SGLang">

<Tabs>
<Tab title="Python">

```bash Command
pip install --upgrade "sglang>=0.5.9"
```

Run the Python command from the deployment panel in this environment. See the [installation guide](/docs/get-started/install) for platform-specific requirements.

</Tab>
<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:latest
```

Select Docker in the deployment panel to generate the container command.

</Tab>
</Tabs>

</Accordion>

Choose your hardware, model size, and checkpoint format. The 9B recipe uses one GPU, 35B-A3B uses two, and 122B and 397B use eight. BF16 397B requires H200 in this matrix; its FP8 checkpoint is also available on H100. Select **BF16** for a BF16 checkpoint or **FP8** for an FP8 checkpoint. All commands use Hugging Face repository IDs and default to port **30000**.

<Deployment config={config} />

## 1. Model introduction

Ornith-1.5 is an agentic model family published by [Ornith AI](https://huggingface.co/ornith-ai). The 9B, 35B-A3B, 122B, and 397B checkpoints use Qwen3.5 architectures. The family extends Ornith-1.0's self-improvement loop by jointly optimizing task generation, scaffold construction, and solution rollouts through reinforcement learning. The model cards describe applications in coding, reasoning, and tool-using agents.

The recipes expose an OpenAI-compatible API with separate reasoning output, parsed tool calls, and a 262,144-token context window. The table lists the checkpoints covered by this recipe.

<table style={{width: "100%", borderCollapse: "collapse"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px"}}>Checkpoint</th>
      <th style={{textAlign: "left", padding: "10px 12px"}}>Format</th>
      <th style={{textAlign: "left", padding: "10px 12px"}}>Recipe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}><a href="https://huggingface.co/ornith-ai/Ornith-1.5-9B">Ornith-1.5-9B</a></td>
      <td style={{padding: "9px 12px"}}>BF16</td>
      <td style={{padding: "9px 12px"}}>Dense; 1× H100 or H200</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><a href="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B">Ornith-1.5-35B-A3B</a></td>
      <td style={{padding: "9px 12px"}}>BF16</td>
      <td style={{padding: "9px 12px"}}>MoE, ~3B active; 2× H100 or H200</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><a href="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-FP8">Ornith-1.5-35B-A3B-FP8</a></td>
      <td style={{padding: "9px 12px"}}>FP8</td>
      <td style={{padding: "9px 12px"}}>2× H100 or H200</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><a href="https://huggingface.co/ornith-ai/Ornith-1.5-122B">Ornith-1.5-122B</a></td>
      <td style={{padding: "9px 12px"}}>BF16</td>
      <td style={{padding: "9px 12px"}}>MoE, 256 experts with 8 routed per token; 8× H100 or H200</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><a href="https://huggingface.co/ornith-ai/Ornith-1.5-397B">Ornith-1.5-397B</a></td>
      <td style={{padding: "9px 12px"}}>BF16</td>
      <td style={{padding: "9px 12px"}}>MoE; 8× H200</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><a href="https://huggingface.co/ornith-ai/Ornith-1.5-397B-FP8">Ornith-1.5-397B-FP8</a></td>
      <td style={{padding: "9px 12px"}}>FP8</td>
      <td style={{padding: "9px 12px"}}>8× H100 or H200</td>
    </tr>
  </tbody>
</table>

The official [9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B), [35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), [122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B), and [397B](https://huggingface.co/ornith-ai/Ornith-1.5-397B) model cards declare the MIT license.

**Resources:** [Ornith-1.5 model repositories](https://huggingface.co/ornith-ai/models?search=Ornith-1.5) · [Ornith-1.5 blog](https://ornith.ai/ornith_1_5.html)

## 2. Configuration tips

- **Model IDs:** use the `ornith-ai` namespace. The 35B repository includes the `-A3B` suffix. FP8 selections use the actual `-FP8` repository rather than the BF16 repository shown in their model-card quickstarts.
- **Context and memory:** every recipe uses `--context-length 262144`. The 9B, 35B-A3B, and 397B recipes use `--mem-fraction-static 0.85`; the 122B recipe uses `0.80`. Lower the context length if you need more memory headroom.
- **Reasoning:** `--reasoning-parser qwen3` exposes reasoning as `message.reasoning_content`, with the answer in `message.content`.
- **Tool calling:** `--tool-call-parser qwen3_coder` exposes parsed calls in `message.tool_calls`. Your application executes the requested function and submits its result in a follow-up tool message.
- **Sampling:** 35B-A3B and 397B recommend `temperature=0.6`, `top_p=0.95`, and `top_k=20`. For 9B and 122B, the model cards recommend `temperature=1.0` and `presence_penalty=1.5` for general tasks, or `temperature=0.6` and `presence_penalty=0.0` for precise coding; both use `top_p=0.95`, `top_k=20`, `min_p=0.0`, and `repetition_penalty=1.0`. These settings are distinct from benchmark-specific settings. The Python examples below use the precise-coding sampling settings.

## 3. Usage examples

These requests use the precise-coding sampling settings shared by 9B and 122B. Set `model` to the deployment panel's `--served-model-name`, such as `Ornith-1.5-9B` or `Ornith-1.5-122B`. The examples use the default port; match the client port if you change the launch port.

<Accordion title="Chat completion with reasoning (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Ornith-1.5-9B",
    messages=[{"role": "user", "content": "Write a compact Python is_prime function."}],
    temperature=0.6,
    top_p=0.95,
    presence_penalty=0.0,
    max_tokens=1024,
    extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
)
message = response.choices[0].message
print("Reasoning:", getattr(message, "reasoning_content", None))
print("Answer:", message.content)
```

</Accordion>

<Accordion title="Tool-call request (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Ornith-1.5-9B",
    messages=[{"role": "user", "content": "Run the parser module's unit tests."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the project's test suite.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        },
    }],
    tool_choice="auto",
    temperature=0.6,
    top_p=0.95,
    presence_penalty=0.0,
    max_tokens=2048,
    extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
)
message = response.choices[0].message
print("Reasoning:", getattr(message, "reasoning_content", None))
print("Tool calls:", message.tool_calls)
```

</Accordion>

### Agent integration

Configure your agent or coding CLI to use the same OpenAI-compatible endpoint:

- **API base URL:** `http://localhost:30000/v1`.
- **API key:** `EMPTY` for a local server without API-key authentication.
- **Model:** the selected checkpoint's `--served-model-name`, such as `Ornith-1.5-122B`.

Use the served model name rather than the Hugging Face repository ID when the two differ. Keep both parsers enabled so the agent receives separate reasoning and structured tool calls. Your agent executes tools and returns their results to the server in subsequent messages.
