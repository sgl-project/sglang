---
title: MiniCPM5-2B
description: "Deploy OpenBMB MiniCPM5-2B with SGLang — a 2.5B dense Llama-architecture on-device model with 131K context, thinking mode, XML tool calling and DSpark speculative decoding on H200, RTX PRO 6000, RTX 5090 and DGX Spark."
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
uv pip install --prerelease=allow "sglang>=0.5.12"
```

Then run the **Python** output of the command panel below in that environment.

</Tab>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:dev
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick your card to generate the launch command. MiniCPM5-2B is a 2.5B dense model and runs single-GPU at TP=1 on every supported card, so the page ships one operating point per card plus an optional **DSpark** speculative-decoding row.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/openbmb/minicpm5-2b.jsx";
import { benchmarks } from "/src/snippets/configs/openbmb/minicpm5-2b-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

<Note>
  Speed numbers exist for the RTX 5090 and DGX Spark cells — a single card at
  random 1024/1024, recorded as Mean. The H200 and RTX PRO 6000 cells are
  pending measurement, and no accuracy numbers have been taken on any platform
  yet. The DSpark overlay carries no speed numbers on any card.
</Note>

## Playground

The Playground is where you experiment with **SGLang features beyond the recipes above**. The Deploy panel emits this model's documented launch recipes; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

**MiniCPM5-2B** is the second model in OpenBMB's MiniCPM5 series, following MiniCPM5-1B. It is a dense 2B-class Transformer built for on-device assistants, local deployment, coding agents and tool-use workflows — scenarios where a compact model with a small deployment footprint is preferred. OpenBMB reports 2B-class open-source SOTA within its comparison set (average 53.9), with its clearest advantages in code reasoning, math reasoning, long-context understanding, tool use and agentic tasks.

The checkpoint uses the standard `LlamaForCausalLM` architecture — no custom kernels and no model-code fork — so SGLang loads it through the stock Llama path. It is 2,516,756,480 parameters (1,981,982,720 non-embedding) over 42 layers with GQA (16 query heads, 2 KV heads), and a native context length of 131,072 tokens. Post-training runs SFT → RL → On-Policy Distillation, which merges 16 RL expert models into the single released checkpoint. Weights are released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) license.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "34%"}} />
    <col style={{width: "22%"}} />
    <col style={{width: "44%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Checkpoint</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Precision</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Role on this page</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/openbmb/MiniCPM5-2B">openbmb/MiniCPM5-2B</a></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>BF16</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>The served model in every cell above (final release, post-trained with RL + OPD).</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.05)"}}><a href="https://huggingface.co/openbmb/MiniCPM5-2B-DSpark">openbmb/MiniCPM5-2B-DSpark</a></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>BF16 draft</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Draft model loaded by the <strong>DSPARK</strong> row of the Deploy panel; not served on its own.</td>
    </tr>
  </tbody>
</table>

OpenBMB also publishes SFT-only, mid-training and base checkpoints, plus GGUF, MLX and GPTQ-Int4 exports for llama.cpp / Ollama / LM Studio / Apple Silicon. Those target other runtimes and are not part of the SGLang matrix above.

**Recommended generation:** `temperature=1.0`, `top_p=0.95` (informational — SGLang reads the checkpoint's `generation_config.json`; do not hardcode these in client code).

**Resources:** [HuggingFace](https://huggingface.co/openbmb/MiniCPM5-2B) · [ModelScope](https://www.modelscope.cn/models/OpenBMB/MiniCPM5-2B) · [GitHub](https://github.com/OpenBMB/MiniCPM) · [Tech report](https://arxiv.org/pdf/2506.07900).

## 2. Advanced Usage

### 2.1 Thinking Mode

Thinking is controlled by the chat template's `enable_thinking` flag, passed per request through `chat_template_kwargs`. The `qwen3` reasoning parser — already in every generated command, and toggleable from the **Parsers** card in the [Playground above](#playground) — splits the `<think>` segment into `message.reasoning_content` and leaves the final answer in `message.content`. Drop the flag and the thinking text stays inline in `content`, closing tag and all.

<Accordion title="Thinking On / Off (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

# Thinking on (chat-template default).
resp = client.chat.completions.create(
    model="openbmb/MiniCPM5-2B",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
msg = resp.choices[0].message
print("Reasoning:", getattr(msg, "reasoning_content", None))
print("Answer:", msg.content)

# Thinking off — a direct answer, no deliberation segment.
resp = client.chat.completions.create(
    model="openbmb/MiniCPM5-2B",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print("Without thinking:", resp.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: We are asked: "What is 15% of 240?" This is a simple percentage calculation. To find 15% of 240, we multiply 240 by 15% (which is 0.15). So: 240 × 0.15 = 36. Alternatively, we can think of it as (15/100) × 240 = (15 × 240)/100 = 3600/100 = 36. So the answer is 36.

We need to respond in a helpful way. The user might be testing or seeking quick answer. Let's provide a clear response with explanation if needed, but since it's straightforward, we can just give the answer directly or briefly explain.

Answer:

15% of 240 is **36**.

To calculate:
\( 240 \times 0.15 = 36 \)
Reasoning: None
Answer: To find 15% of 240, multiply 240 by 15% (which is 0.15):

\[
240 \times 0.15 = 36
\]

So, 15% of 240 is **36**.
```

</Accordion>

### 2.2 Tool Calling

MiniCPM5-2B emits XML-style tool calls (`<function name="..."><param name="...">...</param></function>`), and SGLang's built-in `minicpm5` detector converts them to OpenAI-compatible `tool_calls`, parallel calls included. The parser is already in every command the Deploy panel generates; the **Tool Call Parser** chip in the Playground's **Parsers** card is an opt-out, not an opt-in.

<Accordion title="Tool Calling (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["city"],
        },
    },
}]

resp = client.chat.completions.create(
    model="openbmb/MiniCPM5-2B",
    messages=[{"role": "user", "content": "What is the weather in Beijing today?"}],
    tools=tools,
)

msg = resp.choices[0].message
# With the qwen3 reasoning parser on, the thinking segment lands in
# `reasoning_content` and `content` may be empty on a tool-call turn.
print("Reasoning:", getattr(msg, "reasoning_content", None))
print("Content:", msg.content)
for call in (msg.tool_calls or []):
    print("Tool:", call.function.name, call.function.arguments)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning:
The user is asking about the weather in Beijing today. I have access to a get_weather function that can help with this. Let me call it to get the current weather for Beijing.

Content:


Tool: get_weather {"city": "Beijing"}
```

</Accordion>
