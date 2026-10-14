---
title: Granite 4.2
description: "Deploy Granite 4.2 3B, 8B, and 30B dense models with SGLang on NVIDIA H200 and B200, including thinking modes and structured tool calling."
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
uv pip install --prerelease=allow sglang
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

Pick a Granite 4.2 checkpoint to generate the launch command. The verified matrix covers BF16 serving on one NVIDIA H200 or B200 with tensor parallelism 1.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/ibm-granite/granite-4.2.jsx";
import { benchmarks } from "/src/snippets/configs/ibm-granite/granite-4.2-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

<Note>
  The H200 speed results use `lmsysorg/sglang:dev` at SGLang commit `d59c1ddf7` and the B200 results at commit `d10a656ad8`; the launch recipes were verified end to end on both GPUs against the release checkpoints. Each speed point uses 80 fixed-length random requests at 8,192 input and 1,024 output tokens, 8 warmup requests, a flushed cache, greedy sampling, and ignore-EOS.
</Note>

## Playground

The Playground layers SGLang features on top of the verified recipe. Any override changes the badge to **Not Verified** until that exact configuration is tested end to end.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model introduction

**Granite 4.2** is IBM's dense decoder-only language model family with 3B, 8B, and 30B checkpoints. Each checkpoint uses BF16 weights, has a configured context length of 131,072 tokens, and supports default thinking, non-thinking, low-effort thinking, and structured tool calls through its chat template. The repositories declare the Apache-2.0 license.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Variant</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Total params</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Position in family</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/ibm-granite/granite-4.2-3b">Granite 4.2 3B</a></strong></td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>3B</td>
      <td style={{padding: "9px 12px"}}>Smallest checkpoint</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/ibm-granite/granite-4.2-8b">Granite 4.2 8B</a></strong></td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>8B</td>
      <td style={{padding: "9px 12px"}}>Mid-size checkpoint</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/ibm-granite/granite-4.2-30b">Granite 4.2 30B</a></strong></td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>30B</td>
      <td style={{padding: "9px 12px"}}>Largest checkpoint</td>
    </tr>
  </tbody>
</table>

**Recommended generation:** IBM recommends `temperature=1.0` and `top_p=0.95` for general chat, reasoning, and tool calling. The release checkpoints ship these values in `generation_config.json`; send them per request when you want to be explicit.

**Resources:** [Granite 4.2 3B](https://huggingface.co/ibm-granite/granite-4.2-3b) · [Granite 4.2 8B](https://huggingface.co/ibm-granite/granite-4.2-8b) · [Granite 4.2 30B](https://huggingface.co/ibm-granite/granite-4.2-30b).

## 2. Configuration tips

- **Thinking is enabled by default.** Set `chat_template_kwargs.enable_thinking` to `false` for a direct answer. Set `enable_thinking` and `low_effort` to `true` for a shorter reasoning trace.
- **Give thinking enough tokens.** At `temperature=1.0` the default thinking mode can run past 1,000 tokens on multi-step problems. Use a `max_tokens` of at least 2,048 for thinking requests so the answer is not cut off.
- **Reasoning parser.** Launch with `--reasoning-parser auto`, which resolves to `nemotron_3` for these checkpoints, so OpenAI-compatible responses separate the trace into `message.reasoning_content` and the answer into `message.content`. Without a parser flag the reasoning markup stays inline in `message.content`.
- **Tool-call parser.** Launch with `--tool-call-parser auto`, which resolves to `qwen3_coder` for these checkpoints, so tool requests are returned through `message.tool_calls`. Without it, raw `<tool_call>` markup stays in `message.content`.
- **Single-GPU sizing.** All three BF16 checkpoints loaded and completed chat requests with `--tp 1 --mem-fraction-static 0.8` on one H200 and on one B200. Increase TP only after validating the new topology.
- **Image selection.** The validated path uses `lmsysorg/sglang:dev`. A stable image tested during validation had an incompatible dependency set before model loading, so use the recipe's image until a newer tagged release is confirmed.

## 3. Advanced usage

The outputs below are verbatim captures from Granite 4.2 3B on the verified server. Sampling is stochastic, so a repeated request can produce different wording.

### 3.1 Thinking modes

The `nemotron_3` reasoning parser keeps reasoning and final content in separate fields. Granite 4.2 accepts three chat-template modes: default thinking, non-thinking, and low-effort thinking.

<Accordion title="Thinking modes example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
model = "ibm-granite/granite-4.2-3b"

modes = {
    "thinking": {"enable_thinking": True},
    "non-thinking": {"enable_thinking": False},
    "low-effort": {"enable_thinking": True, "low_effort": True},
}

for name, chat_template_kwargs in modes.items():
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "What is 17 * 23? Answer briefly."}
        ],
        extra_body={"chat_template_kwargs": chat_template_kwargs},
    )
    message = response.choices[0].message
    print(f"[{name}]")
    print("Reasoning:", getattr(message, "reasoning_content", None))
    print("Answer:", message.content)
```

</Accordion>

<Accordion title="Example output">

```text Output
[thinking]
Reasoning: Okay, the user asked "What is 17 * 23? Answer briefly." I need to calculate 17 multiplied by 23.

Let me do the multiplication. 17 times 23.

I can break it down: 17 * 20 = 340, and 17 * 3 = 51. Then add them: 340 + 51 = 391.

Alternatively, 23 * 17: 23*10=230, 23*7=161, 230+161=391. Same result.

So the answer is 391.

The user wants a brief answer, so just state the number.

Answer:
391
[non-thinking]
Reasoning: None
Answer: 391
[low-effort]
Reasoning: Compute 17*23 = 17*20=340, plus 17*3=51 => 391.

Answer:
391
```

</Accordion>

### 3.2 Tool calling

The `qwen3_coder` parser converts the model's tool markup into OpenAI-compatible structured calls.

<Accordion title="Tool calling example (Python)">

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
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="ibm-granite/granite-4.2-3b",
    messages=[{"role": "user", "content": "What is the weather in Boston right now?"}],
    tools=tools,
    tool_choice="auto",
)

choice = response.choices[0]
message = choice.message
print("Reasoning:", getattr(message, "reasoning_content", None))
print("Content:", message.content)
for call in message.tool_calls or []:
    print("Tool:", call.function.name)
    print("Arguments:", call.function.arguments)
print("Finish reason:", choice.finish_reason)
```

</Accordion>

<Accordion title="Example output">

```text Output
Reasoning: Okay, the user is asking for the weather in Boston right now. I need to use the available tool called get_weather. The tool requires the city parameter. Since the user specified Boston, I'll call get_weather with city set to Boston.

Content: None
Tool: get_weather
Arguments: {"city": "Boston"}
Finish reason: tool_calls
```

</Accordion>
