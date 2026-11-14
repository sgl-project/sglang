---
title: Mistral Large 4
description: "Deploy Mistral Large 4 with SGLang — the 1T-parameter (52B active) multimodal MoE with DSA sparse attention, served in FP8 on 8× NVIDIA B300 or as NVFP4 on 4× NVIDIA GB300."
tag: NEW
---

## Deployment

<Accordion title="Install SGLang">

```bash Command
docker pull lmsysorg/sglang:mistral-large-4
```

The image ships with `mistral-common` 1.13.0 or later, required to load the checkpoints' `tekken.json`. For how to launch it, see [Install → Method 3: Using Docker](/docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Accordion>

Pick your hardware to generate the launch command:

- **Low Latency** — fastest reply for a single user, with DSpark speculative decoding from the draft bundled in the checkpoint. Pick for chat.
- **High Throughput** — most tokens per second across many users, without speculative decoding. Best for batch jobs.

The verified recipes are FP8 at TP8 on 8× B300, NVFP4 at TP4 on 4× GB300, and on Vera Rubin (arm64) FP8 at TP8 across two 4-GPU trays or NVFP4 at TP4 on one tray. Commands for other GPUs are badged **Not Verified**; combinations without a command explain why in the panel. Every command carries `--reasoning-parser auto --tool-call-parser auto`: without them, reasoning and tool calls come back as raw text in `content`.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/mistralai/mistral-large-4.jsx";
import { benchmarks } from "/src/snippets/configs/mistralai/mistral-large-4-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations the SGLang team has signed off on; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

**Mistral Large 4** is Mistral AI's 1T-parameter Mixture-of-Experts model with 52B active parameters and a vision encoder. It has 72 layers with 128 routed experts (top-4) plus one shared expert, multi-head latent attention with DSA sparse attention, and cross-layer KV sharing. The checkpoints declare a native 1,048,576-token context and bundle a DSpark draft model for speculative decoding.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Checkpoint</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Weights</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Hardware</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}>[`mistralai/Mistral-Large-4-1T-A52B`](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)</td>
      <td style={{padding: "9px 12px"}}>FP8 E4M3, 128×128 block scales</td>
      <td style={{padding: "9px 12px"}}>8× B300, two Vera Rubin trays</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}>[`mistralai/Mistral-Large-4-1T-A52B-NVFP4`](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4)</td>
      <td style={{padding: "9px 12px"}}>NVFP4 routed experts, block-FP8 attention and dense layers</td>
      <td style={{padding: "9px 12px"}}>Blackwell only: 4× GB300, one Vera Rubin tray</td>
    </tr>
  </tbody>
</table>

**Recommended generation:** `temperature=1.0`, `top_p=0.95` (the checkpoint's `generation_config.json` defaults; informational — do not hardcode in client code).

**License:** Mistral Ecosystem Open-Weight License; see the model card.

**Resources:** HuggingFace [FP8](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) and [NVFP4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4).

## 2. Benchmarks

Accuracy uses `sgl-eval` at temperature 1.0, top-p 0.95, at most 8,192 output tokens and `reasoning_effort` `low`, unless noted. Speed uses `bench_serving` with 1,024-token random prompts and 512 output tokens, 64 requests per concurrency point and a flushed cache; total tok/s/GPU counts input and output tokens. The benchmark card of each recipe carries the full protocol and a ⚡ Reproduce button.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Hardware</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Recipe</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}>8× B300</td>
      <td style={{padding: "9px 12px"}}>FP8, TP8, FP8 KV</td>
      <td style={{padding: "9px 12px"}}>GSM8K **96.74%** (1,319 questions, all stopped); **774** total tok/s/GPU at concurrency 64</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}>8× B300</td>
      <td style={{padding: "9px 12px"}}>FP8, TP8, DSpark</td>
      <td style={{padding: "9px 12px"}}>**3.71** accepted tokens per 4-token verify step; **1,018** total tok/s/GPU at concurrency 64</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}>4× GB300</td>
      <td style={{padding: "9px 12px"}}>NVFP4, TP4</td>
      <td style={{padding: "9px 12px"}}>GSM8K, first 100 questions: **98%** at `low`, 97–98% at `high`; MMMU-Pro **72.43%** (seed 1); **1,560** total tok/s/GPU at concurrency 64</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}>4× GB300</td>
      <td style={{padding: "9px 12px"}}>NVFP4, TP4, DSpark</td>
      <td style={{padding: "9px 12px"}}>**3.69** accepted tokens per 4-token verify step; **2,812** total tok/s/GPU at concurrency 64</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}>Vera Rubin, two trays</td>
      <td style={{padding: "9px 12px"}}>FP8, TP8, BF16 KV</td>
      <td style={{padding: "9px 12px"}}>GSM8K **96.64%**, MMLU **91.70%**, MMMU-Pro **72.22%** (mean of 3 seeds)</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}>Vera Rubin, one tray</td>
      <td style={{padding: "9px 12px"}}>NVFP4, TP4, BF16 KV</td>
      <td style={{padding: "9px 12px"}}>GSM8K **96.74%**, MMLU **91.64%**, MMMU-Pro **72.66%** (mean of 3 seeds)</td>
    </tr>
  </tbody>
</table>

## 3. Configuration Tips

- **Parsers.** `auto` resolves to the `mistral` reasoning and tool-call parsers; `--reasoning-parser mistral --tool-call-parser mistral` is equivalent. Without a reasoning parser the reasoning stays inline in `content` as `[THINK]…[/THINK]`; without a tool-call parser tool calls come back as `[TOOL_CALLS]…` text.
- **Tokenizer.** By default the server tokenizes with the checkpoint's `tokenizer.json` and renders `chat_template.jinja`, which is token-identical to `mistral-common` for chat prompts. `--tokenizer-mode mistral` tokenizes through `mistral-common` instead. Either way `mistral-common` 1.13.0 or later is required: earlier versions cannot load the checkpoints' `tekken.json`.
- **Context and KV cache.** Leave `--context-length` unset for the native 1,048,576 tokens. The KV cache defaults to FP8 (E4M3) on Blackwell and Vera Rubin and BF16 on Hopper; on 8× B300 at `--mem-fraction-static 0.85` the FP8 KV pool holds about 2.49M tokens, and about 1.34M with `--kv-cache-dtype bfloat16`.
- **DSpark.** The server loads the bundled draft from the checkpoint (3 drafted tokens, 4-token verify step); do not pass `--speculative-draft-model-path` or draft-length flags. With speculative decoding `--max-running-requests` defaults to 48. For bursts of short requests, `--max-running-requests 64` raises throughput (the GB300 Low Latency recipe sets it); keep 48 or lower for long-context traffic.
- **Long prompts.** Prefill CP 2 (Playground → **Attention** → **Prefill CP**) on top of the Low Latency recipe lowers long-prompt TTFT at a small decode cost.
- **KV capacity.** **Decode Context Parallel (DCP)** and **HiSparse** in the Playground hold more tokens at a decode-throughput cost. Neither combines with prefill CP.
- **Vera Rubin.** The generated commands carry the environment variables of the measured runs; tokenizer, context and KV dtype resolve from the checkpoint like on other hardware. The `mistral-large-4` image is multi-arch (amd64 + arm64).

## 4. Advanced Usage

### 4.1 Reasoning

The reasoning comes back in `reasoning_content` and the answer in `content`. `reasoning_effort` accepts `low`, `medium` and `high`; a request that omits it reasons at the checkpoint default, `high`, and `"none"` returns HTTP 400.

<Accordion title="Reasoning Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="mistralai/Mistral-Large-4-1T-A52B",
    messages=[{"role": "user", "content": "A train travels 120 km in 1.5 hours. What is its average speed in km/h? Answer briefly."}],
    reasoning_effort="high",
)
msg = resp.choices[0].message
print("Reasoning:", msg.reasoning_content)
print("Answer:", msg.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: The user wants me to calculate the average speed of a train that travels 120 km in 1.5 hours.

 The formula for average speed is:
 Speed = Distance / Time

 Given:
 - Distance = 120 km
 - Time = 1.5 hours

 Calculation:
 Speed = 120 km / 1.5 hours
 Speed = 80 km/h

 The user asked for a brief answer, so I should just provide the numerical answer with units.

 120 / 1.5 = 80

 So the answer is 80 km/h.
Answer: 80 km/h
```

</Accordion>

### 4.2 Image Input

Images go in OpenAI-style `image_url` content parts, as URLs or base64 data URLs.

<Accordion title="Image Example (Python)">

```python Example
import base64, io
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (448, 448), "white")
draw = ImageDraw.Draw(img)
draw.ellipse((60, 60, 388, 388), fill="red")
draw.text((150, 140), "42", fill="white", font=ImageFont.load_default(size=140))
buf = io.BytesIO()
img.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="mistralai/Mistral-Large-4-1T-A52B",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "What number is written in the image, and what color is the circle? Answer briefly."},
        {"type": "image_url", "image_url": {"url": url}},
    ]}],
)
print(resp.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
The number written in the image is **42**, and the circle is **red**.
```

</Accordion>

### 4.3 Tool Calling

Function calls come back as structured `tool_calls` with `finish_reason` `tool_calls`.

<Accordion title="Tool Calling Example (Python)">

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
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]
resp = client.chat.completions.create(
    model="mistralai/Mistral-Large-4-1T-A52B",
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    tools=tools,
)
choice = resp.choices[0]
print("finish_reason:", choice.finish_reason)
for call in choice.message.tool_calls:
    print(call.function.name, call.function.arguments)
```

</Accordion>

<Accordion title="Example Output">

```text Output
finish_reason: tool_calls
get_weather {"city": "Paris"}
```

</Accordion>

## 5. Known Limitations

- **Pipeline parallelism** must keep layer 35 and the layers that read its KV cache on one stage (`SGLANG_PP_LAYER_PARTITION=35,37` for two stages). DSpark accepts pipeline parallelism only on a PD prefill server.
- **PD disaggregation** runs with a peer-config fingerprint guard; a 1P1D setup (one prefill and one decode server) is validated end to end.
- **Prefill CP with DCP** is unsupported; the server rejects the combination at startup.
- **KV-cache sharding** (`--enable-kv-cache-sharding`) is unsupported with the model's cross-layer KV sharing; the server rejects it at startup.
