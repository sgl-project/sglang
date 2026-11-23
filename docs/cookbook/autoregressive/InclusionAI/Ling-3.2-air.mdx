---
title: Ling-3.2-air
description: "Deploy Ling-3.2-air with SGLang — FP8 hybrid KDA + MLA MoE vision-language model (124B total, 5.5B active) with image and video input, a 1M-token context, and built-in MTP, on NVIDIA Blackwell and Hopper GPUs."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

Use the Ling-3.2-air image, built from the model-support pull request [#52593](https://github.com/sgl-project/sglang/pull/52593). It is published for both `linux/amd64` (B200, B300, H200, H20-3e, H100, H800) and `linux/arm64` (GB300):

```bash Command
docker pull lmsysorg/sglang:ling-3.2-air
```

Select **Docker** in the command panel to launch the image on the host. The **Python** command runs inside a container started from the same image. See [Install with Docker](/docs/get-started/install#method-3-using-docker) for host setup.

</Accordion>

Pick your hardware to generate the launch command. Each GPU has one tensor-parallel layout: TP2 on B200, TP4 on GB300, B300, H200 and H20-3e, and TP8 on H100 and H800.

- **Spec Decode → Off** — the default. Standard decoding.
- **Spec Decode → MTP (NEXTN)** — speculative decoding with the MTP layer built into the checkpoint, on the same layout. On the measured GPUs it is slower than standard decoding; see [MTP (NEXTN)](#mtp-nextn).

The B200 and GB300 recipes ran on the release weights through full 1319-question GSM8K, text, reasoning, tool-call, image and video probes, and speed benchmarks; the MTP recipes also passed long-context retrieval. The other GPUs use the model card's layouts and are not verified.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/inclusionAI/ling-3.2-air.jsx";
import { benchmarks } from "/src/snippets/configs/inclusionAI/ling-3.2-air-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

The Playground changes the tensor-parallel size of the selected recipe and shows the command diff. Commands it changes are not verified; the [coverage table](#coverage) lists which TP sizes each GPU can run.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model introduction

[Ling-3.2-air](https://huggingface.co/inclusionAI/Ling-3.2-air) shares the native multimodal architecture of [Ling-3.0-flash-VL](/cookbook/autoregressive/InclusionAI/Ling-3.0-flash-VL) (`BailingMoeV3VLForConditionalGeneration`). It is a hybrid-linear Mixture-of-Experts model with 124B total and 5.5B activated parameters.

- **Language model:** 42 layers, 35 Kimi Delta Attention (KDA) and 7 gated Multi-head Latent Attention (MLA) in a 5:1 pattern. The first 2 layers are dense; the rest use 512 routed experts (8 active per token) plus 1 shared expert, with an expert intermediate size of 768. One MTP layer is included for speculative decoding.
- **Vision:** a 27-layer vision encoder with 16-pixel patches and 2×2 spatial merging feeds image and video tokens to the language model. Positions use three-axis M-RoPE.
- **Precision:** language-model weights are FP8 E4M3 in 128×128 blocks with dynamic activation quantization; the vision encoder, projector, embeddings and LM head are not quantized. The checkpoint is about 129 GB.
- **Context:** 1,048,576 tokens, through YaRN (factor 4) over a 262,144-token native window, configured in the checkpoint's `config.json`.
- **Capabilities:** image, multi-image and video input; reasoning, on by default; structured tool calling.

**Checkpoint:** [inclusionAI/Ling-3.2-air](https://huggingface.co/inclusionAI/Ling-3.2-air). **License:** MIT.

**Generation defaults:** `temperature=1.0`, `top_p=0.95`, `top_k=20`, from the checkpoint's `generation_config.json`. SGLang applies them to requests that do not set these fields.

## 2. Configuration tips

### Coverage

<table style={{width: "100%", borderCollapse: "collapse", fontSize: "14px"}}>
  <colgroup>
    <col style={{width: "28%"}} />
    <col style={{width: "22%"}} />
    <col style={{width: "50%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.35)"}}>
      <th style={{textAlign: "left", padding: "6px 8px"}}>Hardware</th>
      <th style={{textAlign: "left", padding: "6px 8px"}}>Layout</th>
      <th style={{textAlign: "left", padding: "6px 8px"}}>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>2× B200</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>TP2</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Verified, MTP and standard decoding</td>
    </tr>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>4× GB300</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>TP4</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Verified, MTP and standard decoding</td>
    </tr>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>4× B300, H200 or H20-3e</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>TP4</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Not verified; model-card layout</td>
    </tr>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>8× H100 or H800</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>TP8</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Not verified; model-card layout</td>
    </tr>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Other TP sizes</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>TP2, TP4 or TP8</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Not verified; set **TP** in the Playground. TP2 is unavailable on 80 GB GPUs, and TP8 on a four-GPU GB300 node.</td>
    </tr>
    <tr>
      <td style={{textAlign: "left", padding: "6px 8px"}}>AMD, Ascend, multi-node, PD disaggregation</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>—</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Not covered: no recipe has been run, and the image targets NVIDIA GPUs.</td>
    </tr>
  </tbody>
</table>

### Parallelism and memory

- The recipes leave attention and MoE backends, memory fraction and context length at their defaults. SGLang detects FP8 from the checkpoint.
- At TP2 each rank holds 384 expert channels, three full 128-wide FP8 blocks. At TP4 and TP8 the per-rank slice (192 or 96) is not block-aligned, so SGLang pads it to the next block while loading; the TP4 padded path ran on GB300, and TP8 has not been run.
- Measured per-GPU weight memory: 59.8 GB on B200 at TP2 and 39.6 GB on GB300 at TP4. With MTP and the 1M context, the KV pool held 5,717,321 tokens on 2× B200 and 15,666,058 tokens on 4× GB300.
- `--trust-remote-code` lets SGLang load the model config and multimodal processor that the checkpoint declares through `auto_map`.
- `--reasoning-parser auto --tool-call-parser auto` both resolve to `ling3` from the chat template; `--reasoning-parser ling3 --tool-call-parser ling3` is equivalent. Reasoning is returned in `message.reasoning_content`, and tool calls in `message.tool_calls`.

### MTP (NEXTN)

Standard decoding is the default recipe; MTP is an option.

- `--speculative-algorithm NEXTN` loads the draft layer from the same checkpoint and defaults to 3 steps, top-k 1, and 4 draft tokens; no draft model path is needed.
- Measured accept length was 2.81 on B200 TP2 and 2.79 on GB300 TP4 over full GSM8K, and 2.49 on GB300 TP4 over ShareGPT prompts (tokens per verify step, including the target's bonus token). Full GSM8K accuracy with MTP was within 0.4 points of standard decoding on the same GPU.
- With MTP, SGLang caps concurrent requests at 48 unless `--max-running-requests` is set; the measured recipes use this default. Set it for your target concurrency, within the KV-pool capacity.
- MTP had lower output throughput and higher TTFT and TPOT than standard decoding, both in the random-prompt benchmarks on the cards and on real ShareGPT prompts:

<table style={{width: "100%", borderCollapse: "collapse", fontSize: "14px"}}>
  <colgroup>
    <col style={{width: "28%"}} />
    <col style={{width: "18%"}} />
    <col style={{width: "18%"}} />
    <col style={{width: "18%"}} />
    <col style={{width: "18%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.35)"}}>
      <th style={{textAlign: "left", padding: "6px 8px"}}>Decoding</th>
      <th style={{textAlign: "right", padding: "6px 8px"}}>Concurrency</th>
      <th style={{textAlign: "right", padding: "6px 8px"}}>Output tok/s</th>
      <th style={{textAlign: "right", padding: "6px 8px"}}>TTFT P50 (ms)</th>
      <th style={{textAlign: "right", padding: "6px 8px"}}>TPOT P50 (ms)</th>
    </tr>
  </thead>
  <tbody>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Standard</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>1</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>279.2</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>29</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>3.51</td>
    </tr>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>MTP (NEXTN)</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>1</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>169.1</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>161</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>5.60</td>
    </tr>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Standard</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>16</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>2011.2</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>41</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>7.80</td>
    </tr>
    <tr>
      <td style={{textAlign: "left", padding: "6px 8px"}}>MTP (NEXTN)</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>16</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>1356.1</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>179</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>11.30</td>
    </tr>
  </tbody>
</table>

The ShareGPT runs used 4× GB300 at TP4 with the native 256K-token RoPE configuration: the first user turn of each conversation through the chat endpoint with the default thinking template, temperature 0, at most 512 output tokens with EOS respected, 48 prompts at concurrency 1 and 256 at concurrency 16, after one discarded 16-prompt warm-up pass, with the cache flushed before each pass. Output throughput counts actual generated tokens, including reasoning.

<Accordion title="ShareGPT benchmark command">

```bash Command
python3 -m sglang.benchmark.serving \
  --backend sglang-oai-chat \
  --host localhost --port 30000 \
  --model inclusionAI/Ling-3.2-air \
  --dataset-name sharegpt \
  --sharegpt-output-len 512 --disable-ignore-eos \
  --num-prompts 256 --max-concurrency 16 \
  --flush-cache --seed 777 --warmup-requests 0 \
  --extra-request-body '{"stream_options":{"include_usage":true}}'
```

For the concurrency-1 row, use `--num-prompts 48 --max-concurrency 1`.

</Accordion>

### Long context

The checkpoint configuration extends the 262,144-token native window to 1,048,576 tokens with YaRN (factor 4) on all three M-RoPE axes. SGLang reads this at startup, so the recipes need no context-length or RoPE flags.

Single-needle retrieval with MTP, with the needle at 75% depth, thinking disabled, and the cache flushed before each request:

<table style={{width: "100%", borderCollapse: "collapse", fontSize: "14px"}}>
  <colgroup>
    <col style={{width: "30%"}} />
    <col style={{width: "25%"}} />
    <col style={{width: "25%"}} />
    <col style={{width: "20%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.35)"}}>
      <th style={{textAlign: "left", padding: "6px 8px"}}>Hardware</th>
      <th style={{textAlign: "right", padding: "6px 8px"}}>Prompt tokens</th>
      <th style={{textAlign: "left", padding: "6px 8px"}}>Result</th>
      <th style={{textAlign: "right", padding: "6px 8px"}}>TTFT (s)</th>
    </tr>
  </thead>
  <tbody>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>2× B200, TP2</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>199,998</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Retrieved, `stop`</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>9.7</td>
    </tr>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>2× B200, TP2</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>399,940</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Retrieved, `stop`</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>13.9</td>
    </tr>
    <tr style={{borderBottom: "1px solid rgba(127,127,127,0.2)"}}>
      <td style={{textAlign: "left", padding: "6px 8px"}}>2× B200, TP2</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>900,100</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Retrieved, `stop`</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>42.7</td>
    </tr>
    <tr>
      <td style={{textAlign: "left", padding: "6px 8px"}}>4× GB300, TP4</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>399,940</td>
      <td style={{textAlign: "left", padding: "6px 8px"}}>Retrieved, `stop`</td>
      <td style={{textAlign: "right", padding: "6px 8px"}}>16.3</td>
    </tr>
  </tbody>
</table>

These are single synthetic cases at one depth; they show that the 1M window is reachable, not general long-context quality.

### Thinking budget and GSM8K

Thinking is on by default. In full GSM8K runs with a 65,536-token budget, 2 to 7 of the 1,319 responses reached the budget, most of them while still reasoning, and ended with `finish_reason: length`. Set `max_tokens` for your task, and disable thinking for short answers with `"chat_template_kwargs": {"enable_thinking": false}`.

To reproduce the accuracy on the benchmark cards, evaluate all 1,319 questions with the checkpoint's sampling settings and check accuracy, `finish_reason`, truncation, and request errors together:

```bash Command
pip install sgl-eval
sgl-eval run gsm8k \
  --base-url http://localhost:30000/v1 \
  --model inclusionAI/Ling-3.2-air \
  --temperature 1.0 --top-p 0.95 --max-tokens 65536
```

## 3. Advanced usage

The examples below use `temperature=0` so the output matches the capture; remove it to use the checkpoint's sampling defaults. The outputs were captured on 2× B200 at TP2 with MTP and the 1M-token configuration. The later examples reuse this `client`.

### 3.1 Reasoning

Thinking is on by default. The `ling3` parser streams reasoning in `delta.reasoning_content` and the answer in `delta.content`.

<Accordion title="Streaming reasoning request (Python)">

```python Python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
stream = client.chat.completions.create(
    model="inclusionAI/Ling-3.2-air",
    messages=[{"role": "user", "content": "Compute 17+25 and explain briefly."}],
    temperature=0,
    max_tokens=2048,
    stream=True,
)
reasoning, content, finish_reason = "", "", None
for chunk in stream:
    choice = chunk.choices[0]
    reasoning += getattr(choice.delta, "reasoning_content", None) or ""
    content += choice.delta.content or ""
    finish_reason = choice.finish_reason or finish_reason
print("reasoning_content:", reasoning)
print("content:", content)
print("finish_reason:", finish_reason)
```

</Accordion>

<Accordion title="Example Output">

```text Output
reasoning_content: 17 + 25 = 42. Simple arithmetic. Break it down: 17 + 25 = (10 + 7) + (20 + 5) = (10 + 20) + (7 + 5) = 30 + 12 = 42.\
content: **17 + 25 = 42**

Brief explanation: Break each number into tens and ones — 17 = 10 + 7 and 25 = 20 + 5. Add the tens (10 + 20 = 30) and the ones (7 + 5 = 12), then combine: 30 + 12 = **42**.
finish_reason: stop
```

</Accordion>

To request a direct answer, pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`; `reasoning_content` is then empty.

### 3.2 Tool calling

The `ling3` tool-call parser returns structured calls in `message.tool_calls`, with `finish_reason` set to `tool_calls`. Print `reasoning_content` and `content` as well, and keep the complete assistant message when you send the tool result back.

<Accordion title="Tool request (Python)">

```python Python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
response = client.chat.completions.create(
    model="inclusionAI/Ling-3.2-air",
    messages=[{"role": "user", "content": "Use the weather tool for Paris. Think briefly before calling it."}],
    tools=tools,
    tool_choice="auto",
    temperature=0,
    max_tokens=2048,
)
msg = response.choices[0].message
print("reasoning_content:", msg.reasoning_content)
print("content:", msg.content)
print("tool_calls:", [(call.function.name, call.function.arguments) for call in msg.tool_calls])
print("finish_reason:", response.choices[0].finish_reason)
```

</Accordion>

<Accordion title="Example Output">

```text Output
reasoning_content: The user wants me to use the weather tool for Paris. Let me call it.
content: I'll call the weather tool for Paris now.
tool_calls: [('get_weather', '{"city": "Paris"}')]
finish_reason: tool_calls
```

</Accordion>

### 3.3 Image understanding

Send images as `image_url` content parts. For multi-image input, put several `image_url` parts in one message.

<Accordion title="Image request (Python)">

```python Python
response = client.chat.completions.create(
    model="inclusionAI/Ling-3.2-air",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"}},
            {"type": "text", "text": "What is the person doing, and what color are the vehicles?"},
        ],
    }],
    temperature=0,
    max_tokens=2048,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
print("finish_reason:", response.choices[0].finish_reason)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Based on the image, here is what is happening:

*   **What the person is doing:** A man is standing on the back of a vehicle, ironing a blue piece of clothing on a small, portable ironing board that is set up on the tailgate. He appears to be a mobile dry cleaner or laundry service worker.
*   **Color of the vehicles:** Both vehicles are **yellow**. They appear to be taxis (a yellow sedan driving by on the left and a yellow SUV stopped on the right).
finish_reason: stop
```

</Accordion>

### 3.4 Video understanding

Send clips as `video_url` content parts.

<Accordion title="Video request (Python)">

```python Python
response = client.chat.completions.create(
    model="inclusionAI/Ling-3.2-air",
    messages=[{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "https://raw.githubusercontent.com/evolvinglmms-lab/sglang/dev/onevision_local/assets/jobs.mp4"}},
            {"type": "text", "text": "Describe the video action in one short sentence."},
        ],
    }],
    temperature=0,
    max_tokens=2048,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
print("finish_reason:", response.choices[0].finish_reason)
```

</Accordion>

<Accordion title="Example Output">

```text Output
A man on stage holds up a small white device, revealing it to be an iPod Nano, while a large screen behind him displays a close-up of the device.
finish_reason: stop
```

</Accordion>
