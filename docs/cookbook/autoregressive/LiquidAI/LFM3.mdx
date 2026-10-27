---
title: LFM3
description: "Deploy Liquid AI's LFM3 with SGLang — hybrid gated short conv + attention models: LFM3-1.7B text with 256K context and LFM3-VL-2B with native-resolution image and video input, on a single GPU."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all methods and hardware platforms, see the [official SGLang installation guide](/docs/get-started/install).

<Note>
LFM3 support ships on SGLang `main` and is **not yet in a tagged release**, so `pip install sglang` will not include it — use the image below.
</Note>

LFM3 support ships in the dev image built from the model support PR:

```bash Command
docker pull lmsysorg/sglang:dev-lfm3
```

The command panel below has a **Python / Docker** toggle. The Python output is the bare `sglang serve` command: run it inside a container started from this image. The Docker output wraps the same command in `docker run`. On AMD there is no published ROCm image with LFM3 yet — install from source and run the Python output in that environment.

For how to launch the image, see [Install → Method 3: Using Docker](/docs/get-started/install#method-3-using-docker). A minimal example (substitute the inner `sglang serve ...` with whatever the command generator below produces):

```bash Command
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<your-hf-token>" \
    --ipc=host \
    lmsysorg/sglang:dev-lfm3 \
    sglang serve <use args below>
```

</Accordion>

Every LFM3 model runs on a **single GPU (TP=1)** — pick your hardware + model variant to generate the launch command. One recipe covers both variants: the attention backend resolves through the LFM3 model overrides, so the command needs no backend flag on NVIDIA. The reasoning and tool-call parsers are part of the generated command — the examples in [§3](#3-advanced-usage) work as-is.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/LiquidAI/lfm3.jsx";
import { benchmarks } from "/src/snippets/configs/LiquidAI/lfm3-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

<div style={{fontSize: "0.85em", lineHeight: "1.55", color: "#6b7280", margin: "0.5rem 0 1rem 0"}}>
  <p style={{margin: "0 0 0.3rem 0"}}><strong>Panel controls</strong> (top of the command box):</p>
  <ul style={{margin: 0, paddingLeft: "1.25rem"}}>
    <li style={{marginBottom: "0.2rem"}}><strong>Python / Docker</strong> — bare <code>sglang serve …</code> for an existing SGLang env, or a <code>docker run … sglang serve …</code> wrap against the dev image from the <a href="#install">Install SGLang</a> panel above.</li>
    <li style={{marginBottom: "0.2rem"}}><strong>⧉ Copy</strong> — copies the current command (with whichever framing is active) to your clipboard.</li>
    <li style={{marginBottom: "0.2rem"}}><strong>$ cURL</strong> — a sample request against <code>localhost:30000</code> to confirm the server is up.</li>
    <li style={{marginBottom: "0.2rem"}}><strong>⚙ Env</strong> — edits the placeholders (<code>HOST_IP</code>, <code>PORT</code>, <code>HF_TOKEN</code>) the command and cURL share. Persists in localStorage across cookbooks.</li>
    <li><strong>Verified / Not Verified</strong> badge — green when the <code>(hw, variant, quant, strategy, nodes)</code> combo has been run end-to-end on real hardware; yellow when it has not yet been re-checked.</li>
  </ul>
</div>

## Playground

The Playground layers **SGLang features on top of the Deploy recipe**: it lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing. The base is read live from your Deploy selection — only your overrides change.

For LFM3 the Playground exposes:

- **TP override** — both variants are served at TP=1 in the Deploy cells; TP=2 is verified on the text model.
- **PD disaggregation** and **HiCache** — the generic SGLang prefill/decode split and hierarchical KV cache, not yet run end-to-end on LFM3.

Lines highlighted **green** are added by your overrides; lines with **red strikethrough** were in the base but stripped by an override. When no override differs from the base cell, the playground inherits the base's **Verified** badge; any actual change flips it to **Not Verified** until the new configuration is run end-to-end and submitted back.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

<div style={{fontSize: "0.85em", lineHeight: "1.55", color: "#6b7280", margin: "0.5rem 0 1rem 0"}}>
  <p style={{margin: "0 0 0.3rem 0"}}><strong>Panel controls</strong> reuse <strong>Python / Docker</strong> · <strong>⧉ Copy</strong> · <strong>$ cURL</strong> · <strong>⚙ Env</strong> from the Deploy panel, plus one extra:</p>
  <ul style={{margin: 0, paddingLeft: "1.25rem"}}>
    <li><strong>Submit ↗</strong> — opens a pre-filled GitHub issue so you can land your override combo as a new verified cookbook cell. Shown only while the badge says <strong>Not Verified</strong>; click it once you've actually run the command on your hardware and confirmed it works.</li>
  </ul>
</div>

## 1. Model Introduction

LFM3 is [Liquid AI](https://www.liquid.ai/)'s family of hybrid models for on-device deployment, released under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE). Two checkpoints are covered here: the text-only **LFM3-1.7B** and the vision-language **LFM3-VL-2B**.

The backbone interleaves **gated short convolution blocks** (kernel size 7) with **grouped query attention (GQA) blocks** split between sliding-window attention (256-token window) and full attention. Each convolution block applies input-dependent multiplicative gating around a depthwise short convolution for fast local mixing at low compute and memory cost; the attention blocks handle long-range retrieval. With 24 layers, hidden size 2048, 16 query / 4 key-value heads, head dim 128, tied embeddings, and a 128,000-token vocabulary, the language model totals 1.73B parameters in BF16.

**Key Features:**

- **Hybrid gated short conv + attention layout**: the 1.7B text model stacks 13 gated short convolution blocks, 5 sliding-window GQA blocks, and 6 full-attention GQA blocks. Only the 6 full-attention layers grow KV state with sequence length — the 5 sliding-window layers cap theirs at 256 tokens, and each convolution layer carries a fixed 6-token state — so long-context memory stays small.
- **Long context**: 256K tokens on the text model; 128K on the VL model.
- **Thinking by default**: both checkpoints emit a `<think>...</think>` chain-of-thought before answering. Pass `enable_thinking=False` through the chat template to answer directly (see [§3.2](#3-2-reasoning)).
- **Tool calling**: function calls are emitted in an XML-ish `<tool_call><function=name><parameter=arg>...</parameter></function></tool_call>` format that SGLang parses into standard `message.tool_calls` (see [§3.3](#3-3-tool-calling)).
- **Vision and video (VL)**: LFM3-VL pairs the LFM3 language backbone with a Liquid hybrid convolution/attention vision encoder (14 layers: 8 gated depthwise 2D convolution + 6 bidirectional attention, 16×16 patches with 2D rotary embeddings) and a 25M projector. Images and videos are processed at their **native resolution**; every 2×2 patch block merges into one language token. Videos are sampled at 2 fps with per-frame timestamps — the first LFM release with video input.

**Available Models:**

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "30%"}} />
    <col style={{width: "26%"}} />
    <col style={{width: "12%"}} />
    <col style={{width: "32%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Model</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Parameters</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Context</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong><a href="https://huggingface.co/LiquidAI/LFM3-1.7B">LFM3-1.7B</a></strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1.73B (dense hybrid)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>256K</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Text, reasoning, tool use</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><strong><a href="https://huggingface.co/LiquidAI/LFM3-VL-2B">LFM3-VL-2B</a></strong></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1.73B LM + 164M vision + 25M projector</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>128K</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Vision-language (image + video, native resolution)</td>
    </tr>
  </tbody>
</table>

**Recommended generation (LFM3-1.7B):** `temperature=0.1`, `top_k=50`, `repetition_penalty=1.1` (informational — pass per request; `top_k` and `repetition_penalty` ride in `extra_body`).

**License:** [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE).

**Resources:** [LFM3-1.7B on HuggingFace](https://huggingface.co/LiquidAI/LFM3-1.7B), [LFM3-VL-2B on HuggingFace](https://huggingface.co/LiquidAI/LFM3-VL-2B), [LFM docs](https://docs.liquid.ai/lfm/getting-started/welcome).

## 2. Configuration Tips

Only the **H200** cells are verified end-to-end (2026-10-26, BF16, TP=1, flashinfer). The H100 recipe is vendor-tested in FP16; every other platform in the matrix — B200/B300, GB200/GB300, DGX Spark, and the AMD MI300X/MI325X/MI350X/MI355X family — is an **inferred starting point** that follows the same recipe with the platform adjustments below.

- **Remote code (`--trust-remote-code`)**: the checkpoints carry `auto_map` remote modeling code, so the flag is required — it is already part of the generated command.
- **Attention backend**: the LFM3 model overrides default the attention backend to `flashinfer` on NVIDIA GPUs — no `--attention-backend` flag is needed. For sliding-window correctness the overrides also switch FlashInfer prefill to paged mode automatically (`SGLANG_FLASHINFER_USE_PAGED=True`), because FlashInfer's ragged planner does not apply the 256-token window within prefill chunks. The `triton` backend also passes the model PR's text/vision checks on H100, but FlashInfer is the default and the backend these cells were verified with.
- **AMD (ROCm)**: the generated command pins `--attention-backend aiter` (the LFM3 overrides' default on HIP) with `SGLANG_USE_AITER=1`. The gated-conv blocks take the PyTorch fallback path on ROCm — the CuTe DSL conv kernel is CUDA-only (sm90 / sm100 / sm120-class) — so expect lower conv throughput until a ROCm kernel lands. No published ROCm image carries LFM3 support yet: install SGLang from source and use the **Python** output of the command panel.
- **Reasoning parser (`--reasoning-parser auto`)**: LFM3 wraps its chain-of-thought in `<think>...</think>` tags. The generated command passes `--reasoning-parser auto`, which resolves the `qwen3` parser for this checkpoint and splits thinking into `reasoning_content`. Without a reasoning parser the thinking stays inline in `content`.
- **Tool-call parser (`--tool-call-parser auto`)**: LFM3's `<tool_call><function=...><parameter=...>` format matches the `qwen3_coder` parser's sentinels. The generated command passes `--tool-call-parser auto`, which resolves `qwen3_coder` for this checkpoint (the server log prints the resolved parser), and calls surface as standard `message.tool_calls`. If the log names a different parser, pass `--tool-call-parser qwen3_coder` explicitly.
- **Thinking by default**: the chat template enables thinking (`enable_thinking=true`) unless you opt out, and long thinking traces are normal — a mean of ~1100 reasoning tokens per GSM8K answer was measured, so request-level token budgets must accommodate the chain-of-thought, not just the final answer. Pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to answer directly (the generation prompt then ends with an empty `<think></think>` block).
- **Hardware requirements**: both LFM3 checkpoints run on a single GPU (TP=1); the BF16 weights take under 4 GB, leaving most of GPU memory for KV cache and multimodal features. TP=2 is verified on the text model (Playground override) but not required.
- **No speculative decoding**: the server rejects speculative-decoding configurations for this architecture at startup.
- **Precision**: BF16 (or FP16) checkpoints only — there is no packed-quantization (FP8/FP4) variant.
- **Video input (VL)**: LFM3-VL samples videos at 2 fps, pairs consecutive frames, and prefixes each frame with its timestamp. Pass videos as `video_url` content blocks the same way as images (see [§3.4](#3-4-vision-and-video-input)).

## 3. Advanced Usage

The examples assume a server launched from the Deploy command, which includes both parsers. Outputs are captures from the verified H200 servers; sampling is stochastic, so a repeated request can word things differently, and long reasoning traces are abridged as `…`.

### 3.1 Basic Usage

A single OpenAI-compatible client; the examples in the following sections reuse this setup:

<Accordion title="Client setup (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
```

</Accordion>

### 3.2 Reasoning

Both checkpoints think by default. With the reasoning parser on (`--reasoning-parser auto`, resolving `qwen3`) the thinking process is separated into `reasoning_content` and the final answer stays in `content`:

<Accordion title="Reasoning example (Python)">

```python Example
resp = client.chat.completions.create(
    model="LiquidAI/LFM3-1.7B",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
)
msg = resp.choices[0].message
print("Reasoning:", msg.reasoning_content)
print("Answer:", msg.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: …
Answer: 391
```

</Accordion>

To answer directly, add `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to the request.

### 3.3 Tool Calling

LFM3 writes tool calls as `<tool_call><function=name><parameter=arg>...</parameter></function></tool_call>` blocks. With the tool-call parser on (`--tool-call-parser auto`, resolving `qwen3_coder`) they surface as standard `message.tool_calls`. On a thinking turn the response may put text in `reasoning_content` instead of (or with) `content`, so print both:

<Accordion title="Tool calling example (Python)">

```python Example
resp = client.chat.completions.create(
    model="LiquidAI/LFM3-1.7B",
    messages=[{"role": "user", "content": "What's the weather in Warsaw in celsius?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }],
)
msg = resp.choices[0].message
print("Reasoning:", msg.reasoning_content)
print("Content:", msg.content)
for call in msg.tool_calls or []:
    print(call.function.name, call.function.arguments)
print("finish_reason:", resp.choices[0].finish_reason)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: …
Content: None
get_weather {"city":"Warsaw","unit":"celsius"}
finish_reason: tool_calls
```

</Accordion>

Tool-call deltas are also streamed incrementally when `stream=True`.

### 3.4 Vision and Video Input

LFM3-VL-2B accepts images and videos at native resolution via standard OpenAI multimodal content blocks. URLs and base64 data URIs both work.

<Accordion title="Image example (Python)">

```python Example
import base64

with open("test_pattern.png", "rb") as f:
    data_uri = "data:image/png;base64," + base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="LiquidAI/LFM3-VL-2B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }],
)
print(resp.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

For a synthetic test image (dark blue background, red circle on the left, yellow rectangle on the right, white "LFM3" text along the bottom):

```text Output
The image shows a dark blue flag-like background with a red circle on the left and a yellow square on the right, accompanied by the white text "LFM3" centered along the bottom edge.
```

</Accordion>

Video input uses `video_url` blocks the same way; the server samples the clip at 2 fps and feeds the frames with their timestamps:

<Accordion title="Video example (Python)">

```python Example
import base64

with open("test_clip.mp4", "rb") as f:
    data_uri = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="LiquidAI/LFM3-VL-2B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": data_uri}},
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    }],
)
print(resp.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

For a synthetic test clip (vertical color bars, moving shapes, and an on-screen timer):

```text Output
The video is a short abstract, glitch‑style animation in which bright diagonal streaks of color and pixelated shapes (including a plus/cross made of gray pixels) sweep and shift across a background of vertical rainbow stripes (red, green, yellow, blue, magenta and cyan), while a small digital timer in the top‑left corner counts upward.
```

</Accordion>
