---
title: DaisyRose-27B
description: "Deploy DaisyRose-27B with SGLang — RadixArk's reinforcement-learning post-train of Qwen3.8-27B, with BF16/FP8/NVFP4 checkpoints and DSpark speculative decoding, single-GPU on H200, GB300, RTX PRO 6000, RTX 5090 and DGX Spark."
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
docker pull lmsysorg/sglang:latest
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick your card + checkpoint precision to generate the launch command. The model runs single-GPU on every supported card. The panel opens on the configuration we serve and evaluate DaisyRose-27B with: DSpark speculative decoding and the High-Throughput strategy.

<Note>
`--mamba-full-memory-ratio` is the one sizing flag that matters for throughput
on hybrid GDN models: the default (0.9) over-provisions the KV pool and silently
clamps concurrency. Set your average request length in the
[Mamba ratio calculator](#mamba-ratio-calculator) below; the computed value is
pinned into the command.
</Note>

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/RadixArk/daisyrose-27b.jsx";
import { Qwen38MambaRatioCalculator } from "/src/snippets/_qwen38_mamba_ratio_calculator.jsx";

<Deployment config={config} />

<Note>
  DaisyRose-27B has the same architecture and tensor shapes as Qwen3.8-27B, so
  every cell is the [Qwen3.8-27B](/cookbook/autoregressive/Qwen/Qwen3.8-27B)
  recipe for the same card and precision. The GB300 cells with DSPARK are
  measured with DaisyRose-27B weights on **v0.5.23**: all three checkpoints
  serve text, tool-calling and image requests, and DSpark's draft is accepted
  for 3.4-3.5 tokens per step in long-context thinking workloads. The other
  cells have not been re-measured with DaisyRose-27B weights and carry the
  unverified badge.
</Note>

### Mamba ratio calculator

<Qwen38MambaRatioCalculator />

<Accordion title="How --mamba-full-memory-ratio is calculated">

Hybrid GDN models split post-weight memory into a worst-case-reserved **GDN
state pool** (sets the concurrency ceiling) and a paged **attention KV pool**,
divided by `--mamba-full-memory-ratio`. Every parameter below except `L` and the
target concurrency is read live from the Deploy panel and Playground selection;
the balanced value is the per-request cost ratio:

```text Formula
ratio = (S + D) x state_bytes / (L x kv_bytes_per_token)
```

- `S` — state slots per running request: `extra_buffer=5` (default),
  `extra_buffer_lazy=4`, `no_buffer=3`, disabled radix cache `=1`. For the two
  `extra_buffer` strategies, `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` frees one
  slot, and `extra_buffer` frees one more with the overlap scheduler off; the
  calculator reads both knobs.
- `D` — verify intermediate states under speculative decoding:
  `--speculative-num-draft-tokens` for EAGLE/MTP (4 at the recommended 3/1/4)
  and for DFLASH (8, DFlash2's block size);
  `--speculative-dspark-block-size + 1` for DSPARK, where the block size falls
  back to the draft checkpoint's `block_size` when the flag is omitted (7 for
  `RadixArk/Qwen3.8-27B-DSpark`, so `D = 8`); 0 with speculation off or with
  `--enable-linear-replayssm-spec`, which keeps the verify intermediates on a
  fixed ring instead of per-request slots.
- `state_bytes` — one state slot, from the fixed geometry
  (48 GDN layers x 48 heads x 128 x 128 at `--mamba-ssm-dtype`, plus bf16 conv
  state): 153.9 MB at fp32, 78.4 MB at bf16.
- `kv_bytes_per_token` — 16 attention layers x GQA 4 x 256 x K+V:
  32.8 KB at fp8, 65.5 KB at bf16.
- `L` — average total request length in tokens: input + output.

`--max-mamba-cache-size = target_concurrency x S` is the equivalent explicit
pin and overrides the ratio; the calculator emits it alongside. `D` is not a
term here: the engine divides the state pool by `S` alone and sizes the
speculative verify buffer separately, so folding `D` into the pin would
over-provision the pool. After boot, verify with the `max_running_requests`
line in the server log — it should not be capped below your target concurrency.

</Accordion>

## Playground

The Playground is where you experiment with **SGLang features beyond the recipes above**. The Deploy panel emits this model's documented launch recipes; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

**DaisyRose-27B** is RadixArk's first model: Qwen3.8-27B post-trained with
reinforcement learning only, without supervised fine-tuning or distillation, on
about 1,200 long-horizon terminal tasks. It thinks before it answers and is
aimed at agentic coding, terminal and software-engineering work. Evaluation
results are on the [model card](https://huggingface.co/RadixArk/DaisyRose-27B).

It keeps the base model's architecture, tokenizer and chat template: a dense
hybrid Gated Delta Networks (GDN) vision-language model with 48 linear-attention
and 16 full-attention layers, an in-checkpoint MTP head, native image input and
a 262,144-token context. The vision encoder and the MTP head are the base
model's, unchanged. It is a drop-in replacement for Qwen3.8-27B, and the
[Qwen3.8-27B page](/cookbook/autoregressive/Qwen/Qwen3.8-27B) covers the
architecture and platform tuning in depth. License: Apache 2.0.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "30%"}} />
    <col style={{width: "36%"}} />
    <col style={{width: "34%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Model</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Quantization</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>DaisyRose-27B</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>BF16</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/RadixArk/DaisyRose-27B">RadixArk/DaisyRose-27B</a></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>DaisyRose-27B-FP8</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>FP8 (blockwise)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><a href="https://huggingface.co/RadixArk/DaisyRose-27B-FP8">RadixArk/DaisyRose-27B-FP8</a></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>DaisyRose-27B-NVFP4</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>NVFP4 W4A4 + FP8 projections, `lm_head` packed to FP4</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/RadixArk/DaisyRose-27B-NVFP4">RadixArk/DaisyRose-27B-NVFP4</a></td>
    </tr>
  </tbody>
</table>

The NVFP4 checkpoint is a NVIDIA Model Optimizer export with the same
quantized-layer map as `nvidia/Qwen3.8-27B-NVFP4`: FP8 attention and GDN
projections, NVFP4 MLPs, and an FP4-packed `lm_head`. It needs FP4 tensor cores,
so it has no H200 cell.

## 2. Configuration Tips

- **DSpark**: the draft model is a separate checkpoint trained on Qwen3.8-27B,
  and it carries over to DaisyRose-27B — add `--speculative-algorithm DSPARK
  --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark`. DSpark does not
  take `--speculative-num-draft-tokens`; its verify window comes from the draft
  checkpoint (D = 8), which the [calculator](#mamba-ratio-calculator) folds into
  `--mamba-full-memory-ratio`.
- **MTP**: `--speculative-algorithm EAGLE --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` uses the
  in-checkpoint MTP head and needs no extra download.
- **Hardware fit**: BF16 weights ~52GB, FP8 ~29GB, NVFP4 ~23GB. On a 32GB RTX
  5090 only NVFP4 is serviceable, and the recipe pins a single stream.
- **SM120/SM121 (RTX PRO 6000, RTX 5090, DGX Spark)**: use
  `--attention-backend flashinfer`. On DGX Spark keep `--mem-fraction-static
  0.80`; the 128GB is shared with the host.
- **Parsers**: every recipe carries `--reasoning-parser qwen3
  --tool-call-parser qwen3_coder`. Without them a client receives thinking and
  tool calls as raw text.
- Platform notes, GDN state precision and prefill-chunk guidance are the same
  as for the base model: see
  [Qwen3.8-27B → Configuration Tips](/cookbook/autoregressive/Qwen/Qwen3.8-27B#2-configuration-tips).

## 3. Advanced Usage

The examples use the `openai` Python client against the server started above,
with `--served-model-name DaisyRose-27B` added to the launch command.

### 3.1 Reasoning

Thinking is on by default. The `qwen3` reasoning parser returns it in
`reasoning_content` and the answer in `content`.

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="DaisyRose-27B",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
)
print("Reasoning:", response.choices[0].message.reasoning_content)
print("Answer:", response.choices[0].message.content)
```

**Output Example:**

```text Output
Reasoning: We need answer simple multiplication. Need final. Compute 17*23 = 17*20 +17*3=340+51=391.

Answer:

391
```

Turn thinking off per request with `enable_thinking`:

```python Example
response = client.chat.completions.create(
    model="DaisyRose-27B",
    messages=[{"role": "user", "content": "Name a prime above 100."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

**Output Example:**

```text Output
101
```

### 3.2 Tool Calling

```python Example
tools = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Current local time for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }},
]

response = client.chat.completions.create(
    model="DaisyRose-27B",
    messages=[{"role": "user", "content": "Weather and time in Paris?"}],
    tools=tools,
)
for call in response.choices[0].message.tool_calls:
    print(call.function.name, call.function.arguments)
```

**Output Example:**

```text Output
get_weather {"city": "Paris"}
get_time {"city": "Paris"}
```
