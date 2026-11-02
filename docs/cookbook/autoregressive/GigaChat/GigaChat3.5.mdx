---
title: GigaChat 3.5
description: "Deploy GigaChat 3.5 with SGLang — the 432B-A28B hybrid MLA + GDN Mixture-of-Experts instant model with MTP speculative decoding and tool calling, on H100 and H200 in FP8."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all install methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install).

<Tabs>

<Tab title="Python (pip / uv)">

GigaChat 3.5 support is on `main` ([PR #29189](https://github.com/sgl-project/sglang/pull/29189), merged 2026-09-21) and not yet in a `pip` release — install from source:

```bash Command
pip install --upgrade pip
pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python'
```

Then run the **Python** output of the command panel below.

</Tab>

<Tab title="Docker">

<Note>No released `lmsysorg/sglang` tag carries GigaChat 3.5 yet. The nightly `lmsysorg/sglang:dev` is rebuilt from `main` and carries the model in builds after 2026-09-21, but that path has not been run end-to-end, so the command panel shows only the Python run mode. Run the Python command inside the nightly image; the Docker tab returns to the panel after a verified run.</Note>

```bash Command
docker pull lmsysorg/sglang:dev
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

This page covers the **instant** checkpoint (direct answers, no thinking). For the thinking checkpoint, see [GigaChat 3.5 Reasoning](/cookbook/autoregressive/GigaChat/GigaChat3.5-Reasoning).

Pick your hardware and recipe to generate the launch command. Two serving strategies cover the operating points:

- **Low-Latency** — MTP speculative decoding on, Mamba state pool pinned. Fastest reply per user; pick for chat and agent loops.
- **High-Throughput** — MTP off. At saturation the draft + verify overhead outweighs the speedup, so this is the recipe for batch jobs and many concurrent users.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/ai-sage/gigachat3.5.jsx";
import { benchmarks } from "/src/snippets/configs/ai-sage/gigachat3.5-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations that have been run end-to-end; the Playground lets you turn on additional knobs — the tool-call parser, the MTP preset, parallelism — on top of whichever cell the Deploy panel is currently showing. Any change flips the badge to **Not Verified** until the new configuration is run end-to-end.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

**GigaChat 3.5** is a hybrid Mixture-of-Experts model from GigaChat — **432B** total parameters, **28B** active per token, **256K** context (YaRN, factor 8 over a 32K base), released under the **MIT** license. Its **40** layers interleave two attention types: **10** full-attention **MLA** layers (DeepSeek-style latent attention, every 4th layer) and **30** **GDN** linear-attention layers (Qwen3-Next gated delta-net, constant-size state). Only the MLA layers hold a KV cache; the linear layers keep their state in a Mamba-style pool. The MoE block routes to **8 of 256** experts plus **1** shared expert. The checkpoint ships **2** stacked **MTP** draft heads for speculative decoding, and tool calls are emitted in GigaChat's **GCML** format.

This is the **instant** release: it answers directly, without a thinking block. The separately trained **Reasoning** release always thinks first and ships 3 MTP heads — it has [its own page](/cookbook/autoregressive/GigaChat/GigaChat3.5-Reasoning).

**Resources:** HuggingFace — [GigaChat3.5-432B-A28B](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B) (FP8, served here) · [GigaChat3.5-432B-A28B-bf16](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B-bf16) (BF16, for fine-tuning; no single-node recipe).

## 2. Configuration Tips

**Two pools.** GDN layers keep per-request state in a Mamba-style pool, MLA layers use a paged KV pool; `--mamba-full-memory-ratio` (default 0.9) splits the post-weight budget. On 8×H100 at `--mem-fraction-static 0.8` that is ~51 GB of weights per GPU, 347 state slots and 542k KV tokens. The resolver forces `--mamba-radix-cache-strategy no_buffer` (3 slots per request, overlap scheduler off), so the unpinned server admits 115 concurrent requests.

**MTP needs a pinned pool.** Each running request holds extra state copies under speculative decoding, and the default ratio then admits only a few dozen. The Low-Latency recipe pins `--max-mamba-cache-size 240` with `--max-running-requests 80`; the KV pool drops to 185k tokens. Keep `--mem-fraction-static 0.8` on 80 GB — it is already tight in eager prefill at 16 concurrent 8k-token prompts.

**Capacity levers** (toggles in the [Playground](#playground)). `--mamba-ssm-dtype bfloat16` halves every state slot at no cost in accuracy or speed. `--kv-cache-dtype fp8_e4m3` doubles the KV pool but decodes ~2.3× slower on H100 and cannot be combined with MTP on 80 GB.

**MTP flags.** `--speculative-algorithm EAGLE` uses GigaChat's own NextN heads through the multi-layer EAGLE worker, selected automatically. This checkpoint has **2** heads: `--speculative-num-steps 2`, `--speculative-num-draft-tokens 3`.

**Parsers.** `--tool-call-parser gigachat35` for structured tool calls (§3.1). No reasoning parser, not even `auto`: the checkpoint never emits `</think>`, and the forced `gigachat35` splitter would move the whole answer into `reasoning_content`.

## 3. Advanced Usage

### 3.1 Tool Calling

Enable the `gigachat35` tool-call parser (toggle **Tool Call Parser** in the **Parsers** card of the [Playground above](#playground)) to surface structured calls via `message.tool_calls`. GigaChat emits tool calls in its **GCML** format — a `<｜GCML｜tool_calls>` … `</｜GCML｜tool_calls>` block holding one `<｜GCML｜invoke name="…">` per call — inside the assistant turn; the parser reads them out and strips the block from `content`. `--tool-call-parser auto` also resolves to `gigachat35` for this checkpoint, since its chat template carries the GCML marker.

<Accordion title="Tool Calling Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "The city name"}},
                "required": ["city"],
            },
        },
    }
]

resp = client.chat.completions.create(
    model="ai-sage/GigaChat3.5-432B-A28B",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
)
msg = resp.choices[0].message
print("Content:", msg.content)
print("Tool calls:", msg.tool_calls)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Content:
Tool calls: [ChatCompletionMessageFunctionToolCall(id='call_aa689732d4884654a1f76762', function=Function(arguments='{"city": "Paris"}', name='get_weather'), type='function', index=0)]
```

Appending a `tool` message with `{"city": "Paris", "temp_c": 9, "condition": "light rain"}` and re-requesting returns the final answer with no further tool call:

```text Output
Content: The current weather in Paris is light rain with a temperature of 9°C.
Tool calls: None
```

</Accordion>

<Note>
GigaChat tends to call tools **sequentially** — one hop, then the next — so a multi-tool request may come back with a single call. Drive it with an agent loop that appends each `tool` result and re-requests until `finish_reason` is no longer `tool_calls`. Parallel calls (several invokes in one block) are supported by the format when the model chooses them.
</Note>

### 3.2 MTP (Speculative Decoding)

GigaChat 3.5 ships its own NextN draft heads, so speculative decoding needs no separate draft model: `--speculative-algorithm EAGLE` runs the stacked heads through SGLang's multi-layer EAGLE worker, which is selected automatically for this architecture. The **Low-Latency** recipe has it on; in the [Playground](#playground) the **Speculative** card adds the same preset to any other cell.

```bash Flags
--speculative-algorithm EAGLE \
--speculative-num-steps 2 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 3 \
--max-running-requests 80 \
--max-mamba-cache-size 240
```

- **Steps follow the checkpoint.** This checkpoint has 2 draft heads, so use `--speculative-num-steps 2` and `--speculative-num-draft-tokens 3` (steps + 1); the worker rejects other values at startup.
- **Pin the state pool.** Under MTP every running request holds extra GDN state copies for verification. Without `--max-mamba-cache-size` and `--max-running-requests` the resolver clamps concurrency to a few dozen requests; §2 explains what the pin costs in context. The panel shows an amber callout whenever a speculative command lacks `--max-running-requests`.
- **Where it pays off.** Low concurrency: accepted draft tokens skip target forwards, so single-request decode runs about 2× faster with the output distribution unchanged. At high concurrency the verify work and the smaller KV pool eat the gain, which is why the High-Throughput recipe leaves MTP off. Measured numbers are in the benchmark card above.
