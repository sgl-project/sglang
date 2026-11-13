---
title: Step 5 Preview
description: "Deploy StepFun's Step 5 Preview with SGLang — a 600B-parameter (27B active) Mixture-of-Experts model for agentic work with region-sparse attention, three-layer MTP and 1M context, on NVIDIA Blackwell and Hopper."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install). The two paths below match the **Python / Docker** toggle in the command panel.

<Tabs>

<Tab title="Python (pip / uv)">

Step 5 Preview support has **not shipped in an SGLang release yet**, so a stock `pip install sglang` cannot serve it. Use the preview Docker image.

</Tab>

<Tab title="Docker">

Every recipe on this page uses the preview image `lmsysorg/sglang:step-5-preview` (B300 and H200); the command panel picks it automatically. The image already includes the FlashInfer patches Step 5 Preview needs.

```bash Command
docker pull lmsysorg/sglang:step-5-preview

docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<your-hf-token>" \
    --ipc=host \
    lmsysorg/sglang:step-5-preview \
    sglang serve <use args below>
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker).

</Tab>

</Tabs>

</Accordion>

Pick your hardware and parallelism, then the model, MoE backend and serving options. Recommended starting points:

- **Lowest latency** (Blackwell) — TP4 with the FlashInfer TRT-LLM MoE on four B300 GPUs. Fastest reply for a single user.
- **Multi-user serving** (Blackwell) — TP4 with MegaMoE, which speeds up prefill and decode once several requests share a batch.
- **Highest throughput** (Blackwell) — ATTN DP2: TP8 with DP attention 2 and EP8 on eight B300 GPUs, with either MoE backend. Most tokens per second across many users.
- **Hopper** — EP8: TP8 + EP8 on eight H200 GPUs with FA3 attention and Triton MoE. The FP8 weights do not fit four H200 GPUs. ATTN DP2 is also available on Hopper (TP8 with DP attention 2 and EP8, `--mem-fraction-static 0.85`).

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/stepfun-ai/step-5-preview.jsx";

<Deployment config={config} />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations the SGLang team has signed off on; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

**Step 5 Preview** is StepFun's flagship model for agentic work, with particular strength in software engineering, professional knowledge work and finance. It is a sparse Mixture-of-Experts model with 600B total parameters and 27B active parameters. It supports a 1M-token context, takes image input through a built-in vision encoder, and ships three MTP layers for speculative decoding.

**Architecture.** The language model has 92 layers. In each MoE layer, a token is routed to 8 of 352 experts, alongside one shared expert. Attention repeats a four-layer pattern: three sliding-window layers that attend to the most recent 512 tokens, then one sparse-attention layer.

**Sparse attention.** The sparse layers split the KV history into 8-token regions. A lightweight indexer scores a summary of each region and selects up to 512 historical regions; attention then reads the token-level keys and values of those regions plus the current region, with causal masking.

**Precision.** Routed-expert weights are FP8 (E4M3, 128×128 block scales) except in the last three MoE layers, which stay BF16. SGLang runs every layer at its released precision.

**Reasoning.** Step 5 Preview always thinks before answering. `reasoning_effort` sets the depth: `low`, `medium` or `high`; when unset, it defaults to `high`.

**Recommended generation:** `temperature=1.0`, `top_p=0.95`, `top_k=0` (StepFun's recommended settings; informational — do not hardcode them in application code).

**License:** [StepFun Model License 1.0](https://huggingface.co/stepfun-ai/Step-5-Preview/blob/main/LICENSE.txt).

**Resources:** [HuggingFace (FP8)](https://huggingface.co/stepfun-ai/Step-5-Preview) · [HuggingFace (NVFP4)](https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4) · [Model page](https://www.stepfun.com/step-5-preview) · [Documentation](https://platform.stepfun.ai/docs/en/guides/models/step-5-preview).

## 2. Configuration Tips

**MoE runner.** On B300, the default runner is the FlashInfer TRT-LLM MoE. Picking **MegaMoE** in the **MoE** row adds `--moe-a2a-backend megamoe`, which runs the FP8 routed experts on MegaMoE; the BF16 routed experts in the last three MoE layers stay on TRT-LLM. MegaMoE also works with ATTN DP2. On H200, the recipe uses the Triton MoE for all expert layers, including the three BF16 ones, because the TRT-LLM MoE kernels are Blackwell-only.

**NVFP4 checkpoint.** Pick **NVFP4** in the **Model** row to serve `stepfun-ai/Step-5-Preview-NVFP4` on Blackwell. Its routed experts are NVFP4 (the last three MoE layers stay BF16) and run on the FlashInfer FP4 MoE kernels; the smaller weights leave more GPU memory for the KV cache. It runs with TP4, EP4 and ATTN DP2, does not support MegaMoE, and is not available on H200.

**Attention backend.** On Blackwell, Step 5 Preview selects FA4 automatically. On H200, keep `--attention-backend fa3` from the recipe.

**KV cache.** Every recipe uses BF16 KV cache. On both Blackwell and Hopper, pick **FP8** in the **KV Cache** row of the command panel for `--kv-cache-dtype fp8_e4m3` (full-attention layers only): it increases KV capacity but is lossy, so re-verify accuracy before using it.

**Prefill CUDA graph.** Step 5 Preview runs prefill on Breakable CUDA Graph by default, including with ATTN DP2, so the commands do not set it. ATTN DP2 runs TP8 with DP attention 2 and EP8; with DP attention on, the vision encoder and projector use replicated weights.

**FlashInfer patches.** The TRT-LLM MoE path needs Step 5 Preview's FlashInfer patches. They are already in `lmsysorg/sglang:step-5-preview`; for a custom build, see the README under `docker/patches/step5` in the SGLang repository.

**First start.** The first launch compiles FlashInfer modules (about 15 minutes on B300); later launches take about 6 minutes. Persist `FLASHINFER_WORKSPACE_BASE` and `~/.cache/sglang` across restarts.

**HiCache.** Turn it on in the **HiCache** row to keep evicted prefixes in host memory. Size `--hicache-ratio` to the host memory available per rank; larger ratios need proportionally more host RAM.

**Per-platform notes:**

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "16%"}} />
    <col style={{width: "30%"}} />
    <col style={{width: "54%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Platform</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Topology</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr style={{background: "rgba(255,255,255,0.02)"}}>
      <td style={{padding: "9px 12px"}}>B300 1×4</td>
      <td style={{padding: "9px 12px"}}>TP4, or TP4 + EP4</td>
      <td style={{padding: "9px 12px"}}>TRT-LLM MoE or MegaMoE; FA4 attention</td>
    </tr>
    <tr style={{background: "rgba(255,255,255,0.05)"}}>
      <td style={{padding: "9px 12px"}}>B300 1×8</td>
      <td style={{padding: "9px 12px"}}>TP8 + DP attention 2 + EP8</td>
      <td style={{padding: "9px 12px"}}>TRT-LLM MoE or MegaMoE; FA4 attention</td>
    </tr>
    <tr style={{background: "rgba(255,255,255,0.02)"}}>
      <td style={{padding: "9px 12px"}}>H200 1×8</td>
      <td style={{padding: "9px 12px"}}>TP8 + EP8, or TP8 + DP attention 2 + EP8</td>
      <td style={{padding: "9px 12px"}}>FA3 attention, Triton MoE; BF16 or FP8 KV cache</td>
    </tr>
  </tbody>
</table>

## 3. Advanced Usage

### 3.1 Reasoning

The `step5` reasoning parser (already in every recipe) returns the thinking in `reasoning_content` and the answer in `content`. Pass `reasoning_effort` (`low`, `medium` or `high`) per request to trade depth for latency.

<Accordion title="Reasoning Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="stepfun-ai/Step-5-Preview",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    reasoning_effort="low",
)
msg = resp.choices[0].message
print("Reasoning:", msg.reasoning_content)
print("Answer:", msg.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: We need answer simple. Need maybe compute 36. Need provide perhaps concise. Use reasoning low but final. 15% of 240 = 0.15*240=36. final.
Answer: 36
```

</Accordion>

### 3.2 Tool Calling

The `step5` tool-call parser (already in every recipe) returns structured tool calls in `message.tool_calls`.

<Accordion title="Tool Calling Example (Python)">

```python Example
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]
messages = [{"role": "user", "content": "What's the weather like in Beijing right now?"}]

resp = client.chat.completions.create(
    model="stepfun-ai/Step-5-Preview", messages=messages, tools=tools,
)
msg = resp.choices[0].message
for call in msg.tool_calls:
    print("Tool call:", call.function.name, call.function.arguments)

call = msg.tool_calls[0]
messages.append(msg)
messages.append({
    "role": "tool",
    "tool_call_id": call.id,
    "content": json.dumps({"city": "Beijing", "temperature_c": 18, "condition": "Sunny"}),
})
final = client.chat.completions.create(
    model="stepfun-ai/Step-5-Preview", messages=messages, tools=tools,
)
print("Reasoning:", final.choices[0].message.reasoning_content)
print("Answer:", final.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Tool call: get_weather {"city": "Beijing"}
Reasoning: The weather in Beijing is 18°C and sunny.
Answer: The weather in Beijing right now is **sunny** with a temperature of **18°C** — a pleasant day! ☀️
```

</Accordion>

### 3.3 Image Input

<Accordion title="Image Input Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="stepfun-ai/Step-5-Preview",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"}},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }],
)
print(resp.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
A man in a yellow hoodie irons a blue shirt on an ironing board attached to the back of a yellow SUV on a city street, with a yellow taxi behind him.
```

</Accordion>

### 3.4 Speculative Decoding (MTP)

Every recipe enables multi-layer MTP with the checkpoint's three MTP layers (`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-multi-layer-eagle`). Each verification round can advance generation by up to four tokens: three accepted draft tokens plus one target-model token. To turn MTP off, pick **Off** in the **Spec Decode** row of the command panel.

### 3.5 HiCache (Hierarchical KV Caching)

For long multi-turn sessions, turn on **HiCache** in the command panel to keep evicted prefixes in host memory (L2). To add an L3 store such as Mooncake, pick it in the **Hierarchical KV Cache** card of the [Playground above](#playground). See the HiCache note in [Configuration Tips](#2-configuration-tips) for sizing.

### 3.6 PD Disaggregation

Step 5 Preview supports prefill/decode disaggregation over Mooncake, including the MTP draft KV cache. Pick the **Prefill role** and **Decode role** in the **PD Disaggregation** card of the [Playground above](#playground) to generate both commands and the router command.
