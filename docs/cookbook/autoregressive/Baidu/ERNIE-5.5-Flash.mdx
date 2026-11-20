---
title: ERNIE-5.5-Flash
description: "Deploy ERNIE-5.5-Flash with SGLang — Baidu's 280B-total / 14B-active Mixture-of-Experts model with hybrid HCA + MLA-DSA attention, ECHO speculative decoding and up to 1M context. FP8 and BF16 recipes for B300, GB300, B200, GB200 and H100, with HiCache and PD disaggregation."
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
docker pull lmsysorg/sglang:dev-ernie-55-flash
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick your hardware, precision and strategy to generate the launch command. Every recipe runs on the recommended (smallest) GPU count with SGLang's defaults plus `--max-running-requests 256`; size `--max-running-requests` and `--context-length` to your own workload.

- **Low-Latency**: ECHO speculative decoding (2 draft steps). Fastest replies for a single user or light load.
- **High-Throughput**: attention DP without a draft. Most tokens per second across many concurrent requests.

| GPU | FP8 (~280 GB weights) | BF16 (~560 GB weights) |
| --- | --- | --- |
| B300 / GB300 (288 GB) | 2 GPUs | 4 GPUs |
| B200 (192 GB) | 4 GPUs | 8 GPUs |
| GB200 (192 GB) | 4 GPUs | 2 trays × 4 GPUs |
| H100 (80 GB) | 8 GPUs | 2 nodes × 8 GPUs |

The single-node FP8 recipes are verified on B300, GB300, B200, GB200 and H100, as are the BF16 recipes on B300; the other BF16 recipes follow from the same memory budget and are not yet verified.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/baidu/ernie-5.5-flash.jsx";

<Deployment config={config} />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations the SGLang team has signed off on; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

**ERNIE-5.5-Flash** is Baidu's sparse Mixture-of-Experts language model with **280B total and 14B activated parameters per token**, pre-trained from scratch on 33.5T tokens with a training context of up to 128K. It comes as two models: **ERNIE-5.5-Flash-Base** (FP8 and BF16), taken at the end of pre-training before any mid- or post-training, a starting point for domain adaptation, instruction tuning and reinforcement-learning research, served through the completions endpoint; and **ERNIE-5.5-Flash** (FP8 / FP4 / BF16, coming soon), with reasoning, tool calling and a bundled DSpark draft.

Its 43 layers are 1 dense FFN layer plus 42 Multi-Router Latent MoE layers. Attention is hybrid: 37 layers use Heavily Compressed Attention (HCA, one cached entry per 128 tokens plus a 128-token sliding window, 512-dim heads) and one in seven, 6 layers, keep full-context access through MLA with DSA sparse attention (256-dim heads). Every attention layer adds a low-rank output gate, Virtual Head Attention (a low-rank post-mix across heads) and QK-norm. Each MoE layer has 512 routed experts (top-10) plus 1 shared expert in a 2048-d latent space, selected by several sigmoid routers. Manifold-constrained hyper-connections (expansion rate 4) replace the residual stream, a MultiMax head produces the logits, and ECHO reuses the final layer as a one-step speculative draft. It is released under the Apache-2.0 license.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Model</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Weights</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Use</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}><strong>ERNIE-5.5-Flash-Base</strong></td>
      <td style={{padding: "9px 12px"}}><a href="https://huggingface.co/baidu/ERNIE-5.5-Flash-Base">FP8</a> / <a href="https://huggingface.co/baidu/ERNIE-5.5-Flash-Base-BF16">BF16</a></td>
      <td style={{padding: "9px 12px"}}>Base model, completions. Released</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong>ERNIE-5.5-Flash</strong></td>
      <td style={{padding: "9px 12px"}}>FP8 / FP4 / BF16</td>
      <td style={{padding: "9px 12px"}}>Chat, reasoning, tool calling, DSpark. Coming soon</td>
    </tr>
  </tbody>
</table>

All checkpoints share the architecture above: 280B total / 14B activated parameters.

Total parameters include the input embedding and the output head; activated parameters count the output head but not the embedding lookup. Both exclude the 0.36B ECHO parameters. The model was trained with a context of up to 128K tokens (131,072); fine-tuning and inference support up to 1M (1,048,576); pass `--context-length 1048576` to use it.

**Sampling defaults:** `temperature=1.0`, `top_p=0.95` (from the checkpoint's `generation_config.json`); pass `temperature=0` for greedy decoding.

**Resources:** [Hugging Face](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base), [ModelScope](https://modelscope.cn/models/baidu/ERNIE-5.5-Flash-Base), [ERNIE-5.5-Flash-Base technical blog](https://ernie.baidu.com/release/ernie-5-5-flash-base/).

## 2. Configuration Tips

- **What gets set automatically.** The model selects the `ernie55` attention backend, `--page-size 256`, and `--swa-full-tokens-ratio 0.1` on its own. `--kv-cache-dtype` defaults to `fp8_e4m3`; `bfloat16` is the only other supported value.
- **Parallelism.** Attention TP (`tp / attn-dp-size / attn-cp-size`) must divide 8, the number of output groups. The high-throughput recipes use attention TP 1 (`--attn-dp-size` equal to `--tp`); the low-latency recipes use plain TP, except BF16 on H100, where TP 16 keeps `--attn-dp-size 2` so attention TP stays at 8. Expert parallelism is optional: `--ep N --moe-a2a-backend deepep` routes experts over DeepEP, which needs `--ep 4` or more (at `--ep 2` each rank would hold 256 experts, past DeepEP's per-rank limit). Prefill context parallelism (`--attn-cp-size N --enable-prefill-cp --cp-strategy interleave`) and pipeline parallelism also work; under PP use `--moe-a2a-backend none`.
- **Long context.** The released config allows 1,048,576 tokens. Pass `--context-length 1048576`, and lower `--max-running-requests` (to single digits on 2 × B300) and `--cuda-graph-max-bs-decode`: each rank's KV pool must hold whole 1M-token requests, and a 1M prefill needs tens of GB of activation memory on top.
- **DeepEP.** When you add `--moe-a2a-backend deepep`, decode runs DeepEP low-latency inside a CUDA graph and each rank dispatches at most `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` tokens per step (default 128, at most 1024). Keep `--cuda-graph-max-bs-decode` × the tokens per request (1, or `--speculative-num-steps` + 1 with ECHO, or the block size + 1 with DSpark) at or below that value.

## 3. Advanced Usage

### 3.1 Completions (Base)

As a base model, ERNIE-5.5-Flash-Base continues text: send prompts to `/v1/completions`. It has no chat, reasoning or tool-call training, so chat requests, `--reasoning-parser` and `--tool-call-parser` do not apply. For few-shot use, put the examples in the prompt and stop at the end of one answer.

<Accordion title="Completions Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
resp = client.completions.create(
    model="baidu/ERNIE-5.5-Flash-Base",
    prompt="Question: What is 15% of 240?\nAnswer:",
    max_tokens=64,
    temperature=0,
    stop=["\n"],
)
print(resp.choices[0].text)
```

</Accordion>

<Accordion title="Example Output">

```text Output
 36
```

</Accordion>

### 3.2 Reasoning (ERNIE-5.5-Flash, coming soon)

ERNIE-5.5-Flash thinks before it answers. Start the server with `--reasoning-parser ernie55` (the **Reasoning Parser** toggle in the [Playground above](#playground)); thinking then lands in `reasoning_content` and the answer in `content`. Thinking is on by default; to turn it off for a request, pass `enable_thinking: false` through `chat_template_kwargs` (the chat template reads `enable_thinking`, not `thinking`).

<Accordion title="Reasoning Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
messages = [{"role": "user", "content": "What is 15% of 240?"}]

# Thinking is on by default.
msg = client.chat.completions.create(
    model="baidu/ERNIE-5.5-Flash", messages=messages
).choices[0].message
print("Reasoning:", msg.reasoning_content)
print("Answer:", msg.content)

# Turn it off for one request.
msg = client.chat.completions.create(
    model="baidu/ERNIE-5.5-Flash",
    messages=messages,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
).choices[0].message
print("Answer:", msg.content)
```

</Accordion>

### 3.3 Tool Calling (ERNIE-5.5-Flash, coming soon)

Start the server with `--tool-call-parser ernie55` (the **Tool Call Parser** toggle in the [Playground above](#playground)). The parser reads the model's `<ernie:tool_call>` blocks, types each argument from the tool's JSON schema, and returns structured calls in `message.tool_calls`; streaming emits the arguments as they are generated. Use it together with `--reasoning-parser ernie55`, since the model reasons before calling a tool.

<Accordion title="Tool Calling Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
resp = client.chat.completions.create(
    model="baidu/ERNIE-5.5-Flash",
    messages=[{"role": "user", "content": "What is the weather in Beijing?"}],
    tools=tools,
)
for call in resp.choices[0].message.tool_calls or []:
    print("Tool call:", call.function.name, call.function.arguments)
```

</Accordion>

### 3.4 HiCache (Hierarchical KV Caching)

Long shared prefixes (system prompts, agent histories) can spill from GPU memory to host memory, and optionally to Mooncake, instead of being recomputed. Add `--enable-hierarchical-cache --hicache-ratio 2` for a host tier twice the size of the GPU KV pool (only `--hicache-ratio` is supported, not `--hicache-size`). Add `--hicache-storage-backend mooncake` for an L3 tier, or set it from the **HiCache** card in the [Playground above](#playground).

### 3.5 Speculative Decoding

**ECHO.** The checkpoint carries one ECHO layer after the main stack: the backbone's own final layer, trained to predict future tokens, doubles as the draft. Turn it on with `--speculative-algorithm EAGLE` (or `NEXTN`; the **ECHO** options in the [Playground above](#playground)). The layer was trained for one future token and can draft several steps (`--speculative-num-steps`). The attention backend supports chain drafts only, so keep `--speculative-eagle-topk 1`; `--speculative-num-draft-tokens` then follows as `--speculative-num-steps` + 1 (the defaults are 1, 1 and 2). Measured on GSM8K at batch size 1 with greedy decoding, the mean accept length is 1.93 / 2.28 / 2.36 tokens per step for 1 / 2 / 3 draft steps, with unchanged accuracy; 2 steps take most of the gain.

**DSpark (ERNIE-5.5-Flash, coming soon).** The ERNIE-5.5-Flash checkpoint bundles a DSpark draft, served with `--speculative-algorithm DSPARK` together with `--enable-dp-lm-head` (the **DSpark** option in the [Playground above](#playground), with `--model-path` set to `baidu/ERNIE-5.5-Flash`). The draft path defaults to the model path and the block size comes from the checkpoint. DSpark under attention DP fails to start without `--enable-dp-lm-head`; without attention DP the flag is ignored, so always pass it. The Base checkpoints carry no DSpark draft.

```bash Command
sglang serve --model-path baidu/ERNIE-5.5-Flash \
  --tp 2 --context-length 131072 --max-running-requests 256 \
  --reasoning-parser ernie55 --tool-call-parser ernie55 \
  --speculative-algorithm DSPARK --enable-dp-lm-head \
  --host 0.0.0.0 --port 30000
```

### 3.6 PD Disaggregation

Pick the **Prefill role** / **Decode role** in the **PD Disagg** card of the [Playground above](#playground) (Mooncake transfer). With speculative decoding, pass the same speculative flags to the prefill and decode servers. With DeepEP, run decode with `--deepep-mode auto` (the default) so it replays CUDA graphs; under `--deepep-mode normal` decode runs eagerly.

<Warning>
Parallel sampling (`n` > 1 in one request) currently hangs under PD disaggregation: the `n` sub-requests share one bootstrap room. Send `n` = 1 requests to a PD deployment.
</Warning>
