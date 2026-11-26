---
title: GigaChat 4
description: "Deploy GigaChat 4 with SGLang — the 708B-A41B hybrid DeepSeek Sparse Attention (DSA) + Gated DeltaNet (GDN) Mixture-of-Experts model. Deployment recipes and OpenAI-compatible API examples cover instant and reasoning modes, tool calling, and multi-turn conversations.\n\nThe [GigaChat blog](https://giga.chat/blog/en) provides live updates on GigaChat 4 training progress and benchmark results, along with technical posts covering training details and multimodal models."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all installation methods and hardware platforms, see the [SGLang installation guide](/docs/get-started/install).

</Accordion>

GigaChat 4 supports both **instant** and **reasoning** modes in the same checkpoint, selected per request. Pick your hardware and serving strategy to generate the launch command:

- **Low-Latency** — enables speculative decoding with the model’s built-in Multi-Token Prediction (MTP) layer to reduce time between output tokens. Start here for chat and agent loops.
- **High-Throughput** — speculative decoding off, for tuning workloads with many concurrent requests. Compare it with MTP on your workload before choosing a serving strategy.
- **DFlash2** — speculative decoding with a separate BF16 draft and a learned path selector. Download the checkpoint's `dflash_draft/` directory before launching; **DFlash2 draft path** defaults to `dflash_draft`.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/ai-sage/gigachat4.jsx";
import { benchmarks } from "/src/snippets/configs/ai-sage/gigachat4-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

<Note>
The H100 recipes use two nodes with eight H100 80 GB GPUs each and tensor parallelism across all 16 GPUs (TP16). Run the generated command on both nodes, with rank 0 on the head node and rank 1 on the worker. Send API requests to the head node. The single-node H200 recipes use eight H200 141 GB GPUs and TP8; they have not been validated.
</Note>

The serving measurements use 8192 input tokens and 1024 output tokens, greedy decoding, and client concurrency 1, 16, or 64. TTFT and TPOT show P50; throughput per GPU counts input and output tokens. Each run follows four warmup requests and a cache flush. Concurrency includes queueing when requests exceed cache capacity.

The H100 recipes retain EP1. In the 8192/1024 comparison at concurrency 16 and 64, one run per configuration with EP16 showed no output-throughput improvement over EP1 for Off, MTP, or DFlash2.

## Playground

The Playground lets you change parsers, parallelism, GDN state precision, and speculative decoding on top of the selected deployment recipe. Changing the recipe marks the resulting configuration **Not Verified** until it has been run end-to-end.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model introduction

**GigaChat 4** is a hybrid Mixture-of-Experts model — **708B** total parameters and **41B** active per token. Its **64** layers interleave **16 DSA** layers, every fourth layer, with **48 GDN** linear-attention layers. **DeepSeek Sparse Attention (DSA)** combines Multi-head Latent Attention (MLA) with an indexer that selects relevant tokens. **Gated DeltaNet (GDN)** uses a recurrent state instead of a growing KV cache. The MoE block routes to **8 of 256** experts plus **1** shared expert.

The model uses gated normalization, attention gates, SwiGLU clipping at **10**, and final-logit soft-capping at **30**. Its configured context window is **1,048,576 tokens**, using YaRN with factor 32 over a 32K base. The context a server can admit also depends on its available cache memory.

The checkpoint includes **one MTP layer**, reused recurrently for speculative decoding. DFlash2 uses a separately trained draft. Unlike the separate instant and reasoning releases of [GigaChat 3.5](/cookbook/autoregressive/GigaChat/GigaChat3.5), GigaChat 4 switches between direct answers and reasoning on each request. Tool calls use GigaChat's **GCML** format in both modes.

**Resources:** [GigaChat4-708B-A41B on Hugging Face](https://huggingface.co/ai-sage/GigaChat4-708B-A41B) · [SGLang integration](https://github.com/sgl-project/sglang/pull/50613) · [GigaChat 3.5 Reasoning cookbook](/cookbook/autoregressive/GigaChat/GigaChat3.5-Reasoning).

## 2. Configuration tips

**Weight memory.** The recipes serve an FP8 target. Its weights exceed the combined capacity of eight H100 80 GB GPUs, so the H100 recipes use two nodes and TP16. Keep the checkpoint accessible at the same path on both nodes, or download the same Hugging Face revision on each. Eight H200 141 GB GPUs provide more memory per node; the TP8 recipes remain unverified.

**Two cache pools.** DSA layers use a paged attention KV pool; GDN layers keep per-request recurrent and convolution states in a Mamba-style pool. `--mamba-full-memory-ratio` balances the two pools, and `--max-mamba-cache-size` sets an explicit state-pool size. Speculative decoding needs additional GDN state buffers for verification. Tune the state pool together with concurrency rather than copying the GigaChat 3.5 pool size.

**Context and concurrency.** The recipes set `--max-running-requests 64` and `--cuda-graph-max-bs-decode 64`. The first caps concurrent running requests; the second controls decode CUDA graph capture. They leave `--context-length`, `--max-total-tokens`, and `--chunked-prefill-size` to SGLang's defaults and memory planning. A configured 1M-token window does not guarantee enough cache memory to admit a request of that size: check the startup log for `context_len` and `max_total_num_tokens`. Set a smaller `--context-length` if your application does not need the full window.

**Parsers.** The recipes enable `--reasoning-parser auto` and `--tool-call-parser auto`, which resolve to `gigachat4` for this architecture. You can also select `gigachat4` explicitly for both. Keep the reasoning parser enabled for instant requests too; it handles both modes and returns thinking separately from the answer.

**Draft precision.** MTP loads the bundled NextN layer automatically. DFlash2 needs a separate draft path and `--speculative-draft-model-quantization unquant` to keep the MLA draft in BF16 while the target stays FP8. The draft's attention backend is configured independently; the DFlash2 recipe uses `flashinfer`.

## 3. Advanced usage

The examples use the [OpenAI-compatible API](/docs/basic_usage/openai_api_completions). Install the client with `pip install openai`. For a two-node deployment, replace `localhost` with the head node's address when running the client elsewhere.

### 3.1 Instant and reasoning modes

Set `chat_template_kwargs.reasoning` on each request: `false` for a direct answer, `true` for reasoning followed by the answer. The checkpoint's chat template enables reasoning by default when this argument is omitted. With the `gigachat4` reasoning parser enabled, thinking is returned in `message.reasoning_content` and the answer in `message.content`.

<Accordion title="Instant and reasoning example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
model = "ai-sage/GigaChat4-708B-A41B"

for reasoning in (False, True):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 17 times 24?"}],
        temperature=1.0,
        top_p=1.0,
        max_tokens=4096,
        extra_body={
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"reasoning": reasoning},
        },
    )
    message = response.choices[0].message
    print("Reasoning enabled:", reasoning)
    print("Reasoning:", getattr(message, "reasoning_content", None))
    print("Answer:", message.content)
```

</Accordion>

**Sampling.** Set `temperature` and `top_p` in the client request; SGLang's `repetition_penalty` goes in `extra_body`, alongside `chat_template_kwargs`:

```python Sampling parameters
# Reuse client and model from the previous example.
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Write a short story about a lighthouse."}],
    temperature=1.0,
    top_p=1.0,
    seed=42,
    max_tokens=4096,
    extra_body={
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"reasoning": False},
    },
)
print(response.choices[0].message.content)
```

<Note>
`max_tokens` covers both reasoning and the final answer. An empty `content` with `finish_reason: "length"` can mean that generation stopped before the reasoning block closed; increase the limit for longer reasoning tasks.
</Note>

### 3.2 Multi-turn conversations and tool calling

Send the conversation history with each request. Store assistant thinking in `reasoning_content` on the same assistant message as its answer; do not create a separate `reasoning` role. The template argument `drop_history_thinking` can remove thinking from earlier user turns when rendering the history.

<Accordion title="Multi-turn example without tools (Python)">

```python Example
# Reuse client and model from the first example.
messages = []
for prompt in (
    "Remember that my project is named Cedar. Reply briefly.",
    "What is my project called?",
):
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=4096,
        extra_body={
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"reasoning": True},
        },
    )
    message = response.choices[0].message
    print("Reasoning:", getattr(message, "reasoning_content", None))
    print("Answer:", message.content)
    messages.append(message.model_dump(
        include={"role", "content", "reasoning_content", "tool_calls"},
        exclude_none=True,
    ))
```

</Accordion>

The `gigachat4` tool-call parser converts GCML blocks into OpenAI `message.tool_calls` and removes those blocks from `content`. A block can contain several calls. Execute each call, append the assistant message and a `tool` result for every `tool_call_id`, then request the next answer. The server parses calls; your application executes the functions.

<Accordion title="Weather tools and follow-up example (Python)">

```python Example
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
model = "ai-sage/GigaChat4-708B-A41B"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "enum": ["London", "Paris"]}},
            "required": ["city"],
        },
    },
}]
messages = [{"role": "user", "content": "What's the weather in London and Paris?"}]
# Example results; replace this lookup with your weather service.
weather = {
    "London": {"temperature_c": 7, "condition": "clear"},
    "Paris": {"temperature_c": 18, "condition": "cloudy"},
}

for _ in range(4):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=1.0,
        top_p=1.0,
        max_tokens=4096,
        extra_body={
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"reasoning": False},
        },
    )
    message = response.choices[0].message
    print("Reasoning:", getattr(message, "reasoning_content", None))
    print("Content:", message.content)
    print("Tool calls:", message.tool_calls)
    messages.append(message.model_dump(
        include={"role", "content", "reasoning_content", "tool_calls"},
        exclude_none=True,
    ))
    if not message.tool_calls:
        break
    for call in message.tool_calls:
        if call.function.name != "get_weather":
            raise ValueError(f"Unknown tool: {call.function.name}")
        arguments = json.loads(call.function.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(weather[arguments["city"]]),
        })
else:
    raise RuntimeError("Tool-call loop exceeded four turns")
```

</Accordion>

The model can return both weather calls in one response or request them over successive turns. Process every returned call and continue until it returns an answer without calls. `tool_choice="auto"` lets the model choose whether to call a tool; `"required"` requires a call on that turn, and `"none"` disables tool calls. Apply `"required"` only to the turns that need a call, so the model can give its final answer after receiving the results.

### 3.3 Streaming

Set `stream=True` in the same `chat.completions.create` request. Thinking and answer text arrive in separate delta fields. Concatenate each field in arrival order; one chunk can contain several tokens, and chunk boundaries do not correspond to MTP verification steps.

<Accordion title="Streaming reasoning and answer (Python)">

```python Example
# Reuse client and model from the previous examples.
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Explain the Pythagorean theorem simply."}],
    temperature=1.0,
    top_p=1.0,
    max_tokens=4096,
    stream=True,
    extra_body={
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"reasoning": True},
    },
)
content_parts = []
print("Reasoning:")
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if text := getattr(delta, "reasoning_content", None):
        print(text, end="", flush=True)
    if delta.content:
        content_parts.append(delta.content)

print("\nAnswer:", "".join(content_parts))
```

</Accordion>

The fields contain parsed text; control tokens such as `<think>`, `</think>`, and GCML delimiters are removed by the parsers. To stream tool calls, collect `delta.tool_calls` by `index`, retain each call's `id`, and concatenate the function name and `arguments` fragments before parsing JSON. Execute a call only after its arguments are complete.

<Accordion title="Collect streaming tool calls (Python)">

```python Example
# Reuse client, model, and tools from the weather example.
import json

stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What's the weather in London and Paris?"}],
    tools=tools,
    tool_choice="auto",
    temperature=1.0,
    top_p=1.0,
    max_tokens=4096,
    stream=True,
    extra_body={
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"reasoning": False},
    },
)
calls = {}
for chunk in stream:
    if not chunk.choices:
        continue
    for part in chunk.choices[0].delta.tool_calls or []:
        call = calls.setdefault(part.index, {"id": None, "name": "", "arguments": ""})
        if part.id:
            call["id"] = part.id
        if part.function:
            call["name"] += part.function.name or ""
            call["arguments"] += part.function.arguments or ""

for index, call in sorted(calls.items()):
    print(index, call["id"], call["name"], json.loads(call["arguments"]))
```

</Accordion>

For concurrent clients, use independent message histories and up to the server's `--max-running-requests` limit. Both streaming and non-streaming requests can run concurrently; requests above the running limit wait in the queue.

### 3.4 MTP speculative decoding

The **Low-Latency** recipe runs the checkpoint's own NextN layer through EAGLE. GigaChat 4 reuses this single layer for three draft steps; GigaChat 3.5 uses stacked heads. No separate draft checkpoint or `--speculative-draft-model-path` is needed.

```bash Flags
--speculative-algorithm EAGLE \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

**EAGLE 3/1/4** means three draft steps, one candidate per step, and four verification positions. A fully accepted chain can commit up to four output tokens per step, including the target's bonus token. Compare acceptance and decode latency on your own prompts when choosing between the Low-Latency and High-Throughput recipes.

### 3.5 DFlash2 speculative decoding

The **DFlash2** recipe uses a separate draft trained for the target checkpoint. The draft combines grouped convolutions with a learned path selector over token candidates. It has its own weights and layer count; it does not use the target's bundled MTP layer.

Download the draft from the model repository into `./dflash_draft`. Run this command on each node, or use a shared working directory:

```bash Download the draft
hf download ai-sage/GigaChat4-708B-A41B \
  --include "dflash_draft/*" \
  --local-dir .
```

Run the deployment command from the same working directory, using these DFlash2 flags:

```bash Flags
--speculative-algorithm DFLASH \
--speculative-dflash-block-size 8 \
--speculative-draft-model-path dflash_draft \
--speculative-draft-model-quantization unquant \
--speculative-draft-attention-backend flashinfer
```

- **Use a matching draft.** The bundled `dflash_draft/` directory contains the draft for this target. If you download it elsewhere, update **DFlash2 draft path** in the deployment panel. The draft config must identify `GigaChat4DFlash2DraftModel` and contain the trained `dflash_config`, including the target depth. A draft trained for a different target size is not interchangeable.
- **Keep the MLA draft in BF16.** The target can remain FP8. `unquant` prevents the draft from inheriting the target's quantization setting; the current MLA draft implementation requires unquantized weights.
- **Configure the draft backend separately.** `--speculative-draft-attention-backend flashinfer` applies to the draft's dense MLA attention. The target keeps its DSA attention backend.
- **Choose the block size for the trained draft.** The preset uses eight positions: one anchor and seven proposed tokens. Check the draft's configuration before changing it.
