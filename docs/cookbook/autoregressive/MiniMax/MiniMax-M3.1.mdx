---
title: MiniMax-M3.1
description: "Deploy MiniMax-M3.1-Flash with SGLang — the static-NVFP4 build of MiniMax's sparse-attention Mixture-of-Experts reasoning model, with its bundled DSpark draft or RadixArk's DFlash2 draft for speculative decoding, on NVIDIA and AMD."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install). The two paths below match the **Python / Docker** toggle in the command panel.

<Tabs>

<Tab title="Python (pip / uv)">

MiniMax-M3.1 support has not yet been merged into SGLang `main`. Install from the support PR below. The PR number is **TBD**; replace `TBD` with the actual PR number before running these commands.

```bash Command
pip install --upgrade pip
pip install uv

PR_NUMBER=<the PR number of the model support PR>
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin "pull/${PR_NUMBER}/head"
git checkout FETCH_HEAD
uv pip install --prerelease=allow -e python
cd ..
```

On NVIDIA Blackwell (B300 / GB300 / B200), also install the MSA sparse-attention kernels at the tested commit. SGLang imports them at startup, and the server fails without them. Hopper and AMD do not use them. Some of these kernels are compiled at runtime and need a **CUDA toolkit (`nvcc`) >= 13.4** on the machine. The CUDA version of the PyTorch wheel (`cu130`) does not count. See [Blackwell prerequisites](#blackwell-prerequisites).

```bash Command
git clone https://github.com/MiniMax-AI/MSA.git
cd MSA
git checkout c8e1013a16a708d2a7656805d5fc26f2b87715d1
uv pip install --no-deps -e .
cd ..
```

Then run the **Python** output of the command panel below in that environment.

</Tab>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:dev-minimax-m31-flash
# AMD MI350X / MI355X (ROCm 7.2):
docker pull lmsysorg/sglang:dev-minimax-m31-flash-mi35x
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick your hardware + recipe to generate the launch command. The serving strategies cover the common operating points:

- **Low-Latency** — fastest responses with a few users at once. On B300 it serves one request at a time through the MSA-V1 MonoKernel; on AMD it is a TP4 recipe for up to about 32 concurrent requests.
- **Balanced** — good speed with several users at once. Use for typical multi-user serving.
- **High-Throughput** — most tokens per second across many users. Best for batch jobs and many concurrent agent sessions.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/MiniMaxAI/minimax-m3.1.jsx";

<Deployment config={config} />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations the SGLang team has signed off on; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

{/* TODO: 1-2 paragraph intro from the HF card — what the model is, release date,
    license, architecture highlights, context length. Keep it lean. */}

**Resources:** [HuggingFace](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash), [DFlash2 draft](https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2).

## 2. Configuration Tips

{/* TODO: model/hardware-specific tuning notes, caveats, known issues. */}

<a id="blackwell-prerequisites" />

### NVIDIA Blackwell (B300 / GB300 / B200)

- **`FMHA_SM100_ALLOW_JIT=1`** lets the MSA sparse-attention kernels compile on first launch. They are cached under `~/.cache/minfer/msa_v1` (override with `MSA_V1_AOT_CACHE`) and reused afterwards. Without the variable and without a cached build, the server fails at startup with `JIT compilation is disabled and AOT kernel ... was not found`. Only the variable's presence is checked, so `=0` does not turn JIT off. An image that ships a prebuilt cache can drop it.
- **CUDA toolkit >= 13.4.** The decode path dequantizes the NVFP4 KV cache with the `qmul4` kernel, which needs `nvcc` 13.4 or newer. The toolkit is taken from `CUDA_HOME`, then `CUDACXX`, then the `nvcc` on `PATH` or `/usr/local/cuda`. An older toolkit does not fail: the server silently falls back to a slower path with an extra FP8 scratch buffer. Check that the startup log prints `nvfp4_to_fp8 backend=qmul4`, and point `CUDA_HOME` at a 13.4 toolkit (for example `/usr/local/cuda-13.4`) if it does not.
- **The first launch is slow** while these kernels and the FlashInfer kernels compile. Later launches on the same machine reuse the caches.
- **DSpark draft.** The checkpoint ships the DSpark draft in its own `dspark/` subfolder. With `--speculative-algorithm DSPARK` and no `--speculative-draft-model-path`, the server finds that subfolder and loads the draft from it, for a Hub repo id as well as a local directory, so the recipes need no draft flag. Pass `--speculative-draft-model-path` only to use a different draft.
- **DFlash2 draft.** Pick **DFlash2** under **Speculative Decoding** in the command panel to use RadixArk's trained draft instead. It is a separate checkpoint, so the command passes `--speculative-algorithm DFLASH --speculative-draft-model-path RadixArk/Minimax-M3.1-Flash-Dflash2`. It also adds `--speculative-use-block-verification`, which accepts sampled draft proposals by their joint probability instead of token by token. For a greedy draft the two are the same, and the server logs that the flag has no effect. DFlash2 runs on Blackwell only, without DP attention (so not with the B300 / B200 High-Throughput recipes, which use `--attn-dp-size 2`), and only in colocated serving.

<a id="monokernel" />

### MonoKernel (Low-Latency)

The B300 **Low-Latency** recipe serves batch-size-1 decode through the MSA-V1 MonoKernel plugin with `--monokernel-backend msa`. The plugin supports Blackwell sm_100 and sm_103 (B200 / GB200 and B300 / GB300) at `--tp 4` only; it has been measured on B300. Each MoE layer of the DSpark verify step runs as two persistent kernels, MegaAttention then the token-parallel MoE, each fused with its TP4 all-reduce. The DSpark draft step runs as one cooperative kernel per rank. On 4x B300 with the acceptance length fixed at 5, decode reaches 1028 tokens/s at 1K context, 919 at 256K and 723 at 1M, against 479, 465 and 408 on the standard path.

- **Batch size 1 only.** Prefill, plain decode and batches of more than one request run on the standard path, so throughput under concurrency is lower than the Balanced recipe. The recipe caps `--max-running-requests` at 8.
- **Required flags.** The plugin needs `--page-size 128`, a DSpark chain of 8 draft tokens (`--speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8 --speculative-dspark-block-size 7`), and `--disable-attn-tp-gather`; without the last one SGLang pads the 7-row draft block and the server refuses to start. At startup the server lists every unsupported setting it finds, and the log confirms the plugin with `--monokernel-backend msa: minimax_m3_msa_v1 on sm_103, widths [8]`.
- **`ALLOW_JIT=1`** lets the MonoKernel kernels compile on first launch; set `CACHE_DIR` to keep the compiled kernels across restarts.
- **`SGLANG_FORCE_COARSE_WAR_BARRIER=1`** runs MegaAttention in canonical cluster placement, about 3–4% faster end to end. Canonical placement faults if another kernel holds SMs when MegaAttention launches, so the server must have the GPUs to itself. Without the variable, the server keeps cluster placement on, which is safe and slightly slower. The log prints `cluster placement off (canonical)` or `cluster placement on`.
- **Memory.** The plugin keeps its own copy of the routed experts (54.1 GiB per rank), so at `--mem-fraction-static 0.8` the KV pool holds about 6.5M tokens instead of 10.4M.
- DFlash2 is not available with this recipe.

### Memory and workload tuning

- **`--mem-fraction-static 0.8`** is the validated value at `--tp 4` on B300, and also on B200. The pool it reserves holds the NVFP4 main KV cache, the sparse-attention index cache and the bf16 DSpark draft KV cache. B200 has 192 GB per GPU against B300's 288 GB. Its high-throughput recipe (`--tp 8 --attn-dp-size 2`) therefore uses `--mem-fraction-static 0.75` and `--cuda-graph-max-bs-prefill 16384`. With B300's settings, 8x B200 runs out of GPU memory capturing the 32k-token prefill CUDA graphs, and then while preprocessing images on GPU 0.
- **`--tokenizer-worker-num 8`.** With 100K-token prompts a single tokenizer process becomes the bottleneck: tokenization blocks the server's event loop and TTFT climbs under load even when the GPUs have headroom. Eight workers remove that ceiling, so every recipe above sets it.
- **Hierarchical KV cache for multi-turn traffic.** `--enable-hierarchical-cache --hicache-size <GB per rank>` keeps evicted prefixes in host memory so a returning session reloads its prefix instead of recomputing it. Adding `--hicache-storage-backend mooncake` puts a Mooncake storage tier (L3) under it, which a set of replicas can share; pick **Mooncake** under **Storage** in the HiCache card of the [Playground](#playground). The Mooncake tier is validated on GB300 and B300, with DSpark and with DFlash speculative decoding. Other storage backends have not been validated. For the Mooncake master and environment setup, see [Deployment with Mooncake](../../../docs/advanced_features/hicache_best_practices#deployment-with-mooncake). When several replicas serve one endpoint, put the router in front with `--policy cache_aware` so a session's later turns reach the replica that holds its prefix.
- **`--max-running-requests` and `--cuda-graph-max-bs-decode`** are both 128 in the recipes. Keep them equal so every decode batch the scheduler can form replays a captured CUDA graph.

<a id="amd-tips" />

### AMD MI350X / MI355X

- Requires ROCm 7.2 (Triton 3.7); earlier ROCm images are not supported.
- **NVFP4 on MI355X:** the same `MiniMaxAI/MiniMax-M3.1-Flash` checkpoint as on NVIDIA; the routed experts are requantized to MXFP4 while loading.
- **MXFP4 on MI350X / MI355X:** `amd/MiniMax-M3.1-Flash-Quark-MXFP4` (Quark-quantized), same command as NVFP4 Low-Latency on both; select **MI350X / MXFP4** or **MI355X / MXFP4** in the panel. Only the Low-Latency recipe has been run with this checkpoint.
- **One server per four GPUs.** M3.1 NVFP4 on ROCm runs at `--tp 4` or less, and the server refuses a larger TP. On an 8-GPU node, start two servers with `HIP_VISIBLE_DEVICES=0,1,2,3` and `HIP_VISIBLE_DEVICES=4,5,6,7` on different ports, and put the router in front with `--policy cache_aware`.
- **Pick the recipe by concurrency.** Low-Latency is for up to about 32 concurrent requests. High-Throughput is for 48 or more concurrent long multi-turn sessions, such as coding agents, and adds two things to Low-Latency:
  - **Hierarchical KV cache** (`--enable-hierarchical-cache`). At the default ratio of 2, the host pools (main KV, index K and draft KV) take about 350 GB per rank, so one TP4 server needs about 1.4 TB of host memory. On a host with less free memory, the server lowers the ratio until the pools fit in 80% of it. When two servers share a host, lower `--hicache-host-memory-fraction` on each.
  - **FP8 DSpark draft KV cache** (`--speculative-draft-kv-cache-dtype fp8_e4m3`). The same memory holds 9% more tokens. The draft cache only affects which draft tokens get accepted; the target model still verifies every token.
- **`--prefill-decode-interval 16`** runs 16 decode steps after each prefill batch before the next prefill. With 64 concurrent sessions and HiCache on MI350X, it lowered P90 inter-token latency by about 10% at the cost of about 3% throughput.
- The High-Throughput recipe was measured on MI350X. MI355X runs the same command but has not been measured with it.

## 3. Advanced Usage

### 3.1 Reasoning

Enable the `minimax-m3` reasoning parser (toggle **Reasoning Parser** in the **Parsers** card of the [Playground above](#playground)) to separate thinking from the final answer. The `<mm:think>` trace then lands in `message.reasoning_content`, separate from the final answer in `message.content`.

The reasoning traces in the example outputs below are abridged; `…` marks omitted text.

<Accordion title="Reasoning Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M3.1-Flash",
    messages=[{"role": "user", "content": "What is 15% of 240? Explain briefly."}],
    max_tokens=4096,
)

message = response.choices[0].message
print("=============== Reasoning ===============")
print(message.reasoning_content)
print("=============== Answer ==================")
print(message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
=============== Reasoning ===============
… Calculate 15%=0.15*240 = 36. Brief explanation likely 10%=24, 5%=12 total 36 or 0.15×240. …
=============== Answer ==================
15% of 240 is **36**.

Calculation: \(0.15 \times 240 = 36\).
```

</Accordion>

When streaming, the trace arrives on `delta.reasoning_content` and the answer on `delta.content`, so the two can be rendered separately as they arrive:

<Accordion title="Streaming Reasoning (Python)">

```python Example
response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M3.1-Flash",
    messages=[{"role": "user", "content": "Solve step by step: what is 15% of 240?"}],
    max_tokens=4096,
    stream=True,
)

for chunk in response:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):
        print(delta.reasoning_content, end="", flush=True)  # thinking stream
    if delta.content:
        print(delta.content, end="", flush=True)            # answer stream
print()
```

</Accordion>

<Accordion title="Example Output">

```text Output
[delta.reasoning_content — thinking stream]
… Can give 15%=15/100=.15, .15×240=(15×240)/100=3600/100=36. …

[delta.content — answer stream]
1. Convert the percentage to a decimal: \(15\% = 0.15\).
2. Multiply by 240: \(0.15 \times 240 = 36\).

**Answer: 36**
```

</Accordion>

### 3.2 Thinking Mode and Reasoning Effort

M3.1 takes two request-level controls. Pass them as top-level fields of the Chat Completions body, through `extra_body` with the OpenAI client:

- **`thinking`** — `{"type": "enabled" | "disabled" | "adaptive"}`. `enabled` always thinks before answering, `disabled` answers directly, and `adaptive` lets the model decide per turn. Omitting it means `adaptive`. Any other `type` is rejected with HTTP 400.
- **`reasoning_effort`** — `"low"`, `"medium"`, `"high"`, `"xhigh"` or `"max"`. The value is passed to the chat template, which tells the model how much to reason. `"none"` and `"minimal"` turn thinking off, the same as `thinking={"type": "disabled"}`. The effort is ignored when thinking is disabled.

<Accordion title="Thinking Mode and Effort Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
question = [{"role": "user", "content": "How many prime numbers are there below 50?"}]

for label, extra in [
    ("thinking disabled", {"thinking": {"type": "disabled"}}),
    ("adaptive, effort low", {"thinking": {"type": "adaptive"}, "reasoning_effort": "low"}),
    ("enabled, effort max", {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}),
]:
    response = client.chat.completions.create(
        model="MiniMaxAI/MiniMax-M3.1-Flash",
        messages=question,
        max_tokens=8192,
        extra_body=extra,
    )
    message = response.choices[0].message
    reasoning_chars = len(message.reasoning_content or "")
    print(f"[{label}] reasoning chars: {reasoning_chars}")
    print(f"  answer: {message.content.strip()}")
```

</Accordion>

<Accordion title="Example Output">

```text Output
[thinking disabled] reasoning chars: 0
  answer: There are **15** prime numbers below 50:

2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, and 47.
[adaptive, effort low] reasoning chars: 8
  answer: There are **15** prime numbers below 50:

2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47.
[enabled, effort max] reasoning chars: 268
  answer: There are **15 prime numbers** below 50:

2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, and 47.
```

</Accordion>

### 3.3 Tool Calling

Enable the `minimax-m3` tool-call parser (toggle **Tool Call Parser** in the **Parsers** card of the [Playground above](#playground)) to surface structured tool calls via `message.tool_calls`.

<Accordion title="Tool Calling Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M3.1-Flash",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
)

message = response.choices[0].message
if message.tool_calls:
    for call in message.tool_calls:
        print(f"Tool: {call.function.name}")
        print(f"Args: {call.function.arguments}")
```

</Accordion>

<Accordion title="Example Output">

```text Output
Tool: get_weather
Args: {"location": "Beijing", "unit": "celsius"}
```

</Accordion>

To return a tool result, append the assistant turn with its `tool_calls` and one `tool` message per call (matching `tool_call_id`), then ask the model to continue.

### 3.4 Multimodal (Vision) Input

Images go through the standard OpenAI `image_url` content type.

<Accordion title="Vision Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M3.1-Flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/examples/assets/example_image.png"
                    },
                },
                {"type": "text", "text": "Describe this image in two sentences."},
            ],
        }
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
A man in a yellow shirt stands on the rear bumper of a yellow taxi, pressing clothes on an ironing board extending from its open hatch. Another yellow taxi speeds past on the city street lined with tall buildings.
```

</Accordion>

### 3.5 Prefill-Decode (PD) Disaggregation

[PD disaggregation](../../../docs/advanced_features/pd_disaggregation) runs prefill and decode on **separate** SGLang servers, linked by an RDMA KV-transfer fabric and fronted by the PD router. Each M3.1 layer stores three things the decode server needs: the packed NVFP4 main KV cache with its block scales, an NVFP4 K-only sparse-index cache, and, with DSpark, the draft model's KV cache. SGLang sends all three from prefill to decode over the same page mapping, so PD needs no M3.1-specific flags.

**Supported topology**

- **Same attention parallelism on both roles.** Prefill and decode must use the same `--tp` and `--attn-dp-size`. The sparse-index transfer cannot reshard across different attention TP sizes.
- **Single pipeline stage.** PP must be 1, which DSpark also requires.
- **Speculative decoding.** Run DSpark on both roles. DFlash2 is only supported in colocated serving.
- **Transfer backend.** Mooncake over RDMA.
- **KV caching.** Run HiCache on the prefill server. The decode server keeps its default chunk cache. `--disaggregation-decode-enable-radix-cache` lets decode reuse prefixes it already holds. It is experimental and off by default.
- **HiCache on the decode server needs the decode radix cache.** A decode server turns its radix cache off unless `--disaggregation-decode-enable-radix-cache` is set, and `--enable-hierarchical-cache` with the radix cache off fails at startup (`enable-hierarchical-cache and disable-radix-cache are mutually exclusive`). Pass both flags on decode. The decode radix cache works with DSpark or without speculative decoding. With DP attention, route each session to the same DP rank so it finds its prefix. The playground adds the flag when you pick the decode role with HiCache on.

**Recommended layout: DP8 / TP1 / EP8 on each role.** Each role takes one 8×B300 node:

- Attention is data-parallel over eight ranks, each with attention TP 1 (`--tp 8 --attn-dp-size 8`).
- The experts are split eight ways through MegaMoE (`--ep 8`).
- The other flags follow the B300 High-Throughput recipe.

On B300, picking **Prefill role** or **Decode role** in the [Playground](#playground) PD card applies this layout to either recipe. Start the prefill server on node A, the decode server on node B, and then the router. Mooncake picks the RDMA NICs automatically; add `--disaggregation-ib-device` to pin specific ones.

```bash Prefill server (node A)
FMHA_SM100_ALLOW_JIT=1 \
SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768 \
sglang serve \
  --model-path MiniMaxAI/MiniMax-M3.1-Flash \
  --trust-remote-code \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --tp 8 \
  --attn-dp-size 8 \
  --enable-dp-lm-head \
  --ep 8 \
  --speculative-algorithm DSPARK \
  --cuda-graph-backend-prefill breakable \
  --chunked-prefill-size 65536 \
  --max-running-requests 256 \
  --cuda-graph-max-bs-decode 128 \
  --mem-fraction-static 0.8 \
  --tokenizer-worker-num 8 \
  --enable-hierarchical-cache \
  --hicache-size 192 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --host 0.0.0.0 --port 30000
```

```bash Decode server (node B)
FMHA_SM100_ALLOW_JIT=1 \
SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768 \
sglang serve \
  --model-path MiniMaxAI/MiniMax-M3.1-Flash \
  --trust-remote-code \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --tp 8 \
  --attn-dp-size 8 \
  --enable-dp-lm-head \
  --ep 8 \
  --speculative-algorithm DSPARK \
  --cuda-graph-backend-prefill breakable \
  --chunked-prefill-size 65536 \
  --max-running-requests 256 \
  --cuda-graph-max-bs-decode 128 \
  --mem-fraction-static 0.8 \
  --tokenizer-worker-num 8 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --host 0.0.0.0 --port 30100
```

```bash Router
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<node-A>:30000 8998 \
  --decode http://<node-B>:30100 \
  --policy round_robin \
  --host 0.0.0.0 --port 8000
```

The `8998` after the prefill URL is the prefill bootstrap port. It must match the prefill server's `--disaggregation-bootstrap-port`.

- **HiCache host memory.** `--hicache-size` is per rank, so at DP8 the prefill node reserves 8 × 192 GB of host memory for the host pools. Lower it on nodes with less memory.
- **MegaMoE token budget.** `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` must cover each rank's share of a prefill chunk (65536 / 8 here) and its largest decode or verify batch.
- **Media options through the router.** The PD router forwards only the standard OpenAI request fields. Per-request media options such as `max_long_side_pixel` and video `fps` do not reach the servers; this was observed with sglang-router 0.3.2.

<Note>
Correctness was validated on 1P1D with TP4 on each role (8×B300, Mooncake, DSpark on both roles, HiCache on prefill). GSM8K matched colocated serving, and `--disaggregation-enable-kv-checksum` reported no mismatch. The DP8 / TP1 / EP8 layout has not been benchmarked yet.
</Note>
