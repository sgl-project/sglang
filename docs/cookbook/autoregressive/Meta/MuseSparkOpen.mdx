---
title: Muse Spark Open
description: "Deploy Muse Spark Open with SGLang — launch recipes, feature compatibility, and tuning notes for GB300, B300, B200, H200, MI355X, Intel Xeon"
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

```bash Command
docker pull lmsysorg/sglang:dev-muse-spark-open     # NVIDIA B300 / GB300 / B200 / H200
docker pull lmsysorg/sglang-rocm:muse-spark-open    # AMD MI355X
docker pull lmsysorg/sglang:dev-muse-spark-xeon     # Intel Xeon CPU
```

The deployment panel can render a complete `docker run` command for the selected hardware and options. See [Install SGLang with Docker](/docs/get-started/install#method-3-using-docker) for host setup. The CUDA tag does not run on MI355X; use the ROCm image there; Use Xeon image for Xeon CPU and for more SGLang CPU installation approaches, see the [CPU version installation guide](/docs/hardware-platforms/cpu_server#installation).

</Accordion>

Choose your **Hardware Platform**, **Strategy**, and **Quantization** below. The default is **B300 / Low Latency / NVFP4**. **Low Latency** keeps the existing deployment settings; **High Throughput** removes all speculative decoding parameters. Use the Playground to tune parallelism, caching, speculative decoding, multimodal transport, and prefill/decode disaggregation.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/meta-models/muse-spark.jsx";

<Deployment config={config} agenticLink="agentic-long-context-deployment" />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations the SGLang team has signed off on; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## Agentic Long-Context Deployment

Agentic workloads, such as coding agents and other multi-turn, tool-using assistants, send long prompts whose prefixes grow and repeat from turn to turn. Serving them well depends on reusing the KV cache across turns, offloading it beyond GPU memory, and keeping each session on the workers that already hold its prefix. This section gives complete launch commands for such deployments, each benchmarked in [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/) and linked to the run that measured it. Pick a hardware platform, checkpoint, deployment shape, concurrency, KV cache offloading mechanism and router to get every process of the deployment as a raw command, one tab per process in launch order.

- **Router.** Each configuration was submitted behind one front door: Dynamo (`dynamo.frontend` with `dynamo.sglang` workers) or SGLang (`sglang.launch_server`, behind `sglang_router` when there are several workers or DP-attention ranks).
- **KV offload.** Choose among these KV cache offloading mechanisms: none (GPU KV only), HiCache (host DRAM), or a Mooncake store attached through the external linker. The Mooncake option needs an image whose SGLang build includes `--enable-unified-cache-external-linker`.
- **Container image.** Run the commands inside the image listed for each configuration; flag names follow that image.

import { AgentX } from "/src/snippets/_agentx.jsx";
import { agentx } from "/src/snippets/agentx/meta-models/muse-spark-open.jsx";

<AgentX data={agentx} />

## 1. Model introduction

This cookbook provides deployment recipes for the Muse Spark Open 816B-A42B checkpoints:

- **NVFP4:** `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` (NVIDIA default).
- **MXFP4:** `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` (MI355X default; Quark-quantized).
- **NVFP4 on MI355X (direct load):** the same `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` checkpoint runs on MI355X with `--quantization quark_mxfp4`; the weights are requantized to MXFP4 while loading, so no separate export is needed. Select **MI355X / NVFP4** in the panel for this command.
- **INT4** `intel-ai/Muse-Spark-Open-816B-A42B-INT4` (Xeon default; AutoRound-quantized).
- **BF16:** `meta-models/Muse-Spark-Open-816B-A42B`.

## 2. Configuration tips

### H200 configuration

NVFP4 runs on one 8x H200 node with Marlin W4A16 experts (`--moe-runner-backend marlin`), since Hopper has no TRT-LLM FP4 MoE. The BF16 checkpoint does not fit on one node, so it runs TP16/EP16 across two nodes; replace `<node-rank>` and `<node0-ip>` for each node. Prefill sequence parallel and MegaMoE need SM100 kernels and are unavailable on H200.

### Xeon CPU service configuration

- Xeon serves the **INT4** checkpoint (quantized by Intel AutoRound, HF id: `intel-ai/Muse-Spark-Open-816B-A42B-INT4`) on the SGLang CPU engine. A typical Xeon 6 P-core platform (e.g., Xeon 6980P) is recommended.
- Set `--page-size 1` for the `intel_amx` attention backend (the CPU default attention backend).
- We recommend setting the TP size equal to the NUMA count. See the **Notes** under [Launch of the Serving Engine](/docs/hardware-platforms/cpu_server#launch-of-the-serving-engine) in the SGLang CPU server document for how to configure the arguments, especially TP and NUMA binding.

## 3. Advanced Usage

### 3.1 Reasoning

On the server, enable `--reasoning-parser muse` in your `sglang serve` command. The [Deployment](#deployment) recipes above already include this flag; in the [Playground](#playground), keep **Reasoning Parser** enabled under **Parsers**. The parser separates the model's reasoning into `message.reasoning_content` and its final answer into `message.content`.

On the client, set `reasoning_effort` for each chat completion request. Muse Spark Open supports **four levels: `low`, `medium`, `high`, and `xhigh`**. If you omit the parameter, the checkpoint's chat template defaults to **`xhigh`**. Use `low` or `medium` for faster, shorter responses, and `high` or `xhigh` for complex problem solving, coding, and agentic tasks. Other values, including `minimal`, are not supported. See the [model card](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) for the supported settings.

The example below connects to the server on port `30000` and discovers its model ID, so it works with whichever checkpoint you selected in Deployment. Change `base_url` if the server runs on another host or port. The current checkpoint's chat template does not require `chat_template_kwargs={"thinking": True}`; set the reasoning level with `reasoning_effort`.

<Accordion title="Reasoning Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY",
    timeout=600.0,
)
model = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Briefly explain: what is SGLang?"},
    ],
    max_tokens=16384,
    reasoning_effort="xhigh",  # low | medium | high | xhigh (default)
)
choice = response.choices[0]
print("Reasoning:", getattr(choice.message, "reasoning_content", None))
print("Content:", choice.message.content)
print("Finish reason:", choice.finish_reason)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: We need to briefly explain what is SGLang. Likely want concise explanation. SGLang is structured generation language? It's a framework for programming LLMs. Developed by LMSYS? Actually SGLang is a serving framework.

Need to explain: SGLang is a fast serving framework/DSL for LLMs? Let's recall.

SGLang is a structured generation language, a domain-specific language for LLM programs. Created by LMSYS Org. Allows efficient LLM programming with structured prompting, caching, constrained decoding etc. Provides high-performance serving via RadixAttention, paged attention, etc. Used for chaining calls, multi-turn, function calling, reasoning.

We should give brief explanation.

Possibly user wants simple. Provide definition, key features, use cases.

Make it brief but informative.

Maybe need to distinguish from other? Should be neutral.

Potentially also SGLang = Serving framework?

SGLang is open-source system for efficient LLM serving and programming.

Key features:
- Front-end language (Python DSL) for controlling generation
- Backend runtime: RadixAttention for KV cache reuse, continuous batching, tensor parallelism, quantization, speculative decoding.
- Supports OpenAI-compatible API.

Briefly explain.

Keep concise.

Add example maybe?

User: "Briefly explain: what is SGLang?"

Provide maybe 3-4 bullet points.

Possibly mention origin: developed by LMSYS and UC Berkeley etc.

Ensure not too long. Brief.

Maybe add: SGL for Structured Generation Language.

Make answer accessible.

Probably mention advantages: faster, controllable.

We should not hallucinate too much. Provide accurate info.

SGLang GitHub: sgl-project/sglang.

SGLang latest: SGLang is inference engine?

Let's verify facts: SGLang was introduced in paper "SGLang: Efficient Execution of Structured Language Model Programs" arXiv 2312.07104, Dec 2023, authors Lianmin Zheng etc.

It provides primitives: gen(), select(), etc.

Runtime optimizations: RadixAttention, compressed FSM for constrained decoding, API speculative execution.

Provides high throughput, low latency.

Supports many models: Llama, Mistral, Qwen etc. Supports on NVIDIA GPUs, AMD.

Should answer.

User likely wants short. Provide 1 paragraph + bullets.

Make tone helpful.

Add use cases: agents, reasoning, data extraction.

Probably not over.

Potential need to add date knowledge cutoff 2026?

Add nothing disallowed.

Make final answer succinct.

Maybe add link? Not needed.

Use simple language.

We should not mention reasoning strength.

Proceed to final.

Content: SGLang — **Structured Generation Language** — is an open-source system for programming and serving Large Language Models (LLMs).

Created by LMSYS / UC Berkeley, it has 2 parts:

**1. Frontend - a Python DSL:** Lets you write LLM programs with logic, not just string prompts. Primitives like `gen()`, `select()`, `fork()` for chaining calls, controlling reasoning, forcing JSON/tool-calling structure, multi-agent workflows, etc.

**2. Backend - a high-performance inference engine:** Optimized runtime for fast, cheap serving with features like:
* **RadixAttention:** automatic KV-cache reuse for prefixes, few-shot examples, multi-turn chat
* Continuous batching, Tensor Parallelism, quantization, speculative decoding
* Fast constrained/structured decoding

In short: It's an alternative to vLLM / TensorRT-LLM. You get both a cleaner way to *program* LLMs and a very fast engine to *run* them, with an OpenAI-compatible API.

Finish reason: stop
```

</Accordion>

### 3.2 Tool calling

On the server, enable `--tool-call-parser muse` alongside `--reasoning-parser muse` in your `sglang serve` command. The [Deployment](#deployment) recipes above already include both flags; in the [Playground](#playground), keep **Tool Call Parser** and **Reasoning Parser** enabled under **Parsers**. SGLang converts Muse Spark Open's native tool-call format into OpenAI-compatible `message.tool_calls`, while keeping reasoning in `message.reasoning_content`.

On the client, pass function definitions in `tools`, with each function's arguments described by a JSON Schema. The checkpoint's [chat template](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4/blob/main/chat_template.jinja) formats these definitions for the model. With `tool_choice="auto"`, the model decides whether to call a tool. The example below requests a weather lookup and prints each returned function name and its JSON arguments.

<Accordion title="Tool Calling Example (Python)">

```python Example
import json

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY",
    timeout=600.0,
)
model = client.models.list().data[0].id

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, for example Beijing.",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location", "unit"],
                "additionalProperties": False,
            },
        },
    }
]

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Use get_weather to check the current weather in Beijing in Celsius."},
    ],
    tools=tools,
    tool_choice="auto",
    reasoning_effort="xhigh",
    max_tokens=16384,
)
choice = response.choices[0]
message = choice.message
print("Reasoning:", getattr(message, "reasoning_content", None))
print("Content:", message.content)
for tool_call in message.tool_calls or []:
    print("Tool call ID:", tool_call.id)
    print("Function:", tool_call.function.name)
    print("Arguments:", json.loads(tool_call.function.arguments))
print("Finish reason:", choice.finish_reason)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: User wants get_weather to check current weather in Beijing in Celsius.

We have function get_weather with parameters: location (string), unit (string enum celsius/fahrenheit)

We need to call it.

We must call the function with location=Beijing, unit=celsius

Add reasoning.

We should not add extra text beyond function call? In this format, we need to provide function call. Probably we need to output function call directly.

We are Meta AI. Follow guidelines.

We need to invoke get_weather.

Be careful to JSON correct.

Probably need thinking tags? But in this environment, we have tool access. We should call get_weather.

We should do that now.

Possibly reasoning strength 512, need thorough reasoning? No, we just need to call tool.

Ensure we use correct recipient format: get_weather

We need to produce "<atem:function_calls>..." block.

We need to provide location: "Beijing", unit: "celsius"

Make call.

Content: Beijing in Celsius — checking the weather for you immediately.
Tool call ID: call_43c0efae809d46988855bb79
Function: get_weather
Arguments: {'location': 'Beijing', 'unit': 'celsius'}
Finish reason: tool_calls
```

</Accordion>

This example inspects the requested tool calls; it does not fetch weather data. Your application executes the corresponding function. To continue the conversation, append the assistant message containing `tool_calls`, then a `role="tool"` message for each result with the matching `tool_call_id`, and send the updated messages with the same tool definitions.

### 3.3 Encode Prefill Decode Disaggregation

Encoder–Prefill–Decode (EPD) disaggregation separates media encoding, prompt processing, and token generation, so encoding-heavy image/video requests can be scheduled separately from language inference. The encoder sends features through `zmq_to_tokenizer`; prefill transfers KV state to decode through Mooncake, with a router exposing one client endpoint.

The examples below use one 8× B300 NVLink host: encoder TP2 shares GPUs 0–1 with prefill TP4/EP4 on GPUs 0–3; decode TP4/EP4 uses GPUs 4–7. Run each command in a separate shell using the NVIDIA image from [Deployment](#deployment), with a shared host network. Start encoder, prefill, decode, then router, waiting for each to become ready. Prefill (`30001`), decode (`30002`), and router (`8000`) match the Playground ports; the encoder uses `30010`.

<Note>
These are simplified examples derived from [PR #265](https://github.com/radixark/sglang-muse-spark/pull/265). The source run checked transport with simulated outputs; this ordinary-serving configuration still needs runtime and accuracy validation. Leave GPU memory headroom for the encoder sharing the prefill GPUs.
</Note>

<Accordion title="Encoder server (B300, GPUs 0–1)">

```bash Command
CUDA_VISIBLE_DEVICES=0,1 sglang serve \
  --model-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
  --quantization modelopt_fp4 \
  --tp-size 2 \
  --encoder-only \
  --encoder-transfer-backend zmq_to_tokenizer \
  --host 0.0.0.0 \
  --port 30010
```

</Accordion>

<Accordion title="Prefill server (B300, GPUs 0–3)">

```bash Command
export MC_INTRANODE_NVLINK=true
export MC_TCP_ENABLE_CONNECTION_POOL=true
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK

CUDA_VISIBLE_DEVICES=0,1,2,3 sglang serve \
  --model-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --tp-size 4 --ep-size 4 \
  --mem-fraction-static 0.8 \
  --language-only \
  --encoder-urls http://127.0.0.1:30010 \
  --encoder-transfer-backend zmq_to_tokenizer \
  --encoder-bootstrap-port 8997 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --reasoning-parser muse --tool-call-parser muse \
  --host 0.0.0.0 \
  --port 30001
```

</Accordion>

<Accordion title="Decode server (B300, GPUs 4–7)">

```bash Command
export MC_INTRANODE_NVLINK=true
export MC_TCP_ENABLE_CONNECTION_POOL=true
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK

CUDA_VISIBLE_DEVICES=4,5,6,7 sglang serve \
  --model-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --tp-size 4 --ep-size 4 \
  --mem-fraction-static 0.75 \
  --language-only \
  --encoder-urls http://127.0.0.1:30010 \
  --encoder-transfer-backend zmq_to_tokenizer \
  --encoder-bootstrap-port 8999 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --reasoning-parser muse --tool-call-parser muse \
  --host 0.0.0.0 \
  --port 30002
```

</Accordion>

<Accordion title="Router">

```bash Command
sglang-router launch \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30001 8998 \
  --decode http://127.0.0.1:30002 \
  --host 0.0.0.0 --port 8000
```

</Accordion>

The language workers use distinct encoder-bootstrap ports (`8997` / `8999`) on this shared host; the router's `8998` matches the prefill KV bootstrap port. Send client requests to `http://localhost:8000/v1`. See the [EPD guide](/docs/advanced_features/epd_disaggregation) for the architecture and additional deployment options.

### 3.4 Sequence Parallel for Prefill

Sequence parallelism (SP) splits activations along the token dimension across TP ranks. Each rank processes its own token shard for normalization, pooling, and unpooling, reducing duplicated computation and activation storage during long-prompt prefill. NVFP4 MegaMoE connects these shards to expert parallelism through integrated all-to-all dispatch and combine: tokens reach their selected experts, and weighted expert outputs return to the original token ranks.

For a Blackwell NVFP4 configuration (B300, B200, or GB300), use the [Playground](#playground) to enable **Prefill Sequence Parallel** and **Prefill MegaMoE** together. This adds `--enable-layernorm-sp --enable-megamoe-for-layernorm-sp` to the launch command. Keep **Attention TP** and **MoE EP** equal; MegaMoE requires SP to remain on. These prefill options are unavailable on H200 and are omitted for a PD decode worker.

### 3.5 Speculative Decoding with DFlash v2 and Block Verification

DFlash V2 uses the target model's hidden states to propose a block of future tokens in one draft-model forward pass; Muse Spark Open then verifies the block in a single pass, advancing by multiple tokens when proposals are accepted. V2 mixes information across neighboring draft positions and selects candidates with their dependencies in mind. Optional [block verification](https://arxiv.org/abs/2403.10444) evaluates draft prefixes jointly instead of token by token, potentially accepting longer prefixes while preserving the target sampling distribution.

In the [Playground](#playground), select **DFlash V2** under **Speculative Decoding** to add `--speculative-algorithm DFLASH` and `--speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2`. Then adjust:

- **Draft attention backend:** `fa4` for NVIDIA or `triton` for MI355X, inherited from Deployment and set with `--speculative-draft-attention-backend`.
- **Num draft tokens:** 1–8, default **8**, set with `--speculative-num-draft-tokens`.
- **Block Verification:** turn it on to add `--speculative-use-block-verification`; it is off by default.

The Deployment **Low Latency** strategy enables DFlash V2 on supported GPUs; **High Throughput** leaves speculative decoding off. The Playground inherits that choice and lets you override it.

### 3.6 MegaKernel with Helion (Experimental)

We support persistent Helion megakernels for DFlash V2 target verification. They fuse attention, projection and normalization boundaries, routing, MoE, and the layer tail, with a separate fused LM head, reducing kernel launches and communication overhead. TP4/EP1 shards each expert across four GPUs, avoiding the expert-load imbalance of EP4 at batch size 1. With **BS=1 + DFlash V2**, the reported decode performance is roughly **10% faster** than the TP4/EP4 baseline without megakernels on 4×GB300; gains depend on the workload.

Use this experimental recipe **only on a PD-disaggregated decode worker**, with four **GB300 (SM 10.3)** GPUs, NVFP4 weights, FP8 E4M3 KV cache, and `helion==1.5.0`. Keep prefill on the regular path: EP1 prefill is slower than the TP4/EP4 baseline. This specialized TP4/EP1 command is separate from the Playground's equal-TP/EP configuration.

<Accordion title="Helion decode worker (4×GB300)">

Install the dependency in the serving environment, then launch the decode worker:

```bash Command
python -m pip install helion==1.5.0

export NCCL_NVLS_ENABLE=1
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=1

SGLANG_FORCE_FUSED_OP_BACKEND=helion \
SGLANG_ENABLE_MUSE_SPARK_FINALIZE_AR=1 \
SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 \
python -m sglang.launch_server \
  --model-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
  --tokenizer-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4 \
  --served-model-name external \
  --json-model-override-args '{"eos_token_id":[200001,200008]}' \
  --tp-size 4 \
  --ep-size 1 \
  --context-length 131072 \
  --max-running-requests 1 \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --attention-backend trtllm_mha \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 128 \
  --disable-flashinfer-autotune \
  --disable-prefill-cuda-graph \
  --cuda-graph-max-bs-decode 1 \
  --reasoning-parser muse \
  --tool-call-parser muse \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2 \
  --speculative-draft-attention-backend fa4 \
  --speculative-num-draft-tokens 8 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --mem-fraction-static 0.80 \
  --host 0.0.0.0 \
  --port 30002
```

</Accordion>

Pair this worker with a compatible prefill worker using the same target checkpoint, TP4, `--kv-cache-dtype fp8_e4m3`, `--page-size 128`, and `--served-model-name external`, with bootstrap port `8998`. Use the [Playground](#playground) PD router command with the decode URL pointing to this worker on port `30002`; send client requests to the router on port `8000`.
