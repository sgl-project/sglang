---
title: Ling-3.1-flash
description: "Deploy Ling-3.1-flash with SGLang — FP8 hybrid KDA + MLA MoE recipes for H200 and GB300 GPUs, measured accuracy and decode speed, and Mooncake prefix caching."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

Use the Ling-3.1-flash image, which includes model support and the decode optimizations used by these recipes. The image supports multiple CPU architectures:

```bash Command
docker pull lmsysorg/sglang:dev-ling-3.1-flash
```

Select **Docker** in the command panel to launch the matching image. The **Python** command requires an environment with the same Ling-3.1-flash build. See the [installation guide](/docs/get-started/install#method-3-using-docker) for Docker setup.

<Note>
The first launch in a fresh container spends about five extra minutes compiling JIT kernels for the speculative-decoding graphs (about 4–5 minutes once, then about 30 seconds). Mount a persistent cache volume, for example `-v ~/.cache:/root/.cache`, to skip it on later starts.
</Note>

<Note>
**AMD MI325X / MI350X:** the published ROCm images predate the ROCm gated-MLA fix that Ling-3.1-flash needs; without it the server fails while capturing decode CUDA graphs. Until a ROCm image includes the fix, build a thin image that switches the nightly's editable SGLang source to the Ling-3.1-flash branch (use the `mi35x` base for MI350X), then substitute it for the image in the generated Docker command:

```dockerfile Dockerfile
FROM lmsysorg/sglang-rocm:v0.5.21-rocm724-mi30x-20261013
RUN cd /sgl-workspace/sglang \
 && git fetch https://github.com/sgl-project/sglang.git bailing-v3-day0 \
 && git checkout FETCH_HEAD
```
</Note>

</Accordion>

Pick your hardware to generate the launch command:

- **Low-Latency** — TP-only (EP=1) with NEXTN speculative decoding for single-user chat and interactive requests. Verified on H200.
- **High-Throughput** — expert-parallel (EP) serving for batch and multi-user traffic: TP8 + EP8 on H200/H800/H100/B200-class nodes, TP4 + EP4 on GB300, TP8 + EP8 with AITER on AMD MI300X/MI325X/MI350X/MI355X (without speculative decoding). Verified on H200 and GB300; other platforms are unverified. AMD MI325X and MI350X recipes (TP8 + EP8 with AITER, without speculative decoding) are provided unverified; the MI325X accuracy in the benchmark card was measured by AMD.

For prefix-heavy multi-turn workloads, enable **HiCache** in the Playground after starting the Mooncake services in §3.3. The combination is verified: prefix hits return from Mooncake L3 on every TP rank.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/inclusionAI/ling-3.1-flash.jsx";
import { benchmarks } from "/src/snippets/configs/inclusionAI/ling-3.1-flash-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

Experiment with SGLang features on top of the selected deployment. The speculative-decoding options show the checkpoint requirements; the multi-turn prefix-cache preset adds the paired scheduler flags and environment setting.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model introduction

[Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash) is a hybrid-attention Mixture-of-Experts language model using the `BailingMoeV3ForCausalLM` architecture. Its 72 layers combine Kimi Delta Attention (KDA) with Multi-head Latent Attention (MLA), with 512 routed experts. The checkpoint uses FP8 E4M3 weights in 128×128 blocks with dynamic activation quantization.

The model supports reasoning and structured tool calling, with a context limit of 262,144 tokens.

**Checkpoint:** [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash). **License:** MIT.

The model has about 560B total parameters (512 routed experts with 8 active plus one shared expert per token, about 17B active), 72 layers, and FP8 block-quantized weights.

## 2. Configuration tips

- Use eight H200 GPUs with `--tp 8 --ep 8`, or four GB300 GPUs with `--tp 4 --ep 4`. SGLang detects FP8 from the checkpoint.
- On H200, enable `SGLANG_ENABLE_FP8_BLOCK_GEMV=1` and `SGLANG_ENABLE_FUSED_KDA_OUTPUT_QUANT=1`. Both are included in the generated commands; they are no-ops on GB300 (SM10x).
- Leave `--mem-fraction-static` at its default. SGLang auto-selects the `ling3` reasoning and tool-call parsers for this architecture; you can change the parser settings in the Playground.
- **NEXTN:** add `--speculative-algorithm NEXTN` for built-in MTP speculative decoding. Measured acceptance length is 2.61 on random tokens and 2.13 on real-text probes.
- **Long context:** the checkpoint supports 262,144 tokens natively. For up to 1,048,576 tokens, add the YaRN override from §3.4.
- **Topology:** the Low-Latency recipe runs TP-only (no `--ep-size`) and is clearly faster at batch size 1 (about 9-13% lower TPOT on H200, verified on both 2048/512 and 8192/1024 workloads). At concurrency 16-32 the two recipes deliver the same throughput within ~2%. TP-only relies on the loader's TP-requant support; it fails loudly at startup on builds without it.
- For prefix-heavy, multi-turn workloads, select **Cache fork and tail** in the Playground. It combines `--mamba-radix-cache-strategy extra_buffer`, `--chunked-prefill-size 4096`, and `SGLANG_SPLIT_PREFILL_AT_MAMBA_FORK=1`. GPU checks confirmed both fork and tail states are cached, saving about 576 tokens per subsequent turn in the measured workload.

### Playground support

- **Attention:** BailingKDA shards its projections on attention TP; DP-attention size must divide TP. Prefill CP is disabled because BailingKDA does not implement the required sequence gather and recurrent-state handling.
- **MoE:** Auto, DeepGEMM, and Triton use the FP8 MoE runner paths. EP must divide TP.
- **PD disaggregation:** the Playground exposes separate prefill and decode roles with Mooncake transfer. The runtime has hybrid recurrent-state transfer support, but this model's PD deployment has not been verified. Provision separate workers and route requests through a PD-aware gateway; select the role-specific command for each worker.
- **HiCache:** enabling Mooncake on the H200 base preserves NEXTN and adds direct page-first I/O, hybrid-KDA prefix settings, and TCP storage endpoints.

### Performance

On H200 TP8 with NEXTN and the decode gates enabled, batch-size-1 output throughput measures **269.2 tokens/s** (TP8 + EP8: 239.5 tokens/s). GB300 TP4 with NEXTN measures **175.3 tokens/s** (`TPOT 5.22 ms, accept 2.60`; TP4 + EP4: 160.0 tokens/s).

Acceptance length is the mean number of tokens accepted per speculative verification step, including the target bonus token:

<table>
  <thead>
    <tr>
      <th style={{textAlign: "left", padding: "10px 12px"}}>Workload</th>
      <th style={{textAlign: "right", padding: "10px 12px"}}>Acceptance length</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style={{padding: "9px 12px"}}>Random-token benchmark (bs=1)</td><td style={{padding: "9px 12px", textAlign: "right"}}>2.61</td></tr>
    <tr><td style={{padding: "9px 12px"}}>GSM8K smoke (HiCache serve)</td><td style={{padding: "9px 12px", textAlign: "right"}}>2.29</td></tr>
    <tr><td style={{padding: "9px 12px"}}>Full GSM8K (cumulative)</td><td style={{padding: "9px 12px", textAlign: "right"}}>2.43</td></tr>
    <tr><td style={{padding: "9px 12px"}}>Real-text probes (30 prompts)</td><td style={{padding: "9px 12px", textAlign: "right"}}>2.13</td></tr>
  </tbody>
</table>

The image includes KDA normalization/quantization fusion, FP8 block GEMV, router-gate matrix-vector kernels, and a 512-expert top-k router.

<Accordion title="Reproduce the H200 speed workload">

Launch the H200 **Low-Latency** recipe, then run:

```bash Command
python3 -m sglang.bench_serving \
  --backend sglang \
  --host localhost --port 30000 \
  --model inclusionAI/Ling-3.1-flash \
  --dataset-name random \
  --random-input-len 2048 --random-output-len 512 \
  --random-range-ratio 1 \
  --num-prompts 16 --max-concurrency 1 \
  --flush-cache --seed 11
```

For the saturation row in the benchmark widget, launch the H200 **High-Throughput** recipe instead and run:

```bash Command
python3 -m sglang.bench_serving \
  --backend sglang \
  --host localhost --port 30000 \
  --model inclusionAI/Ling-3.1-flash \
  --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 \
  --random-range-ratio 1 \
  --num-prompts 256 --max-concurrency 48 \
  --flush-cache --seed 11
```

</Accordion>

<Accordion title="Reproduce GSM8K accuracy">

```bash Command
pip install sgl-eval
sgl-eval run gsm8k \
  --base-url http://localhost:30000/v1 \
  --num-threads 32
```

Evaluate all 1,319 questions and inspect completion reasons as well as accuracy. For the H200 NEXTN recipes, compare the stop rate with the measured 99.77% (Low-Latency) or 99.92% (High-Throughput); accuracy alone can hide generations that reach the token limit.

</Accordion>

## 3. Advanced usage

### 3.1 Reasoning

The `ling3` parser returns reasoning in `message.reasoning_content` and the final answer in `message.content`. Thinking is enabled by default. To request a direct answer, pass `"chat_template_kwargs": {"enable_thinking": false}` in the request body, or `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` with the OpenAI Python client.

<Accordion title="Thinking-on request (Python)">

```python Python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Ling-3.1-flash",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
)
print(response.choices[0].message.reasoning_content)
print(response.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
15% of 240 = 0.15 × 240 = 36.

Quick check: 10% of 240 = 24, 5% = 12, total 36. Correct.
**36**

Here's the calculation:
- 15% = 0.15
- 0.15 × 240 = 36

Or break it down: 10% of 240 = 24, and 5% of 240 = 12, so 24 + 12 = **36**
```

The first block is `reasoning_content` (59 tokens); the second is `content`. `finish_reason` is `stop`.

</Accordion>

<Accordion title="Thinking-off request (Python)">

```python Python
response = client.chat.completions.create(
    model="Ling-3.1-flash",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.reasoning_content)  # None
print(response.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
None
15% of 240 is **36**.

Calculation:
- 10% of 240 = 24
- 5% of 240 = 12
- 24 + 12 = **36**

Or directly: 240 × 0.15 = **36**
```

`reasoning_content` is `None` and `reasoning_tokens` is 0.

</Accordion>

### 3.2 Tool calling

The `ling3` tool-call parser returns structured calls in `message.tool_calls`, with `finish_reason` set to `tool_calls`. Send tools through the OpenAI-compatible `tools` field and return each tool result with its matching `tool_call_id`. Read both `reasoning_content` and `content` on the follow-up assistant turn.

<Accordion title="Tool-calling request and follow-up (Python)">

```python Python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. Beijing"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]

import json


def get_weather(city: str, unit: str = "celsius"):
    # Stand-in for your real tool implementation.
    return {"city": city, "temperature": 22, "condition": "sunny", "unit": unit}


response = client.chat.completions.create(
    model="Ling-3.1-flash",
    messages=[{"role": "user", "content": "What's the weather like in Hangzhou right now?"}],
    tools=tools,
    tool_choice="required",
)
msg = response.choices[0].message
call = msg.tool_calls[0]
print(msg.model_dump_json(indent=2, exclude_none=True))
print(response.choices[0].finish_reason)

# Execute the tool yourself, then return its result on the matching tool_call_id
result = get_weather(**json.loads(call.function.arguments))

follow_up = client.chat.completions.create(
    model="Ling-3.1-flash",
    messages=[
        {"role": "user", "content": "What's the weather like in Hangzhou right now?"},
        msg,
        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)},
    ],
    tools=tools,
)
print(follow_up.choices[0].message.reasoning_content)
print(follow_up.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output (first turn)">

```json Output
{
  "content": "",
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call_ecd286d7dccb4c70a8c9c586",
      "function": {
        "arguments": "{\"city\": \"Hangzhou\", \"unit\": \"celsius\"}",
        "name": "get_weather"
      },
      "type": "function",
      "index": 0
    }
  ],
  "reasoning_content": "The user is asking about the current weather in Hangzhou. I need to call the get_weather function with the city parameter set to \"Hangzhou\". The unit parameter is optional - I'll default to celsius since that's commonly used in China."
}
```

</Accordion>

<Accordion title="Example Output (follow-up)">

```text Output
The user asked about the weather in Hangzhou and I got the result. Let me provide a natural response.
The weather in Hangzhou right now is sunny with a temperature of 22°C. It's a pleasant day!
```

The first line is `reasoning_content`; the second is `content`. `finish_reason` is `stop`.

</Accordion>

### 3.4 Long context with YaRN (up to 1M tokens)

The model's native context is 262,144 tokens. Verified on H200 and GB300, YaRN extrapolation serves up to **1,048,576 tokens**:

```bash Command
--json-model-override-args '{"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144, "rope_theta": 6000000, "partial_rotary_factor": 0.5}, "max_position_embeddings": 1048576}'
```

Add it to the Low-Latency serve command. All four keys are required: the override replaces the checkpoint's RoPE parameters, so omitting `rope_theta` or `partial_rotary_factor` silently changes the RoPE base, and omitting `max_position_embeddings` keeps the 262,144 limit. Only enable it for prompts beyond 256K.

Verification on the current checkpoint: a fact planted at token 700,023 of a 900,022-token prompt is retrieved exactly, NEXTN acceptance stays about 2.1 at 900K, and a 300,101-token prompt is answered correctly (H200 TTFT 15.4 s, GB300 12.6 s). One 1M-token request fits alongside normal serving traffic on both platforms.

### 3.3 HiCache with Mooncake

With HiCache enabled, prefix state is written through to Mooncake L3 and later requests prefetch it back: measured 5952 of 6000 tokens restored from storage on every TP rank. Mooncake runs as L3 prefix storage over TCP; start the metadata server, master, and storage client before launching SGLang. Run these services on the same host as the SGLang container:

<Accordion title="Start Mooncake services">

```bash Command
docker run --rm --network host --ipc=host \
  lmsysorg/sglang:dev-ling-3.1-flash \
  bash -lc '
    python3 -m mooncake.http_metadata_server --port 8290 &
    mooncake_master --port 50171 --metrics_port 9024 &
    exec mooncake_client \
      --host=127.0.0.1 \
      --port=50172 \
      --master_server_address=127.0.0.1:50171 \
      --metadata_server=http://127.0.0.1:8290/metadata \
      --protocol=tcp \
      --device_names= \
      --global_segment_size=4294967296 \
      --enable_http_server=true \
      --http_port=8291
  '
```

</Accordion>

<Accordion title="Example Output">

```text Output
I1014 11:47:24.542050 328820 master.cpp:1522] Master service started on port 50171, max_threads=16, enable_metric_reporting=1, metrics_port=9024, ...
I1014 11:47:24.558478 328853 master_admin_service.cpp:301] Master Admin Metrics: role=leader, state=serving, service_ready=true, ...
I1014 11:47:27.557325 328879 real_client.cpp:1935] Client HTTP server started on port 8291
```

</Accordion>

Select **H200 → Low-Latency** in Deployment, then enable **HiCache → Mooncake** in the Playground. Keep NEXTN inherited from the base. Docker commands use host networking so the SGLang container can reach these localhost services. Change the master and metadata endpoints in the **Env** dialog for a remote store.

The TCP recipe leaves `MOONCAKE_DEVICE` and `--device_names` empty. The generated command includes `extra_buffer`, direct page-first I/O, and `hicache_storage_pass_prefix_keys=true` for hybrid KDA prefix reuse. To verify L3 round-trip: send a long prompt (thousands of tokens), wait for the server to go idle, flush with `curl -X POST http://localhost:30000/flush_cache -H 'Content-Type: application/json' -d '{}'`, then repeat the same prompt.

<Accordion title="Verify L3 prefetch (serve log)">

```text Output
[2026-10-14 11:59:25 TP0 EP0] HiCache prefetch success req=2619007c50024156a9872f10ec14606a completed=5952 matched=0 loaded=5952 occupied=0
[2026-10-14 11:59:25 TP1 EP1] HiCache prefetch success req=2619007c50024156a9872f10ec14606a completed=5952 matched=0 loaded=5952 occupied=0
[2026-10-14 11:59:25 TP2 EP2] HiCache prefetch success req=2619007c50024156a9872f10ec14606a completed=5952 matched=0 loaded=5952 occupied=0
[2026-10-14 11:59:25 TP7 EP7] HiCache prefetch success req=2619007c50024156a9872f10ec14606a completed=5952 matched=0 loaded=5952 occupied=0
[2026-10-14 11:59:25 TP6 EP6] HiCache prefetch success req=2619007c50024156a9872f10ec14606a completed=5952 matched=0 loaded=5952 occupied=0
[2026-10-14 11:59:25 TP5 EP5] HiCache prefetch success req=2619007c50024156a9872f10ec14606a completed=5952 matched=0 loaded=5952 occupied=0
[2026-10-14 11:59:25 TP4 EP4] HiCache prefetch success req=2619007c50024156a9872f10ec14606a completed=5952 matched=0 loaded=5952 occupied=0
[2026-10-14 11:59:25 TP3 EP3] HiCache prefetch success req=2619007c50024156a9872f10ec14606a completed=5952 matched=0 loaded=5952 occupied=0
```

</Accordion>

<Note>
`POST /flush_cache` is refused with 400 while any request is still in flight — the L3 write-through finishes after the response returns. Wait a few seconds of idle before flushing.
</Note>

For storage sizing and backend details, see [HiCache best practices](/docs/advanced_features/hicache_best_practices).
