---
title: K2 Horizon
description: "Deploy the IFM K2 Horizon family with SGLang — BF16 recipes for six dense and Mixture-of-Experts checkpoints on NVIDIA H200 GPUs."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

Install from a source checkout that includes the K2 Horizon runtime support in [PR #37654](https://github.com/sgl-project/sglang/pull/37654) before using the generated commands:

```bash Command
pip install --upgrade pip
pip install -e "python"
```

For general source-install guidance, see [Install → Method 2: From source](/docs/get-started/install#method-2-from-source).

</Accordion>

Choose a model size to generate its validated single-node H200 launch command. The serving settings in all six base recipes completed two independent H200 benchmark launches at the pinned model revisions using the K2 Horizon runtime support in [PR #37654](https://github.com/sgl-project/sglang/pull/37654).

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/IFM/k2-horizon.jsx";
import { benchmarks } from "/src/snippets/configs/IFM/k2-horizon-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

Each speed and accuracy value in the benchmark cards is the arithmetic mean of two independent server launches.

## Playground

The base recipes enable the `k2_horizon` reasoning parser. Use the Playground to configure the `k2_horizon` tool-call parser, TP and MoE expert-parallel overrides, NGRAM speculative decoding, PD disaggregation, and HiCache. A separate validation pass functionally exercised the tool-call parser on all six checkpoints; every other Playground override remains unverified.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model introduction

**K2 Horizon** is an IFM model family spanning six checkpoints from **0.9B** to **375B** parameters. The family includes dense models, Mixture-of-Experts models, and the MoVA-36B-A4B checkpoint's routed Mixture-of-Value-Attention path. This page provides native SGLang serving recipes for the family.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "22%"}} />
    <col style={{width: "25%"}} />
    <col style={{width: "23%"}} />
    <col style={{width: "30%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Checkpoint</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Architecture</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Context length</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>H200 recipe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/IFM/K2-Horizon-0.9B">0.9B</a></strong></td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>Dense</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>131,072</td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>BF16 · TP1</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/IFM/K2-Horizon-3.7B">3.7B</a></strong></td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>Dense</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>524,288</td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>BF16 · TP1</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/IFM/K2-Horizon-7B">7B</a></strong></td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>Dense</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>524,288</td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>BF16 · TP1</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/IFM/K2-Horizon-32B">32B</a></strong></td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>Dense</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>524,288</td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>BF16 · TP2</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B">MoVA-36B-A4B</a></strong></td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>MoE + MoVA</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>524,288</td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>BF16 · TP2</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong><a href="https://huggingface.co/IFM/K2-Horizon-375B-A23B">375B-A23B</a></strong></td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>MoE</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>524,288</td>
      <td style={{padding: "9px 12px", background: "rgba(255,255,255,0.02)"}}>BF16 · TP8</td>
    </tr>
  </tbody>
</table>

<Note>
  All six checkpoints use the Apache License 2.0.
</Note>

**Resources:** [K2 Horizon collection](https://huggingface.co/collections/IFM/k2-horizon) · [K2 Horizon 375B-A23B model card](https://huggingface.co/IFM/K2-Horizon-375B-A23B).

## 2. Configuration tips

- **Hardware and precision:** The current matrix covers NVIDIA H200 GPUs and BF16 checkpoints only.
- **Parallelism:** Use TP1 for 0.9B, 3.7B, and 7B; TP2 for 32B and MoVA-36B-A4B; and TP8 for 375B-A23B. Each recipe stays within one node.
- **Attention:** The verified base recipes select FlashAttention-3 with `--attention-backend fa3`.
- **Revisions:** The recipes pin revisions `9b9ec1f7e17f62ed218df542687a144116219d84` (0.9B), `c177771836a4c460743c00002c22483f6f18d1eb` (3.7B), `69ada542b68fe13d767479db2ab9421baff88681` (7B), `e1fd0277713e4eefcd3416348fd6fedacf7f2392` (32B), `16d20c739c687c08423422d1a2fbba6c529014cd` (MoVA-36B-A4B), and `12812264242a14dce44aa7ae27f931ff4584bcbf` (375B-A23B).
- **Parsers:** The base recipes enable reasoning with `--reasoning-parser k2_horizon`. Enable `--tool-call-parser k2_horizon` from the **Parsers** card in the [Playground](#playground) when you need tool calling.
- **MoVA-36B-A4B router provenance:** The MoVA-36B-A4B checkpoint's source xLLM router used two GEMM partitions. `--json-model-override-args '{"xllm_source_router_gemm_partitions":2}'` preserves its BF16 partial-GEMM rounding and FP32 reduction order; SGLang does not infer this source topology from runtime TP.
- **375B-A23B loading:** The 375B-A23B recipe disables multithreaded weight loading with `--model-loader-extra-config '{"enable_multithread_load":false}'`.
- **Playground features:** TP, EP, NGRAM, PD disaggregation, and HiCache are optional, unverified command overrides. The tool-call parser was functionally exercised separately.
- **Deliberate omissions:** Context parallelism is unsupported or unvalidated for xLLM. DP-attention and alternate MoE backend selectors are not exposed until they are validated.
- **Validation:** All six recipes completed native speed and full GSM8K evaluation on H200 using the K2 Horizon runtime support in [PR #37654](https://github.com/sgl-project/sglang/pull/37654).
- **375B-A23B provenance:** The 375B-A23B benchmark launched a directly mounted checkpoint and passed revision `12812264242a14dce44aa7ae27f931ff4584bcbf` to the server. The other five launches used materialized Hugging Face snapshots at their pinned revisions.

## 3. Benchmark results

Speed was measured with native `sglang.benchmark.serving` using fixed random token IDs: 8,192 input tokens, 1,024 output tokens, 64 warmup requests, a cache flush, and seed `20260901`. The latency run used 32 prompts at concurrency 1. The throughput run used 256 prompts at concurrency 64 for TP1, 32 for TP2, and 8 for TP8. TTFT and TPOT are medians within each launch; the benchmark cards show the arithmetic mean across the two launches. Throughput per GPU is `(total input tokens + total output tokens) / duration / TP`.

GSM8K used the full 1,319-example test split with `sgl-eval`, 32 client threads, a 32,768-token output limit, temperature 0, top-p 0.95, seed 0, and high reasoning effort. The mean and both independent server-launch scores are shown below. All requests completed without errors. Responses that reached the output limit remain in the score: truncation was 6.90% in both 0.9B launches, 4.09% and 4.70% for 3.7B, and at most 1.14% for the other checkpoints.

| Checkpoint | Launch 1 | Launch 2 | Mean |
| --- | ---: | ---: | ---: |
| 0.9B | 85.22% | 85.29% | 85.25% |
| 3.7B | 92.12% | 91.89% | 92.00% |
| 7B | 95.30% | 94.47% | 94.88% |
| 32B | 95.75% | 96.21% | 95.98% |
| MoVA-36B-A4B | 95.45% | 94.84% | 95.15% |
| 375B-A23B | 95.53% | 95.60% | 95.56% |

## 4. Advanced usage

### 4.1 Reasoning

The base recipes include `--reasoning-parser k2_horizon`. The parser returns thinking in `reasoning_content` and the final answer in `content`.

The example output below was captured with the 3.7B checkpoint and both parser flags enabled.

<Accordion title="Reasoning example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="IFM/K2-Horizon-3.7B",
    messages=[{"role": "user", "content": "Hello"}],
)

message = response.choices[0].message
reasoning = getattr(message, "reasoning_content", None)
if reasoning is None:
    reasoning = (message.model_extra or {}).get("reasoning_content")
print("Reasoning:", reasoning)
print("Answer:", message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: We have a user who just said "Hello". We need to respond. As an AI, we should be friendly, perhaps ask how can we help. If says "Hello" alone, respond with greeting and offer assistance. Potentially ask what they need. There's no special instruction; just typical.

But maybe we must think about the context: The user gave just a greeting; we should respond with a greeting and ask how we can help. Could also ask if they want anything specific.

We should be concise but friendly: "Hello! How can I assist you today?" Provide some content.

But maybe the user expects something else. But in this scenario, we just reply with a greeting.

Thus answer: "Hello! What can I do for you?" Etc.

Check any style guidelines: We want to keep it short or perhaps ask follow-up. The user didn't ask any question, just saying hello. Could also respond: "Hello! ..."

Thus arguably the answer: "Hi there! How can I help you today?" Let's do that.

But also maybe we could ask if they need help with something specific. That's appropriate.

Thus the final answer: "Hello! How can I assist you today?" Possibly add friendly banter.

We'll respond.

Answer:
Hello! 👋 How can I help you today?
```

</Accordion>

### 4.2 Tool calling

Enable **Tool Call Parser** in the **Parsers** card of the [Playground](#playground) to add `--tool-call-parser k2_horizon`. The parser exposes K2 tool output as OpenAI-compatible `message.tool_calls`.

The example output below was captured with the 32B checkpoint and both parser flags enabled.

<Accordion title="Tool-calling example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="IFM/K2-Horizon-32B",
    messages=[{"role": "user", "content": "Call get_weather for Paris."}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
    max_tokens=1024,
)

message = response.choices[0].message
reasoning = getattr(message, "reasoning_content", None)
if reasoning is None:
    reasoning = (message.model_extra or {}).get("reasoning_content")
print("Reasoning:", reasoning)
print("Content:", message.content or "")
for tool_call in message.tool_calls or []:
    print(f"Tool call: {tool_call.function.name}({tool_call.function.arguments})")
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: Okay, the user wants me to call the get_weather function for Paris. Let me check the tools available. There's a function called get_weather that takes a city parameter. The required parameter is city, and it's a string. So I need to make sure to pass "Paris" as the city. I don't see any other parameters needed. The user didn't mention any other details, so I'll just use the city name. Let me structure the tool call correctly. The function name is get_weather, and the arguments should be a JSON object with "city": "Paris". I need to make sure the JSON is properly formatted. Alright, that should do it.

Content:
Tool call: get_weather({"city": "Paris"})
```

</Accordion>

### 4.3 HiCache

For large checkpoints or repeated long-context prefixes, enable **HiCache** in the [Playground](#playground), then choose a storage backend and write policy. Treat the generated override as unverified until it is exercised on your storage stack.
