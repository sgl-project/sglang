---
title: Kimi-K3.1-Pro
description: "Deploy Moonshot AI's Kimi K3.1 Pro with SGLang: 2.8T parameters (104B active) and a 1M context, on NVIDIA Vera Rubin, Blackwell and Hopper, AMD Instinct, PPU and Google TPU."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

Kimi K3.1 Pro is served from the `kimi-k3.1-day-0` branch of [SGLang](https://github.com/sgl-project/sglang), based on `main`. To build and install from source, follow `docker/k31.Dockerfile` so that every dependency is in place.

<Tabs>

<Tab title="Python (from source)">

```bash Command
git clone -b kimi-k3.1-day-0 https://github.com/sgl-project/sglang.git && cd sglang
pip install -e "python"
```

</Tab>

<Tab title="Docker">

Use the Kimi K3.1 Pro image `lmsysorg/sglang:dev-kimi-k31-pro` (`lmsysorg/sglang:dev-kimi-k31-pro-mi35x` on AMD MI350X / MI355X, ROCm 10) and substitute the inner `sglang serve ...` with what the command generator below produces. See [Install → Method 3: Using Docker](/docs/get-started/install#method-3-using-docker).

</Tab>

<Tab title="PPU (M890 / 810E)">

Use the PPU image `egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun-tmp/inference-xpu-pytorch:kimi-k3.1-pro-sglang0.5.22-cuda13.0-pytorch2.13.0-20261103` with a prepared [SAIL environment](/docs/hardware-platforms/sail), and run the command panel's **Python** output inside the container on each worker node.

</Tab>

<Tab title="TPU (TPU7x)">

Install [SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu#get-started) on every host, then follow [TPU7x deployment](#tpu).

</Tab>

</Tabs>

</Accordion>

Pick your hardware, deployment mode and serving strategy to generate the launch command:

- **Low-Latency (TP8)**: the fastest replies with a few users at once.
- **High-Concurrency (DCP8)** (B300, GB300, Unified): the most requests and context per GPU, with the KV cache split across the eight GPUs.
- **Balanced** (decode role, other platforms): good speed with several users at once, for typical multi-user serving.
- **High-Throughput**: the most tokens per second across many users, for batch jobs and many concurrent agent sessions.
- **Prefill / Decode**: run the two phases on separate nodes (see [PD disaggregation](#pd-disaggregation)). The prefill role offers Low-Latency (TP8) and High-Throughput (PP4 × PCP2); the decode role offers Balanced (DCP8 · EP2) and High-Throughput (DCP8). On MI350X / MI355X, both roles offer High-Throughput (DCP8).

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/moonshotai/kimi-k3.1-pro.jsx";
import { benchmarks } from "/src/snippets/configs/moonshotai/kimi-k3.1-pro-benchmarks.jsx";
import { KimiK31MambaRatioCalculator } from "/src/snippets/_kimi_k31_mamba_ratio_calculator.jsx";

<Deployment config={config} benchmarks={benchmarks} agenticLink="agentic-long-context-deployment" />

Each command carries a badge:

- **Verified**: that exact hardware and command were tested.
- **Final Verification In Progress**: the recipe runs and is queued for measurement on the final weights.
- **Not Verified** (Playground only): the Playground command does not match a Verified cell.

<Note>
  The B300 `Decode` Balanced cell is **Verified** with DFLASH (1P1D round below, AIME26 ×8 = 1.000), and the B300 `Decode` High-Throughput cell without speculation. The other cells are still in verification. No cell has a cookbook-format speed sweep yet, so the benchmark card shows *pending*.

  The MI350X / MI355X cells are **Verified** with DFLASH; [AMD MI350X / MI355X](#amd-tips) says which to pick for your agent count.
</Note>

### Mamba ratio calculator

On `Unified`, `--mamba-full-memory-ratio` follows your average request length. The prefill and MI350X / MI355X High-Throughput recipes pin the ratio they were measured with; the decode and MI350X / MI355X Low-Latency recipes size the KDA pool directly, so the ratio does not apply.

<KimiK31MambaRatioCalculator />

<a id="playground" style={{ scrollMarginTop: "96px" }} />

## Advanced Features Playground

Turn on additional knobs on top of whichever cell the Deploy panel is showing:

- **Serving:** parsers, the PD transfer, HiCache storage and write policy, DFLASH proposed tokens, the prefill CUDA graph and the PD decode radix cache.
- **WideEP Decode** (GB300 `Decode`): one decode server on four trays, attention DP2 with TP8 · DCP8 in each 8-GPU group and MegaMoE EP16 across all 16 GPUs. In a fixed-batch survey it beat 8-GPU decode workers from 8 concurrent requests up, by 15% at 64. That survey ran without the radix cache; turning the PD decode radix cache on still keeps one KDA state slot per request, since the decode tree caches KV only. Not Verified on the current code.
- **Tuning:** the DSA attention backend and the MoE runner (B300, GB300, B200, GB200), the DCP communication backend, the KDA state slot count, KDA states per radix path, the MegaMoE token cap (when MoE Runner is MegaMoE) and the PP layer partition (PP4 × PCP2 prefill).

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## Agentic Long-Context Deployment

Agentic workloads, such as coding agents and other multi-turn, tool-using assistants, send long prompts whose prefixes grow and repeat from turn to turn. Serving them well depends on reusing the KV cache across turns, offloading it beyond GPU memory, and keeping each session on the workers that already hold its prefix. This section gives complete launch commands for such deployments on GB300 with Dynamo and SGLang, and on MI355X with SGLang, measured on the AgentX agentic-coding trace replay. Pick a deployment shape, concurrency, KV cache offloading mechanism and router to get every process of the deployment as a raw command, one tab per process in launch order.

<Note>
  These commands are a **status snapshot** (2026-11-03) of a GB300 tuning campaign that is still running. They are not an InferenceX submission. Each was measured with the DFlash acceptance length fixed to the draft's expected value (see the note under each configuration). The aggregated deployment at concurrency 1 missed its time-to-first-token target; the other seven points met theirs.
</Note>

- **Router.** Every configuration was measured behind Dynamo (`dynamo.frontend` with `dynamo.sglang` workers and the KV-aware router with session affinity). The SGLang router (`sglang_router` in front of `sglang.launch_server` workers) is derived from the same engine settings and was not measured.
- **KV offload.** The disaggregated prefill workers were measured with HiCache in host DRAM; the aggregated workers without KV offload. The other tiers (none, HiCache, or a Mooncake store attached through the external linker) are derived and were not measured.
- **Container image.** The commands show `<K31_IMAGE>`. The measured image was built with `docker/k31.Dockerfile` from SGLang commit `9ab3c28176` plus one unmerged change, and bundles FlashInfer 0.7.2, `sgl-deep-gemm` and a Dynamo 1.6.0 build with Kimi K3.1 Pro and session-affinity patches. `lmsysorg/sglang:dev-kimi-k31-pro` has not been checked against these commands.
- **Draft.** The commands load the [Kimi-K3.1-Dflash2-Preview](https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview) draft by its Hugging Face id; the measured runs read a local copy of it.

On MI355X:

- **Shape.** Up to 56 agents, one SGLang server on one node: TP8 for 1 and 4 agents, TP8 · DCP8 from 8; 1 to 8 agents also run the decode monokernel (`--enable-amd-monokernel` with `SGLANG_AMD_MONO_K31_WIDE=1`). `--max-running-requests` and `--cuda-graph-max-bs-decode` follow the agent count (64 at 56). At 92 agents, the Deploy panel's `High-Throughput` prefill and decode run as a 1P1D on two nodes behind the SGLang router.
- **KV offload.** From 16 agents, HiCache in host DRAM (write-back), with the no-offload tier derived; up to 8 agents, none, since the monokernel does not accept KV offload. Mooncake has not run on MI355X.
- **Container image.** The commands show `<K31_MI35X_IMAGE>`. The measured image was a ROCm 10 SGLang image with the Kimi K3.1 ROCm port; `lmsysorg/sglang:dev-kimi-k31-pro-mi35x` has not been checked against these commands.

The measured MI355X points; tok/s/user is 1000 / P90 inter-token latency and TTFT is p50 / p90:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "13%"}} />
    <col style={{width: "17%"}} />
    <col style={{width: "18%"}} />
    <col style={{width: "18%"}} />
    <col style={{width: "16%"}} />
    <col style={{width: "18%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Agents</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Shape</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>tok/s/GPU</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>tok/s/user</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>TTFT (s)</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Prefix hit</th>
    </tr>
  </thead>
  <tbody>
    <tr style={{backgroundColor: "rgba(255,255,255,0.02)"}}>
      <td style={{padding: "9px 12px"}}>16</td>
      <td style={{padding: "9px 12px"}}>1×8</td>
      <td style={{padding: "9px 12px"}}>8,868</td>
      <td style={{padding: "9px 12px"}}>60.2</td>
      <td style={{padding: "9px 12px"}}>0.65 / 1.9</td>
      <td style={{padding: "9px 12px"}}>96.4%</td>
    </tr>
    <tr style={{backgroundColor: "rgba(255,255,255,0.05)"}}>
      <td style={{padding: "9px 12px"}}>32</td>
      <td style={{padding: "9px 12px"}}>1×8</td>
      <td style={{padding: "9px 12px"}}>15,053</td>
      <td style={{padding: "9px 12px"}}>36.3</td>
      <td style={{padding: "9px 12px"}}>1.2 / 5.2</td>
      <td style={{padding: "9px 12px"}}>96.5%</td>
    </tr>
    <tr style={{backgroundColor: "rgba(255,255,255,0.02)"}}>
      <td style={{padding: "9px 12px"}}>48</td>
      <td style={{padding: "9px 12px"}}>1×8</td>
      <td style={{padding: "9px 12px"}}>17,818</td>
      <td style={{padding: "9px 12px"}}>28.6</td>
      <td style={{padding: "9px 12px"}}>2.0 / 10.4</td>
      <td style={{padding: "9px 12px"}}>96.2%</td>
    </tr>
    <tr style={{backgroundColor: "rgba(255,255,255,0.05)"}}>
      <td style={{padding: "9px 12px"}}>56</td>
      <td style={{padding: "9px 12px"}}>1×8</td>
      <td style={{padding: "9px 12px"}}>17,529</td>
      <td style={{padding: "9px 12px"}}>25.6</td>
      <td style={{padding: "9px 12px"}}>3.3 / 20.3</td>
      <td style={{padding: "9px 12px"}}>95.7%</td>
    </tr>
    <tr style={{backgroundColor: "rgba(255,255,255,0.02)"}}>
      <td style={{padding: "9px 12px"}}>92</td>
      <td style={{padding: "9px 12px"}}>1P1D, 2×8</td>
      <td style={{padding: "9px 12px"}}>17,883</td>
      <td style={{padding: "9px 12px"}}>30.6</td>
      <td style={{padding: "9px 12px"}}>3.7 / 10.3</td>
      <td style={{padding: "9px 12px"}}>94.4%</td>
    </tr>
  </tbody>
</table>

import { AgentX } from "/src/snippets/_agentx.jsx";
import { agentx } from "/src/snippets/agentx/moonshotai/kimi-k3.1-pro.jsx";
import { agentxMi355x } from "/src/snippets/agentx/moonshotai/kimi-k3.1-pro-mi355x.jsx";

<AgentX data={[agentx, agentxMi355x]} />

## 1. Model Introduction

**Kimi K3.1 Pro** is Moonshot AI's open-weight, natively multimodal agentic model and the first model in the K3.1 series, post-trained from Kimi-K3 with large-scale joint reinforcement learning. It is a **2.8T-parameter Mixture-of-Experts model with 104B activated**: 93 layers (1 dense) combining **69 KDA + 24 Gated Sparse-MLA (DSA)** attention layers, **896 experts with 16 selected per token plus 2 shared**, SiTU-GLU activations, a 160K vocabulary, the MoonViT-V2 vision encoder (401M) for image input, and a **1,048,576-token** context. Weights are **MXFP4 with MXFP8 activations** (quantization-aware training). SGLang serves it as `KimiK31ForConditionalGeneration` and forces FP8 KV with page size 64.

Kimi K3.1 Pro always thinks; set the depth with the top-level `reasoning_effort` field: `low`, `high` or `max` (default `max`). For multi-turn conversations and tool calls, pass the complete assistant message back in `messages` as-is, including `reasoning_content` and `tool_calls`, not just `content`.

**Recommended generation:** `temperature=1.0`, `top_p=0.95` (fixed by the model; informational — do not hardcode in sample code).

**License:** [Kimi K3.1 Pro License](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/LICENSE).

**Resources:** [HuggingFace](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) · [Tech Blog](https://www.kimi.ai/blog/kimi-k3-1-pro) · [API Quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart) · [DFLASH draft](https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview).

## 2. Configuration Tips

### Recipe shapes

- **GB300**: every recipe runs on two trays (2×4 GPUs over MNNVL); the command adds the multi-node flags.
- **MI350X / MI355X**: one 8-GPU node per server; the 1P1D runs on two nodes (see [AMD MI350X / MI355X](#amd-tips)).
- **PD Mode**
  - `Unified` serves prefill and decode together.
  - `Prefill` / `Decode` split them (see [PD disaggregation](#pd-disaggregation)).
- **Prefill strategy** (B300, GB300)
  - `Low-Latency (TP8)` is flat TP8.
  - `High-Throughput (PP4 × PCP2)` is PP4 × PCP2 with MegaMoE EP2 (the cell carries MegaMoE; switch the MoE runner in the Advanced Features Playground): the most prefill tokens per second, and it serves the 1M window.
- **Unified strategy**
  - `Low-Latency (TP8)` (B300, GB300, MI350X, MI355X) is flat TP8.
  - `High-Concurrency (DCP8)` (B300, GB300) is TP8 · DCP8 · EP2.
  - `High-Throughput (DCP8)` (MI350X, MI355X) is TP8 · DCP8 over the a2a backend.
  - `Balanced` (other platforms) is the placeholder shape for each platform; the parentheses give its layout: TP8 on Vera Rubin, TP16 · DCP16 on B200 and GB200, TP16 · EP16 on H200, TP32 · EP32 on H100, TP32 · EP8 on TPU7x.
- **Decode strategy**
  - `Balanced (DCP8 · EP2)` is TP8 · DCP8 · EP2.
  - `High-Throughput (DCP8)` is the capacity-first decode: TP8 · DCP8, measured on B300 without speculation.
- **Attention Backend** (B300, GB300, B200, GB200; in the Advanced Features Playground)
  - `FlashMLA` is the default DSA kernel and keeps the reference FP8 KV format.
  - `TRT-LLM` and `Prims-TS` add `--dsa-prefill-backend` / `--dsa-decode-backend`. They store the KV as plain FP8 and also run the query in FP8, so the numerics differ slightly from FlashMLA.
  - With PD disaggregation, launch the prefill and decode roles with the same backend.
- **MoE Runner** (B300, GB300, B200, GB200; in the Advanced Features Playground)
  - `TRT-LLM` is the default: the FlashInfer trtllm-gen fused MXFP4 MoE, with no extra flag.
  - `MegaMoE` adds `--moe-a2a-backend megamoe`, which sets EP to the TP size. On the flat TP cells it also adds `--shared-experts-tp-size` and the `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` cap (16384 for prefill and unified, 4096 for decode).
  - MegaMoE has only been measured on the High-Throughput (PP4 × PCP2) prefill; TRT-LLM on the other cells.
  - MegaMoE reserves a symmetric workspace that grows with EP size × the token cap; lower `--mem-fraction-static` if startup runs out of memory.
- **Spec Decode**
  - DFLASH with the [Kimi-K3.1-Dflash2-Preview](https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview) draft.
  - Block 8; block 6 on the High-Throughput (DCP8) decode.
- **HiCache**
  - Off by default; L2 and L3 keep DCP on Kimi K3.1 Pro.
  - On a PD decode server HiCache also turns on the decode radix cache.
  - On MI350X / MI355X, `L1+L2 (host)` is the default except on Low-Latency and the PD decode (see [AMD MI350X / MI355X](#amd-tips)).

### Tuning notes

- **Memory.** The KDA state pool (FP32, ~52 MB per request per rank at TP8) sets the concurrency ceiling; the MLA + indexer KV (~18.9 KB per token) fills the rest. TP and PCP replicate the MLA KV; DCP is the only axis that shards it.
- **DFLASH.** Decode needs `--enable-linear-replayssm-spec` (the cells carry it) and an explicit `--max-running-requests`. The draft KV ring is on by default and adds ~15% target KV.
- **MegaMoE.** Non-prefill roles must set `--max-running-requests`; do not add `--enable-symm-mem` (it deadlocks).
- **PP prefill.** Keep `SGLANG_PP_LAYER_PARTITION=24,24,24,21` and `CUDA_MODULE_LOADING=EAGER` (both in the High-Throughput prefill cell).
- **HiCache.** Use `--hicache-io-backend direct --hicache-mem-layout page_first_direct` with `--mamba-max-states-per-path 1` (the HiCache overlay adds them).

<a id="amd-tips" />

### AMD MI350X / MI355X

- Requires ROCm 10; the recipes have not run on ROCm 7 images.
- **Environment.** Every MI350X / MI355X cell sets `SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1`, as on [Kimi-K3](/cookbook/autoregressive/Moonshotai/Kimi-K3#amd-env): the AITER ROCm path, its K3 fused kernels, the FlyDSL MoE kernels and the SiTU v2 MoE with A8W4 activation quantization.
- **Pick the recipe by agent count.**
  - 1 to 4: `Unified` `Low-Latency (TP8)`, on the monokernel.
  - 8 to 56: `Unified` `High-Throughput (DCP8)` with HiCache; set `--max-running-requests` and `--cuda-graph-max-bs-decode` to the agent count. The cell is the 48-agent shape: 32k prefill chunks every 8 decode steps, block 5, and the row-sharded prefill (`SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT=1`, `SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM=1`) with `SGLANG_OPT_DFLASH_RING_LOCS_FUSED=1`. From 8 to 32 agents, use `--chunked-prefill-size 16384`, `--prefill-decode-interval 16` and block 6, without those three variables.
  - Past 56: the 1P1D, `High-Throughput` prefill and decode on two nodes (see [On AMD MI355X](#pd-amd)).
- **DFLASH** proposes 5 draft tokens (block 6) on the Triton draft attention backend, 4 (block 5) on the single-node `High-Throughput` cell from 48 agents, with ReplaySSM on every role but the PD prefill. The single-node cells add `--speculative-use-block-verification`, which the measured acceptance used for sampled drafts.
- **Monokernel (1 to 4 agents).** `--enable-amd-monokernel` with `SGLANG_AMD_MONO_K31_WIDE=1` runs decode and verify steps through the persistent kernels of `sgl-amd-mono` (in the image); other steps use the standard forward. It takes TP8 or DCP8 with DFLASH but not HiCache, and SGLang stops at startup with the reasons if a setup is outside what it supports. The log shows `[k31 wide] first wide step` once it engages. The `Low-Latency` cell keeps 150 KDA state slots (`--max-mamba-cache-size 150`) so agent sessions' prefixes stay cached.
- **HiCache.** `L1+L2 (host)` is the default: write-back, `--hicache-ratio 2.5` and `SGLANG_HICACHE_MAMBA_HOST_RATIO=10` (8 on the PD prefill). Low-Latency and the PD decode run without it.
  - On agentic replay the host KDA-state slots bound the prefix hit rate before the host KV does, so `SGLANG_HICACHE_MAMBA_HOST_RATIO` sizes them apart from `--hicache-ratio`. The recipes keep every KDA state per path; `--mamba-max-states-per-path 1` lowered the hit rate.
  - The host pools pin 170-185 GB per GPU at DCP8, and the node's 8 ranks share its GTT limit (half of RAM, 1511 GiB on MI355X). At ratio 12 they overflowed it at 32 agents, and `hipHostMalloc` failed at startup. `HSA_USERPTR_FOR_PAGED_MEM=0` backs them with `hipHostMalloc` (GTT memory, which never migrates); page migration in registered host memory stalls every GPU queue.
  - L3 has not run on MI350X / MI355X.
- **Memory.** `SGLANG_ROCM_RUNTIME_RESERVE_GB` (10 at DCP8, 6 at TP8) keeps GPU memory free for the HIP queues' scratch, which is allocated outside torch. `--mem-fraction-static` 0.87 or 0.88 overflowed GTT at startup or ran slower on the single-node recipe.
- **`--prefill-decode-interval`** sets how many decode steps run after each prefill batch before the next prefill. The single-node `High-Throughput` cell runs 8 with 32k prefill chunks from 48 agents; below 48 agents, 16 with 16k chunks.
- **MI350X** runs the MI355X commands, single-node and 1P1D.

## 3. Advanced Usage

### 3.1 Reasoning

The `kimi_k3` reasoning parser puts thinking in `message.reasoning_content` and the answer in `message.content`.

<Accordion title="Reasoning Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="moonshotai/Kimi-K3.1-Pro",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
    reasoning_effort="high",  # "low" | "high" | "max" (default max)
)
msg = resp.choices[0].message
print("Reasoning:", getattr(msg, "reasoning_content", None))
print("Answer:", msg.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Pending update...
```

</Accordion>

### 3.2 Tool Calling

The `kimi_k3` tool-call parser surfaces structured calls in `message.tool_calls`; print `reasoning_content` too, since the model thinks before calling.

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
    model="moonshotai/Kimi-K3.1-Pro",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
)
msg = resp.choices[0].message
print("Reasoning:", getattr(msg, "reasoning_content", None))
print("Tool calls:", msg.tool_calls)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Pending update...
```

</Accordion>

<a id="pd-disaggregation" />

### 3.3 PD Disaggregation

- **Transfer**: Mooncake (DCP rejects other backends on either role). Prefill listens on `30000`, decode on `30100`; the `8998` after `--prefill` must match `--disaggregation-bootstrap-port`.
- **When using DFLASH, run it on both roles**: the prefill writes the draft KV ring that decode reads.
- **Restart the prefill role whenever the decode role restarts**: prefill keeps the old decode's Mooncake mapping.

<Accordion title="Router">

Use the Kimi K3.1 Pro build of `sgl-router` (the stock `sglang-router` 0.3.2 lacks the K3 chat rendering).

```bash Command
SGLANG_ROUTER_STREAM_IDLE_TIMEOUT_SECS=3600 \
sgl-router \
  --pd-disaggregation \
  --prefill http://<prefill-host>:30000 8998 \
  --decode http://<decode-host>:30100 \
  --host 0.0.0.0 --port 8000 \
  --policy power_of_two \
  --model-id moonshotai/Kimi-K3.1-Pro \
  --tokenizer-path <Kimi-K3.1-Pro tokenizer.json> \
  --request-timeout-secs 540
```

</Accordion>

**Measured 1P1D** (B300 2×8, SWA1024 draft, `High-Throughput` prefill without the prefill CUDA graph + `Decode` Balanced):

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "40%"}} />
    <col style={{width: "60%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Workload</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr style={{backgroundColor: "rgba(255,255,255,0.02)"}}>
      <td style={{padding: "9px 12px"}}>Prefill, 90% prefix hit, TTFT p90 ≤ 15 s</td>
      <td style={{padding: "9px 12px"}}>362,093 tok/s at concurrency 32</td>
    </tr>
    <tr style={{backgroundColor: "rgba(255,255,255,0.05)"}}>
      <td style={{padding: "9px 12px"}}>Decode, 8K context, batch 64</td>
      <td style={{padding: "9px 12px"}}>2,583 tok/s (TPOT p90 26.07 ms, AL 2.77)</td>
    </tr>
    <tr style={{backgroundColor: "rgba(255,255,255,0.02)"}}>
      <td style={{padding: "9px 12px"}}>Decode, 8K context, concurrency 1</td>
      <td style={{padding: "9px 12px"}}>TPOT 4.89 ms (AL 2.96); 9.6 ms without speculation</td>
    </tr>
    <tr style={{backgroundColor: "rgba(255,255,255,0.05)"}}>
      <td style={{padding: "9px 12px"}}>AIME26 ×8</td>
      <td style={{padding: "9px 12px"}}>1.000</td>
    </tr>
  </tbody>
</table>

<a id="pd-amd" />

#### On AMD MI355X

- **Container**: add `--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 --network=host` to the usual ROCm flags; Mooncake runs over RDMA and finds the RoCE NICs itself.
- **Prefill host tier**: HiCache's host pools are pinned and count against the node's GTT limit, so a tier larger than that fails at startup. If yours does, lower `--hicache-ratio` or `SGLANG_HICACHE_MAMBA_HOST_RATIO`.
- **Router**: the panel's MI355X router (`sglang_router` from the ROCm image) is the one measured.

**Measured 1P1D** (MI355X 2×8, `High-Throughput` prefill + `High-Throughput` decode, AgentX at concurrency 92 for 3600 s, acceptance length simulated at 3.399):

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "40%"}} />
    <col style={{width: "60%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Workload</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr style={{backgroundColor: "rgba(255,255,255,0.02)"}}>
      <td style={{padding: "9px 12px"}}>AgentX, concurrency 92, 3600 s</td>
      <td style={{padding: "9px 12px"}}>17,883 tok/s per GPU at 30.6 tok/s per user (P90 ITL 32.7 ms)</td>
    </tr>
    <tr style={{backgroundColor: "rgba(255,255,255,0.05)"}}>
      <td style={{padding: "9px 12px"}}>Time to first token</td>
      <td style={{padding: "9px 12px"}}>P50 3.66 s, P90 10.26 s (94.4% prefix hit)</td>
    </tr>
  </tbody>
</table>

<a id="tpu" />

### 3.4 TPU7x deployment

Kimi K3.1 Pro runs on a 16-chip TPU7x slice (four hosts) as one TP32 · EP8 engine through [SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu). Run its recipe on each host:

```bash Command
python3 -m benchmark.tpu.launch_server --config recipe/configs/kimi_k3_1/server.py
```

Single request, all 93 layers: 133 tok/s at 8K input (TPOT 7.52 ms) and 112 tok/s at 128K. With the experimental decode megakernel and DFlash, single-user decode reaches up to about 590 tok/s.

DFlash ([`RadixArk/Kimi-K3.1-Dflash2-Preview`](https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview)) serves one request at a time. Structured output, tool-call grammar and image input are not yet supported on TPU7x.
