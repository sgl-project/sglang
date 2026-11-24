---
title: Kimi-K3.1-Pro
description: "Deploy Moonshot AI's Kimi K3.1 Pro with SGLang: 2.8T parameters (104B active) and a 1M context, on NVIDIA Vera Rubin, Blackwell and Hopper, AMD Instinct, PPU, Ascend NPUs and Google TPU."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

Install [SGLang](https://github.com/sgl-project/sglang) from source on `main`. The Kimi K3.1 Pro image recipe, with every dependency in place, is [`docker/k31.Dockerfile`](https://github.com/sgl-project/sglang/blob/37794659f6732c59288acc83d21cad481a6c2cd2/docker/k31.Dockerfile); it packages the SGLang source of commit `37794659f6` on the Kimi K3.1 branch, not `main`. Build it from a checkout of that commit with `docker buildx build -f docker/k31.Dockerfile .`.

<Tabs>

<Tab title="Python (from source)">

```bash Command
git clone https://github.com/sgl-project/sglang.git && cd sglang
pip install -e "python"
```

</Tab>

<Tab title="Docker">

Use the Kimi K3.1 Pro image `lmsysorg/sglang:dev-kimi-k31-pro` (`lmsysorg/sglang:dev-kimi-k31-pro-mi35x` on AMD MI350X / MI355X, ROCm 10) and substitute the inner `sglang serve ...` with what the command generator below produces. See [Install → Method 3: Using Docker](/docs/get-started/install#method-3-using-docker).

</Tab>

<Tab title="PPU (M890 / 810E)">

Use the PPU image `egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun-tmp/inference-xpu-pytorch:kimi-k3.1-pro-sglang0.5.22-cuda13.0-pytorch2.13.0-20261103` with a prepared [SAIL environment](/docs/hardware-platforms/sail), and run the command panel's **Python** output inside the container on each worker node.

</Tab>

<Tab title="NPU (A3 Series)">

```bash Command
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.2.0-a3-B050
```

The A3 Series image is a CANN 9.2.0 build (tag `B050`); use a build whose SGLang carries Kimi K3.1 Pro support. For the host driver, CANN and container setup, follow the [Ascend installation guide](/docs/hardware-platforms/ascend-npus/getting-started/installation) and the [quick start guide](/docs/hardware-platforms/ascend-npus/getting-started/quick_start), then run the command panel's output on all four nodes (see [Ascend A3 Series](#npu-tips)). Replace `/path/to/Kimi-K3.1-Pro-w4a8` with the directory where the checkpoint is mounted on every node.

**Weights (NPU):** [Kimi-K3.1-Pro-w4a8](https://www.modelscope.cn/models/Eco-Tech/Kimi-K3.1-Pro-w4a8) (A3 Series, ModelSlim W4A8).

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

On `Unified`, `--mamba-full-memory-ratio` follows your average request length. The prefill and MI350X / MI355X High-Throughput recipes pin the ratio they were measured with; the decode and MI350X / MI355X Low-Latency recipes size the KDA pool directly, so the ratio does not apply. On the Ascend A3 Series the ratio does not apply either: the cell pins `--max-mamba-cache-size 64` and the Ascend path takes no ratio.

<KimiK31MambaRatioCalculator />

<a id="playground" style={{ scrollMarginTop: "96px" }} />

## Advanced Features Playground

Turn on additional knobs on top of whichever cell the Deploy panel is showing:

- **Serving:** parsers, the PD transfer, HiCache storage and write policy, DFLASH proposed tokens, the prefill CUDA graph and the PD decode radix cache.
- **WideEP Decode** (GB300 `Decode`): one decode server on four trays, attention DP2 with TP8 · DCP8 in each 8-GPU group and MegaMoE EP16 across all 16 GPUs. In a fixed-batch survey it beat 8-GPU decode workers from 8 concurrent requests up, by 15% at 64. That survey ran without the radix cache; turning the PD decode radix cache on still keeps one KDA state slot per request, since the decode tree caches KV only. Not Verified on the current code.
- **Tuning:** the DSA attention backend and the MoE runner (B300, GB300, B200, GB200), the DCP communication backend, the KDA state slot count, KDA states per radix path, the MegaMoE token cap (when MoE Runner is MegaMoE) and the PP layer partition (PP4 × PCP2 prefill).

On the Ascend A3 Series cell only the **Parsers** card applies: the recipe runs Non-Spec, without HiCache, and its MoE runner and graph come from the deployment script.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## Agentic Long-Context Deployment

Agentic workloads, such as coding agents and other multi-turn, tool-using assistants, send long prompts whose prefixes grow and repeat from turn to turn. Serving them well depends on reusing the KV cache across turns, offloading it beyond GPU memory, and keeping each session on the workers that already hold its prefix. This section gives complete launch commands for such deployments on GB300 with Dynamo and SGLang, and on MI355X with SGLang, measured on the AgentX agentic-coding trace replay. Pick a deployment shape, concurrency, KV cache offloading mechanism and router to get every process of the deployment as a raw command, one tab per process in launch order.

<Note>
  These commands are a **status snapshot** (2026-11-03) of a GB300 tuning campaign that is still running. They are not an InferenceX submission. Each was measured with the DFlash acceptance length fixed to the draft's expected value (see the note under each configuration). The aggregated deployment at concurrency 1 missed its time-to-first-token target; the other seven points met theirs.
</Note>

- **Router.** Every configuration was measured behind Dynamo (`dynamo.frontend` with `dynamo.sglang` workers and the KV-aware router with session affinity). The SGLang router (`sgl-router` in front of `sglang.launch_server` workers) is derived from the same engine settings and was not measured.
- **KV offload.** The disaggregated prefill workers were measured with HiCache in host DRAM; the aggregated workers without KV offload. The other tiers (none, HiCache, or a Mooncake store attached through the external linker) are derived and were not measured.
- **Container image.** The commands show `<K31_IMAGE>`. The measured image was built with `docker/k31.Dockerfile` from SGLang commit `9ab3c28176` plus one unmerged change, and bundles FlashInfer 0.7.2, `sgl-deep-gemm` and a Dynamo 1.6.0 build with Kimi K3.1 Pro and session-affinity patches. `lmsysorg/sglang:dev-kimi-k31-pro` has not been checked against these commands.
- **Draft.** The commands load the [Kimi-K3.1-Dflash2-Preview](https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview) draft by its Hugging Face id; the measured runs read a local copy of it.

On MI355X:

- **Shape.** One node up to 56 agents: TP8 on the monokernel for 1 and 4, TP8 · DCP8 with HiCache from 8. The 1P1D runs on two nodes.
- **Image.** `lmsysorg/sglang:dev-kimi-k31-pro-mi35x`; the recipes were measured on `sha256:5901db14`.

import { AgentX } from "/src/snippets/_agentx.jsx";
import { agentx } from "/src/snippets/agentx/moonshotai/kimi-k3.1-pro.jsx";
import { agentxMi355x } from "/src/snippets/agentx/moonshotai/kimi-k3.1-pro-mi355x.jsx";

<AgentX data={[agentx, agentxMi355x]} />

## 1. Model Introduction

**Kimi K3.1 Pro** is Moonshot AI's open-weight, natively multimodal agentic model and the first model in the K3.1 series, post-trained from Kimi-K3 with large-scale joint reinforcement learning. It is a **2.8T-parameter Mixture-of-Experts model with 104B activated**: 93 layers (1 dense) combining **69 KDA + 24 Gated Sparse-MLA (DSA)** attention layers, **896 experts with 16 selected per token plus 2 shared**, SiTU-GLU activations, a 160K vocabulary, the MoonViT-V2 vision encoder (401M) for image input, and a **1,048,576-token** context. Weights are **MXFP4 with MXFP8 activations** (quantization-aware training). SGLang serves it as `KimiK31ForConditionalGeneration` and forces FP8 KV with page size 64. The Ascend A3 Series recipe serves a ModelSlim **W4A8** build instead (`--quantization modelslim`).

Kimi K3.1 Pro always thinks; set the depth with the top-level `reasoning_effort` field: `low`, `high` or `max` (default `max`). For multi-turn conversations and tool calls, pass the complete assistant message back in `messages` as-is, including `reasoning_content` and `tool_calls`, not just `content`.

**Recommended generation:** `temperature=1.0`, `top_p=0.95` (fixed by the model; informational — do not hardcode in sample code).

**License:** [Kimi K3.1 Pro License](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/LICENSE).

**Resources:** [HuggingFace](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) · [Tech Blog](https://www.kimi.ai/blog/kimi-k3-1-pro) · [API Quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart) · [DFLASH draft](https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview).

## 2. Configuration Tips

### Recipe shapes

- **GB300**: every recipe runs on two trays (2×4 GPUs over MNNVL); the command adds the multi-node flags.
- **MI350X / MI355X**: one 8-GPU node per server; the 1P1D runs on two nodes (see [AMD MI350X / MI355X](#amd-tips)).
- **A3 Series (NPU)**: four nodes, each an 8-card / 16-die A3 node, run as one 64-die engine; Unified only (see [Ascend A3 Series](#npu-tips)).
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
  - `Balanced (TP16 · PP4 · EP16)` (A3 Series) is one PP stage per node over four nodes.
  - `Balanced` (other platforms) is the placeholder shape for each platform; the parentheses give its layout: TP8 on Vera Rubin, TP16 · DCP16 on B200 and GB200, TP16 · EP16 on H200, TP32 · EP32 on H100, TP32 · EP8 on TPU7x.
- **Decode strategy**
  - `Balanced (DCP8 · EP2)` is TP8 · DCP8 · EP2.
  - `High-Throughput (DCP8)` is the capacity-first decode: TP8 · DCP8, measured on B300 without speculation.
- **Attention Backend** (B300, GB300, B200, GB200; in the Advanced Features Playground)
  - `Prims-TS` is the default DSA kernel on these GPUs. It stores the KV as plain FP8 and also runs the query in FP8, so the numerics differ slightly from FlashMLA.
  - `FlashMLA` and `TRT-LLM` add `--dsa-prefill-backend` / `--dsa-decode-backend`. FlashMLA keeps the reference FP8 KV format and is the default on other GPUs and with HiSparse; TRT-LLM stores plain FP8 like Prims-TS.
  - With PD disaggregation, launch the prefill and decode roles with the same backend.
- **MoE Runner** (B300, GB300, B200, GB200; in the Advanced Features Playground)
  - `TRT-LLM` is the default: the FlashInfer trtllm-gen fused MXFP4 MoE, with no extra flag.
  - `MegaMoE` adds `--moe-a2a-backend megamoe`, which sets EP to the TP size. On the flat TP cells it also adds `--shared-experts-tp-size` and the `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` cap (16384 for prefill and unified, 4096 for decode).
  - MegaMoE has only been measured on the High-Throughput (PP4 × PCP2) prefill; TRT-LLM on the other cells.
  - MegaMoE reserves a symmetric workspace that grows with EP size × the token cap; lower `--mem-fraction-static` if startup runs out of memory.
- **Spec Decode**
  - DFLASH with the [Kimi-K3.1-Dflash2-Preview](https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview) draft.
  - Block 8; block 6 on the High-Throughput (DCP8) decode.
  - Not offered on the A3 Series: its recipe runs Non-Spec.
- **HiCache**
  - Off by default; L2 and L3 keep DCP on Kimi K3.1 Pro.
  - On a PD decode server HiCache also turns on the decode radix cache.
  - On MI350X / MI355X, `L1+L2 (host)` is the default except on Low-Latency and the PD decode (see [AMD MI350X / MI355X](#amd-tips)).

### Tuning notes

- **Memory.** The KDA state pool (FP32, ~52 MB per request per rank at TP8) sets the concurrency ceiling; the MLA + indexer KV (~18.9 KB per token) fills the rest. TP and PCP replicate the MLA KV; DCP is the only axis that shards it.
- **DFLASH.** Decode needs `--enable-linear-replayssm-spec` (the cells carry it) and an explicit `--max-running-requests`. On NVIDIA, the DFLASH commands set `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1` to turn on the draft KV ring, which adds ~15% target KV; without it the ring is off.
- **MegaMoE.** Non-prefill roles must set `--max-running-requests`; do not add `--enable-symm-mem` (it deadlocks).
- **PP prefill.** Keep `SGLANG_PP_LAYER_PARTITION=24,24,24,21` and `CUDA_MODULE_LOADING=EAGER` (both in the High-Throughput prefill cell).
- **HiCache.** Use `--hicache-io-backend direct --hicache-mem-layout page_first_direct` with `--mamba-max-states-per-path 1` (the HiCache overlay adds them).

<a id="amd-tips" />

### AMD MI350X / MI355X

- **ROCm 10** only. MI350X runs the MI355X commands.
- **By agent count.** 1-4: `Unified` `Low-Latency`, on the monokernel (no HiCache); at 1 agent run 1 request. 8-56: `Unified` `High-Throughput` with HiCache, at its 48-agent shape; at 56 agents run 56 requests, and below 48 use 16k chunks with 6, 8, 13 and 20 requests at 8, 16, 24 and 32 agents. Beyond that, the 1P1D (see [On AMD MI355X](#pd-amd)).
- **DFLASH.** Triton draft attention and block verification; block 8 at 1-4 agents, 6 at 8, and 5 from 16 agents and on both PD roles.
- **HiCache.** The pinned host pools share the node's GTT limit (1511 GiB on MI355X); larger ratios fail at startup.

<a id="npu-tips" />

### Ascend A3 Series

- **Shape.** Four 8-card nodes (16 dies each) as one 64-die engine: `--tp-size 16 --ep-size 16 --pp-size 4`, so each node is one PP stage. Unified only — prefill and decode share the four nodes.
- **Checkpoint.** A local ModelSlim **W4A8** build of [`Eco-Tech/Kimi-K3.1-Pro-w4a8`](https://www.modelscope.cn/models/Eco-Tech/Kimi-K3.1-Pro-w4a8) (`--quantization modelslim`), mounted on every node. The command shows `/path/to/Kimi-K3.1-Pro-w4a8`; replace it with your mount point — it must hold the config, tokenizer and quantization metadata, and it doubles as the served model name. `--mem-fraction-static 0.72` caps the static memory and leaves headroom for the KDA state pool and the a2a workspaces.
- **MoE.** `--moe-a2a-backend ascend_ops` with `--moe-ascend-ops-config '{"prefill":"alltoallv","decode":"low_latency"}'` picks the communication mode per phase on the same engine.
- **State pool.** `--max-mamba-cache-size 64` sizes the KDA state pool directly; `--mamba-full-memory-ratio` and the [Mamba ratio calculator](#mamba-ratio-calculator) do not apply.
- **No speculation, no HiCache.** The cell runs Non-Spec and without HiCache. The `SGLANG_ENABLE_SPEC_V2`, `SGLANG_RAGGED_VERIFY_MODE=static` and `SGLANG_DSPARK_*` environment is carried from the deployment script and is inert without a speculative algorithm.
- **Startup.** `--skip-server-warmup` skips the warm-up request, and `--cuda-graph-max-bs-decode 16` sizes the decode graph.
- **Network.** On every node, point `GLOO_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME` and `TP_SOCKET_IFNAME` at the cross-node NIC. `--node-rank` runs 0-3 and `--dist-init-addr` is the head node on port 20000; source both `set_env.sh` scripts (toolkit and ATB) before launching outside the image.

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

- **Transfer**: Mooncake (DCP rejects other backends on either role). Prefill listens on `30000`, decode on `30100`.
- **When using DFLASH, run it on both roles**: with the draft KV ring on, the prefill writes the ring that decode reads, so set `SGLANG_ENABLE_DFLASH_DRAFT_KV_RING` the same on both.
- **Restart the prefill role whenever the decode role restarts**: prefill keeps the old decode's Mooncake mapping.
- **Not on the A3 Series**: its recipe is Unified only, so there are no NPU prefill / decode roles here.

<Accordion title="Router">

```bash Command
sgl-router \
  --model-id moonshotai/Kimi-K3.1-Pro \
  --no-tokenizer \
  --worker-urls http://<prefill-host>:30000 http://<decode-host>:30100 \
  --host 0.0.0.0 --port 8000 \
  --policy power_of_two \
  --stream-idle-timeout-secs 3600 \
  --stale-request-timeout-secs 3600 \
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

- **Container**: add `--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 --network=host`. On Pollara (Ionic) NICs also mount the host's `libionic-rdmav34.so` and `/etc/libibverbs.d/ionic.driver`, as the panel's commands do; without them Mooncake moves KV over TCP.
- **Prefill host tier**: HiCache's host pools are pinned and count against the node's GTT limit, so a tier larger than that fails at startup. If yours does, lower `--hicache-ratio` or `SGLANG_HICACHE_MAMBA_HOST_RATIO`.
- **Router**: the MI355X numbers were measured behind an earlier router; the panel's MI355X command now runs `sgl-router`, which the ROCm image ships.

<a id="tpu" />

### 3.4 TPU7x deployment

Kimi K3.1 Pro runs on a 16-chip TPU7x slice (four hosts) as one TP32 · EP8 engine through [SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu). Run its recipe on each host:

```bash Command
python3 -m benchmark.tpu.launch_server --config recipe/configs/kimi_k3_1/server.py
```

Single request, all 93 layers: 133 tok/s at 8K input (TPOT 7.52 ms) and 112 tok/s at 128K. With the experimental decode megakernel and DFlash, single-user decode reaches up to about 590 tok/s.

DFlash ([`RadixArk/Kimi-K3.1-Dflash2-Preview`](https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview)) serves one request at a time. Structured output, tool-call grammar and image input are not yet supported on TPU7x.

### 3.5 Video Input

Kimi K3.1 Pro shares K3's multimodal processor: each `video_url` is sampled into
timestamped 4-frame chunks, laid out as Moonshot's hosted API does. See
[Kimi K3 video input](./Kimi-K3#video-input) for sampling details.

<Accordion title="Video Input Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="moonshotai/Kimi-K3.1-Pro",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/clip.mp4"},
                },
                {"type": "text", "text": "What happens in this video?"},
            ],
        }
    ],
)
print(resp.choices[0].message.content)
```

</Accordion>
