---
title: Hy4 preview
description: "Deploy Tencent Hy4 preview with SGLang — launch recipes for the 770B-parameter Mixture-of-Experts model with MLA, DeepSeek Sparse Attention (DSA), and MTP speculative decoding, in BF16 on H200/B200/B300/GB300 and MXFP8 on Blackwell GPUs."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all methods and hardware platforms, see the [official SGLang installation guide](/docs/get-started/install). The Docker path below matches the **Docker** framing in the command panel.

<Tabs>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:hy4-preview
```

<Note>
The `hy4-preview` image bundles the HYV4 model code, the suffix-aware `hunyuan` reasoning/tool-call parsers, and the NEXTN MTP runtime. Switch to `:latest` once a tagged release picks them up.
</Note>

For how to launch the image, see [Install → Method 3: Using Docker](/docs/get-started/install#method-3-using-docker), substituting the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick your hardware + recipe to generate the launch command. The two serving strategies map to whether the MTP (NextN) draft layer is on:

- **Low-Latency** — MTP speculative decoding on (steps=3, draft-tokens=4). Fastest reply for a single user; pick for chat.
- **High-Throughput** — MTP off. At saturation the draft+verify overhead outweighs the speedup; best for batch jobs.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/tencent/hy4-preview.jsx";
import { benchmarks } from "/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

<div style={{fontSize: "0.85em", lineHeight: "1.55", color: "#6b7280", margin: "0.5rem 0 1rem 0"}}>
  <p style={{margin: "0 0 0.3rem 0"}}><strong>Panel controls</strong> (top of the command box):</p>
  <ul style={{margin: 0, paddingLeft: "1.25rem"}}>
    <li style={{marginBottom: "0.2rem"}}><strong>Python / Docker</strong> — bare <code>sglang serve …</code> for an existing SGLang env, or a <code>docker run … sglang serve …</code> wrap against the per-hardware image from the <a href="#install">Install SGLang</a> panel above.</li>
    <li style={{marginBottom: "0.2rem"}}><strong>⧉ Copy</strong> — copies the current command (with whichever framing is active) to your clipboard.</li>
    <li style={{marginBottom: "0.2rem"}}><strong>$ cURL</strong> — a sample request against <code>localhost:30000</code> to confirm the server is up.</li>
    <li style={{marginBottom: "0.2rem"}}><strong>⚙ Env</strong> — edits the placeholders (<code>HOST_IP</code>, <code>PORT</code>, <code>HF_TOKEN</code>, <code>NODE_RANK</code>, <code>NODE0_IP</code>) the command and cURL share. Persists in localStorage across cookbooks.</li>
    <li><strong>Badge</strong> — single-node recipes show green <strong>Verified</strong> (run end-to-end on real hardware); the 2-node BF16 recipes show <strong>In Progress</strong> and flip to <strong>Verified</strong> as their multi-node verification lands.</li>
  </ul>
</div>

## Playground

The Playground lets you turn on additional knobs on top of whichever Deploy cell is currently selected. The base is read live from your Deploy selection — only your overrides change.

The knobs come in two flavors:

- **Built-in SGLang features** — parallelism overrides (TP / DP-Attention), the MoE a2a backend (DeepEP, which sets EP = TP), reasoning / tool-call parsers, speculative-decoding presets, prefill/decode disaggregation, and HiCache tiers.
- **Hy4 specific** — `--reasoning-parser auto` / `--tool-call-parser auto` resolve to the suffix-aware Hunyuan parsers and read the real structural-token strings from the tokenizer vocab at runtime.

Lines highlighted **green** are added by your overrides; lines with **red strikethrough** were in the base recipe but stripped by an override. Parallelism combinations beyond the listed recipes (DP-Attention, DeepEP, other TP degrees) are experimentation territory — any override flips the badge to **Not Verified** until the configuration is run end-to-end and submitted back.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

<div style={{fontSize: "0.85em", lineHeight: "1.55", color: "#6b7280", margin: "0.5rem 0 1rem 0"}}>
  <p style={{margin: "0 0 0.3rem 0"}}><strong>Panel controls</strong> reuse <strong>Python / Docker</strong> · <strong>⧉ Copy</strong> · <strong>$ cURL</strong> · <strong>⚙ Env</strong> from the Deploy panel, plus one extra:</p>
  <ul style={{margin: 0, paddingLeft: "1.25rem"}}>
    <li><strong>Submit ↗</strong> — opens a pre-filled GitHub issue so you can land your override combo as a new verified cookbook cell. Shown only while the badge says <strong>Not Verified</strong>; click it once you've actually run the command on your hardware and confirmed it works.</li>
  </ul>
</div>

## 1. Model Introduction

**Hy4 preview** is Tencent's next-generation flagship Mixture-of-Experts language model: 770B total parameters with 49B active per token, pairing a DeepSeek-style MLA + sparse-attention stack with Hunyuan's own residual control, MoE routing, and attention gating. It is a **text-only** model (vocab 120,832; the endpoint rejects image input by design), released under **Apache-2.0**.

**Key architecture:**

- **MoE**: 78 layers — layer 0 is a dense MLP, the remaining 77 are sparse MoE with 256 routed experts + 1 shared expert, top-8 sigmoid-scored routing (routed scaling 2.827), expert intermediate size 2048, and bounded SwiGLU (clamp 10.0).
- **MLA + DSA on every layer**: Multi-head Latent Attention (`q_lora_rank` 2048, `kv_lora_rank` 512, 192 nope + 64 rope head dims, `v_head_dim` 256) under DeepSeek Sparse Attention — indexer top-k 2048 with 32 index heads, indexers alternating full/shared (one full indexer every 4 layers), and an FP8 index cache.
- **iHC residual control**: Hunyuan's own inter-layer residual gating (`enable_ihc`, `hc_mult` 4) with pre- and post-residual gate groups. Semantically distinct from DeepSeek-V4's mHC (no combination step, no Sinkhorn) — the implementations are not interchangeable.
- **Gated MLA + attention sink**: element-wise attention output gating evaluated in fp32, plus a learnable per-head attention sink propagated through all attention backends.
- **MTP (NextN)**: one built-in multi-token-prediction draft layer (`model.mtp_layers.0`, 10B parameters with ~0.7B active, present in both checkpoints) enabling speculative decoding out of the box.
- **Long context**: 1M max positions (RoPE theta 1e7). Size `--context-length` to your KV budget — the sizing table in §2 suggests 262,144 (131,072 on H200).

**Available models:**

- [tencent/Hy4-preview](https://huggingface.co/tencent/Hy4-preview) — BF16 (~1.5TB weights)
- [tencent/Hy4-preview-FP8](https://huggingface.co/tencent/Hy4-preview-FP8) — MXFP8 (ModelOpt recipe, UE8M0 group-32 weight scales, dynamic activations; ~760GB weights)

**Recommended generation:** `temperature=0.9`, `top_p=1.0` (informational — SGLang applies the checkpoint's `generation_config.json` defaults; don't hardcode sampling parameters in client code). Thinking depth is controlled per request via `reasoning_effort` (defaults to `high`; `no_think` for direct responses — see §3.1).

**Resources:** [HuggingFace](https://huggingface.co/tencent/Hy4-preview) · [GitHub](https://github.com/Tencent-Hunyuan/Hy4-preview).

**Special tokens.** The Hy4 preview tokenizer's structural tokens are suffix-bearing (`<think:opensource>`, `<tool_calls:opensource>`, `<tool_call:opensource>`, `<arg_key:opensource>`, `<arg_value:opensource>`). SGLang's Hunyuan reasoning/tool-call parsers resolve the real token strings from the tokenizer vocab at runtime, so `--reasoning-parser auto --tool-call-parser auto` work out of the box.

## 2. Configuration Tips

**Hardware sizing.** BF16 weights are ~1.5TB and MXFP8 ~760GB; the MLA KV cache (compressed `kv_lora` 512 + rope 64, plus the DSA FP8 indexer cache, ≈95KB/token) is replicated per TP rank, so the per-rank pool left after weights sets the context ceiling:

<table style={{width: "100%", borderCollapse: "collapse"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #0052d9"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.02)"}}>GPU</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.05)"}}>VRAM</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.02)"}}>MXFP8 (~760GB)</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.05)"}}>BF16 (~1.5TB)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>H200</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>141GB</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Not supported — the MXFP8 kernel path requires SM100 (Blackwell)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>TP16, 2×8 nodes · 131K context</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>B200</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>192GB</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>TP8, single node · 262K context</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>TP16, 2×8 nodes · 262K context (8×192GB ≈ the weights alone)</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>B300</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>288GB</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>TP4, single node · 262K context (~190GB/rank)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>TP8, single 8-GPU node · 262K context</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>GB300</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>288GB</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>TP4, single node · 262K context</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>TP8, 2×4 nodes · 262K context (GB300 hosts carry 4 GPUs)</td>
    </tr>
  </tbody>
</table>

On H200 (BF16, TP16 → ~95GB of weights per rank) the per-rank pool holds roughly 260K tokens of KV — set `--context-length 131072` there (the model default is 1M positions, far beyond the pool); the 192GB+ parts comfortably support 262144.

**DSA attention backend.** Every layer runs DeepSeek Sparse Attention; SGLang auto-selects the DSA backend for HYV4 (`--attention-backend dsa` with `flashmla_sparse` prefill/decode and an FP8 indexer cache over a bf16 KV pool), so the recipes don't pass attention flags. Override only with a kernel-specific reason.

**MXFP8 kernel stack.** The MXFP8 checkpoint self-describes via its ModelOpt `hf_quant_config` (dynamic activations, UE8M0 group-32 weight scales) — no `--quantization` flag needed; the recipes pin `--moe-runner-backend deep_gemm --fp8-gemm-backend deep_gemm`, the validated HYV4 MXFP8 path (the runtime also defaults both to `deep_gemm` for HYV4 when the flags are left unset). The MXFP8 kernel path requires SM100+ (Blackwell); H200 (SM90) cannot serve the MXFP8 checkpoint — use BF16 there.

**CUDA graph decode vs eager.** Decode CUDA-graph capture is on by default; long-duration soak validation of the graph path on Hy4 is still in progress. If you hit instability under long mixed agentic workloads, pass `--disable-cuda-graph` to fall back to eager decode (a restart recovers cleanly either way).

**MTP (NextN) speculative decoding.** Both checkpoints ship one draft layer; the preset is `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4` (top-k 1). Speculative decoding reserves 4 draft-token slots per request, so the effective request budget is `prompt_tokens + max_tokens + 4 ≤ context length` — requests at the exact context boundary are rejected with the reservation accounted for.

**Fail-fast guardrails.** The model rejects pipeline parallelism and `--enable-prefill-cp` before allocation. The recipes run pure TP; DP-Attention, DeepEP (which sets EP = TP at runtime), and other TP degrees are Playground experimentation territory.

**Multi-node BF16.** The BF16 weights don't fit a single H200/B200/GB300 host, so those cells are 2-node TP recipes — run the generated command on every node (the panel injects `--nnodes 2 --node-rank --dist-init-addr`) and keep the weights on storage shared across ranks.

**Text-only.** Image input is rejected with HTTP 400 by design — don't route vision traffic to this endpoint.

**Large prefills under concurrency.** First-prefill latency on very large prompts can exceed 30 s under high concurrency; use a client timeout of 300 s (and moderate concurrency) for long-context agentic workloads instead of the common 30 s default.

## 3. Advanced Usage

### 3.1 Reasoning (`reasoning_effort`)

Hy4 preview is a hybrid-thinking model: reasoning defaults to `high` (deep chain-of-thought — suits math, coding, and agentic tasks). The top-level OpenAI `reasoning_effort` field accepts the standard tiers (e.g. `high`); for direct responses, pass the model-specific `no_think` through `extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}}`. The Deploy recipes enable the reasoning parser (`--reasoning-parser auto`) so thinking is separated into `reasoning_content` and the final answer into `content`:

<Accordion title="Example: thinking (reasoning_effort=high) (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="tencent/Hy4-preview-FP8",
    messages=[{"role": "user", "content": "Solve step by step: What is 15% of 240?"}],
    reasoning_effort="high",
    max_tokens=2048,
)

msg = response.choices[0].message
print("=============== Thinking =================")
print(msg.reasoning_content)
print("=============== Content =================")
print(msg.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Pending update — will be captured verbatim from a live Hy4 preview server.
```

</Accordion>

<Accordion title="Example: instant mode (reasoning_effort=no_think) (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="tencent/Hy4-preview-FP8",
    messages=[{"role": "user", "content": "Give me a one-line summary of relativity."}],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
    max_tokens=256,
)

print("Content:", response.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Pending update — will be captured verbatim from a live Hy4 preview server.
```

</Accordion>

### 3.2 Tool Calling

Hy4 preview emits tool calls through suffixed structural tokens with an `arg_key` / `arg_value` argument format; SGLang's Hunyuan tool-call parser reassembles them into OpenAI-compatible `message.tool_calls` with schema-aware type coercion, for both streaming and non-streaming requests. The Deploy recipes enable both parsers together (`--reasoning-parser auto --tool-call-parser auto`) — the reasoning parser strips thinking tokens before the tool-call parser runs.

<Note>
Tool-call output is parsed, not grammar-constrained: `tool_choice: "required"` / named-function forcing is not enforced with structural-tag guided decoding on the current implementation.
</Note>

<Accordion title="Example: non-streaming tool call (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="tencent/Hy4-preview-FP8",
    messages=[{"role": "user", "content": "What's the weather in Beijing? Use fahrenheit."}],
    tools=tools,
)

msg = response.choices[0].message
print("Reasoning:", msg.reasoning_content)
print("Content:  ", msg.content)
for tc in msg.tool_calls or []:
    print(f"Tool Call: {tc.function.name}")
    print(f"  Arguments: {tc.function.arguments}")
```

</Accordion>

<Accordion title="Example Output">

```text Output
Pending update — will be captured verbatim from a live Hy4 preview server.
```

</Accordion>
