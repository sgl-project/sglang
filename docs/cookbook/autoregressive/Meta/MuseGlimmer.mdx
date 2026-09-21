---
title: Muse Glimmer
description: "A multimodal reasoning model served from a BF16, NVFP4 + MXFP8, vendor GGUF, or MLX checkpoint."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

See the [official SGLang installation guide](../../../docs/get-started/install) for all installation methods and hardware platforms. The steps below match the **Python** and **Docker** options in the command panel.

<Tabs>

<Tab title="Python (pip / uv)">

```bash Command
pip install --upgrade pip
pip install uv
# Muse Glimmer support is not in a release yet -- build the PR branch:
# https://github.com/sgl-project/sglang/pull/34262
git clone -b muse-glimmer https://github.com/sgl-project/sglang.git
cd sglang
uv pip install --prerelease=allow -e "python[all]"
```

Run the **Python** output of the command panel below in that environment.

</Tab>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:dev-muse-glimmer
```

See [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker) to start the image. Replace the inner `sglang serve ...` command with the command from the panel below.

</Tab>

</Tabs>

</Accordion>

Select a checkpoint format. Select whether to use speculative decoding:

- **Standard**: Use normal autoregressive decoding.
- **DFlash**: Use speculative decoding with the DFlash draft model. The draft serves as published, with no conversion step. See [§2](#2-configuration-tips).

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/meta-models/muse-glimmer.jsx";
import { benchmarks } from "/src/snippets/configs/meta-models/muse-glimmer-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

Use the Playground to test SGLang features that are not in the verified matrix. The Deploy panel above shows only combinations that the SGLang team has verified. The Playground lets you add more options to the command from the Deploy panel.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

Muse Glimmer is a multimodal reasoning model. You can serve Muse Glimmer in four formats:

- A BF16 checkpoint (`MuseGlimmerForConditionalGeneration`).
- A set of vendor GGUF files.
- A ready-to-serve NVFP4 + MXFP8 checkpoint.
- Three MLX repacks for Apple Silicon.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Form</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Source</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}><strong>BF16</strong></td>
      <td style={{padding: "9px 12px"}}><code>meta-models/Muse-Glimmer-30B</code></td>
      <td style={{padding: "9px 12px"}}>Supports image input.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong>GGUF Q4_K_M</strong></td>
      <td style={{padding: "9px 12px"}}><code>meta-models/Muse-Glimmer-30B-GGUF</code></td>
      <td style={{padding: "9px 12px"}}>Text only. This path is not optimized. SGLang shows a warning at startup.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong>NVFP4</strong></td>
      <td style={{padding: "9px 12px"}}><code>RadixArk/Muse-Glimmer-NVFP4</code></td>
      <td style={{padding: "9px 12px"}}>Text only. Ready to serve, no conversion needed.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong>MLX Q4</strong></td>
      <td style={{padding: "9px 12px"}}><code>RadixArk/Muse-Glimmer-q4-MLX</code></td>
      <td style={{padding: "9px 12px"}}>Text only. Apple Silicon (MLX backend). Same serve recipe as gs128, no measured round yet. See §3.4.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong>MLX Q4_K_M (gs128)</strong></td>
      <td style={{padding: "9px 12px"}}><code>RadixArk/Muse-Glimmer-q4km-gs128-MLX</code></td>
      <td style={{padding: "9px 12px"}}>Text only. Apple Silicon (MLX backend). Carries the vendor GGUF's exact quantization codes in MLX format. The measured MLX artifact. See §3.4.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><strong>MLX Q4_K (dynamic)</strong></td>
      <td style={{padding: "9px 12px"}}><code>RadixArk/Muse-Glimmer-q4k-dynamic-MLX</code></td>
      <td style={{padding: "9px 12px"}}>Text only. Apple Silicon (MLX backend). Same serve recipe as gs128, no measured round yet. See §3.4.</td>
    </tr>
  </tbody>
</table>

**Resources:** [Muse-Glimmer-30B (BF16)](https://huggingface.co/meta-models/Muse-Glimmer-30B) · [Muse-Glimmer-30B-assistant (DFlash draft)](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant) · [Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) · [Muse-Glimmer-NVFP4](https://huggingface.co/RadixArk/Muse-Glimmer-NVFP4) · MLX · [q4](https://huggingface.co/RadixArk/Muse-Glimmer-q4-MLX) · [q4km-gs128](https://huggingface.co/RadixArk/Muse-Glimmer-q4km-gs128-MLX) · [q4k-dynamic](https://huggingface.co/RadixArk/Muse-Glimmer-q4k-dynamic-MLX).

## 2. Configuration Tips

**The GGUF format is text only.** SGLang has no `mmproj` path. You cannot use the vision GGUF files. Use the BF16 checkpoint for multimodal input.

**The NVFP4 checkpoint.** `RadixArk/Muse-Glimmer-NVFP4` is a ready-to-serve NVFP4 + MXFP8 checkpoint. No conversion needed — point `--model-path` straight at it.

**The DFlash draft.** `meta-models/Muse-Glimmer-30B-assistant` is the vendor's native draft export and serves directly. No conversion needed.

**DFlash with a GGUF target model** needs `--speculative-draft-load-format auto`. Without this flag, the draft model uses the `gguf` load format from the target model. The loader then rejects the draft directory.

**Apple Silicon uses an MLX checkpoint, not the GGUF files.** The MLX backend has no GGUF path. Serve one of the three `RadixArk/Muse-Glimmer-*-MLX` artifacts with `SGLANG_USE_MLX=1` (see the Apple Silicon cells in the command panel). All three take the same flags; `q4km-gs128` is the one with a measured round. Keep `--disable-radix-cache` — the windowed KV storage for the sliding-window layers requires it — and set `SGLANG_MLX_CACHE_LIMIT_GB=8` so the MLX buffer cache does not grow the footprint under concurrent load. Speculative decoding is not available on the MLX backend.

## 3. Advanced Usage

### 3.1 Reasoning

Muse Glimmer enables the `muse` reasoning parser by default. This parser separates the reasoning text from the final answer.

<Accordion title="Reasoning Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="meta-models/Muse-Glimmer-30B",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
)
msg = resp.choices[0].message
print("Reasoning:", getattr(msg, "reasoning_content", None))
print("Answer:", msg.content)
```

</Accordion>

### 3.2 Tool Calling

Muse Glimmer enables the `muse` tool-call parser by default. This parser sends structured tool calls in `message.tool_calls`.

### 3.3 Multimodal

The BF16 checkpoint supports image input. It defaults to text only. To switch, select **Modality** in the command panel above.

**Text only** adds `--language-model-only`. This flag turns off the vision tower. SGLang does not build or load the vision weights. This frees memory for the KV cache. SGLang rejects image requests in this mode.

Select **Image + text** to turn on image input.

NVFP4, GGUF, and the MLX artifacts are text only. The Modality option does not appear for GGUF or MLX; NVFP4 only offers **Text only**.

### 3.4 Apple Silicon (MLX)

The MLX backend serves three Muse Glimmer artifacts on Apple Silicon Macs (48 GB unified memory or more). All three are text only — the MLX backend has no vision path — and all three take the same flags, so pick one in the command panel:

- `RadixArk/Muse-Glimmer-q4-MLX` — no measured round yet.
- `RadixArk/Muse-Glimmer-q4km-gs128-MLX` — a lossless repack of the vendor's Q4_K_M (gs128) GGUF: every weight keeps the GGUF's exact quantization code, with the group scales re-expressed in MLX affine bf16 (≤2⁻⁸ relative rounding). The numbers below are for this artifact.
- `RadixArk/Muse-Glimmer-q4k-dynamic-MLX` — no measured round yet.

Choose along the speed-versus-accuracy axis: footprint and expected accuracy both grow `q4` → `q4km-gs128` → `q4k-dynamic`, and decode speed moves the other way. Decode on Apple Silicon is memory-bandwidth-bound, so a smaller artifact reads fewer weight bytes per token — more tokens per second, and more unified memory left over for the KV cache. Take `q4` for the fastest responses on the smallest machine, `q4k-dynamic` to stay closest to BF16, and `q4km-gs128` for the middle ground — it is also the only one of the three with a measured round, below.

This table shows accuracy for the gs128 checkpoint, with the vendor llama.cpp fork serving the source GGUF on the same machine as the reference. GSM8K: 200 questions, no-thinking chat template, temperature 0, max 2048 new tokens. CIMemories: 1 profile, full combo, single trial, DeepSeek-R1-0528 judge.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left",  padding: "10px 12px", fontWeight: 700}}>Benchmark</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>SGLang MLX</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>llama.cpp (same GGUF)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}>GSM8K (200q, no-thinking, greedy)</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>0.970</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>0.970</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}>CIMemories — violation rate (lower is better)</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>0.00%</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>8.27%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}>CIMemories — coverage (higher is better)</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>76.0%</td>
      <td style={{padding: "9px 12px", textAlign: "right"}}>68.4%</td>
    </tr>
  </tbody>
</table>

CIMemories is a single-trial benchmark with a nondeterministic judge; treat the SGLang-vs-llama.cpp gap on that row as run noise, not a runtime effect. GSM8K parity is exact.

Decode throughput for gs128 on an M5 Pro (64 GB), 1k-in/1k-out greedy: 15.3 tok/s at batch 1, rising to 52.6 tok/s aggregate at batch 8 — ahead of llama.cpp on the same GGUF codes at every batch size above 1.
