---
title: Dots3-Note
description: "Deploy RedNote dots3.note with SGLang — a native multimodal omni model (MoE ViT + Whisper-derived audio encoder + native video flattening) on the dots3 hybrid MLA/SWA language model, with DSA and full-sharing MTP speculative decoding."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

dots3.note support is in [SGLang PR #33829](https://github.com/sgl-project/sglang/pull/33829). Until that PR is included in a tagged SGLang release, install from a build that contains the PR.

<Tabs>

<Tab title="Python (pip / uv)">

```bash Command
pip install -U uv
uv venv --python 3.12 && source .venv/bin/activate

git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin pull/33829/head && git checkout FETCH_HEAD
uv pip install --prerelease=allow -e python
```

Then run the **Python** output of the command panel below in that environment.

</Tab>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:dev-dots3-note
```

This image packages SGLang with the dots3.note support from PR #33829 and is the recommended way to deploy until the PR lands in a tagged SGLang release — it saves you from building the branch yourself.

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick the hardware and the checkpoint precision. The recipe runs on a single 8-GPU Hopper node with DP8 attention × TP8 × EP8 and DeepEP as the MoE all-to-all transport. Blackwell is not supported yet.

**Precision** — selects the MoE path, not just the weights. The BF16 cells pin `--moe-runner-backend deep_gemm` with BF16 DeepEP dispatch output (JIT DeepGEMM is enabled via `SGLANG_ENABLE_JIT_DEEPGEMM=1`). The FP8 cells leave both at `auto` and let SGLang resolve the runner from the checkpoint's quantization config.

**Spec Decode** — NEXTN is on in every cell: 3 draft steps, 4 draft tokens per step, and the draft model path pointing at the target checkpoint itself. dots3's MTP layer is full-sharing — it carries the dots3 sliding-window attention geometry and reuses the target LM head — so no separate draft checkpoint is needed. Target verification and draft extension run on the paged, absorbed SWA-MLA FA3 path.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/rednote/dots3-note.jsx";

<Deployment config={config} />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations signed off on this page; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

dots3.note is RedNote's native multimodal omni model, built on the dots3 language model. It accepts text, image, audio, and native video input.

- **Native multimodality** — a custom MoE vision transformer and a Whisper-derived audio encoder run in-process with the language model, loaded from the same checkpoint directory. Image and audio placeholders are expanded by a model-specific processor.
- **Native video pipeline** — the server jointly samples and interleaves frames, timestamps, and audio segments under a token budget, reproducing the training-time flattening algorithm. A generic uniform-frame video processor would silently change the modality ordering and token allocation (inference/training mismatch), so the pipeline is vendored into the serving path.
- **Hybrid attention** — dots3 combines MLA with full-attention and sliding-window layers of different geometry, attention gates, and optional DSA indexing on full-attention layers.
- **MTP speculative decoding** — a full-sharing MTP/NextN architecture exposes one recursively shared, SWA-shaped MTP layer and shares the target LM head.

**Available checkpoints:**

- **BF16**: [dots-studio/dots3-note-prev](https://huggingface.co/dots-studio/dots3-note-prev)
- **FP8**: [dots-studio/dots3-note-prev-fp8](https://huggingface.co/dots-studio/dots3-note-prev-fp8)

**Resources:** [Hugging Face (BF16)](https://huggingface.co/dots-studio/dots3-note-prev) · [Hugging Face (FP8)](https://huggingface.co/dots-studio/dots3-note-prev-fp8) · [SGLang PR #33829](https://github.com/sgl-project/sglang/pull/33829)

## 2. Configuration Tips

**Hybrid KV pool.** dots3 mixes full-attention and sliding-window layers, and its MTP draft layer is an ordinary SWA layer — not a full-attention one. SGLang sizes the pool accordingly, with `--swa-full-tokens-ratio 0.03` setting the ratio of SWA-layer KV tokens to full-layer KV tokens (`swa_tokens ≈ full_tokens × ratio`). Lower it when long full-attention contexts dominate and the full pool fills first; raise it when the SWA pool is the bottleneck.

**MoE runner.** Leave the runner at the cell default: `deep_gemm` for BF16 checkpoints, `auto` for quantized ones. DeepEP is the all-to-all transport in every cell (`--moe-a2a-backend deepep`, dispatch tokens per rank tuned via `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128`).

**Attention backend.** FA3 across the board: prefill, decode, and draft (`--prefill-attention-backend fa3 --decode-attention-backend fa3 --speculative-draft-attention-backend fa3`) with `--page-size 64`. MTP target verification uses FA3's absorbed SWA-MLA fallback, which consumes the same paged latent KV view as decode.

**DSA.** DSA indexing on full-attention layers is on by default. To disable it, add `--json-model-override-args '{"index_topk":null}'`.

**CUDA graphs.** The cells enable decode-side CUDA graphs only (`--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled`, max batch size 32) and are sized for GPUs with at least 120 GiB of memory. On smaller GPUs, switch to `--cuda-graph-backend-decode disabled` (and expect `--deepep-mode normal` to be the better fit).

**Context length.** `--context-length 524288` is the model's window. Like other SGLang models, it bounds the longest accepted request; it does not size the KV pool.

**Language-only mode.** Add `--language-only` to skip constructing the vision and audio towers entirely — the freed memory goes to the language model. This is also the language role of an encoder/LLM-disaggregated (EPD) deployment; see [EPD](#epd-disaggregation) below.

## 3. Advanced Usage

### 3.1 Native video input

dots3.note accepts a native `video_url`. The server decodes the remote video in memory and applies the training-consistent flattening pipeline — interleaving timestamps, frames, and audio under a token budget, with a deterministic seed derived from the video and the question.

<Accordion title="Video Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="dots3.note",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/sample.mp4"},
                },
                {"type": "text", "text": "Summarize what happens in this video."},
            ],
        }
    ],
    extra_body={
        "video_config": {
            "seq": 131072,
            "audio_cap": 0.5,
            "audio_sr": 16000,
            "k_mode": "eval_ek",
        },
    },
)

print(response.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Pending update...
```

</Accordion>

Per-request video preprocessing controls are grouped under `video_config` in
`extra_body`:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "22%"}} />
    <col style={{width: "18%"}} />
    <col style={{width: "60%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.02)"}}>Field</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.05)"}}>Default</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.02)"}}>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><code>seq</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>131072</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Total sequence budget used by the video flattener.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><code>audio_cap</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>1.0</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Maximum fraction of the input budget assigned to audio; <code>0</code> disables audio processing.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><code>audio_sr</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>16000</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Audio sample rate.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><code>k_mode</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>eval_ek</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Deterministic evaluation/sampling mode of the flattener.</td>
    </tr>
  </tbody>
</table>

These controls are request-scoped so that evaluation jobs with different context budgets can share one server. For example: `extra_body={"video_config": {"seq": 131072, "audio_cap": 0.5}}`. The flattener reserves room for `max_new_tokens` inside the budget and falls back to visual-only processing if audio would exceed the configured token budget.

A request may carry several videos, and videos can be mixed with image and audio parts. Each video is flattened independently under the same per-request budget, and the flattened frames and audio segments are spliced back at the position of their `video_url` part, so the modality ordering of the prompt is preserved.

### 3.2 Image and audio input

Outside the native-video path, images and audio clips use the standard OpenAI multimodal message format. `--enable-multimodal` is in every cell; the vision and audio towers run in-process, so no extra server is needed.

<Accordion title="Image Example (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="dots3.note",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/sample.jpg"},
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
)

print(response.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Pending update...
```

</Accordion>

### 3.3 Tool Calling

Toggle **Tool Call Parser** (`--tool-call-parser dots`) and **Reasoning Parser** (`--reasoning-parser dots`) in the **Parsers** card of the [Playground above](#playground). Structured tool calls then surface via `message.tool_calls`.

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
    model="dots3.note",
    messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    tools=tools,
)
print(resp.choices[0].message.tool_calls)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Pending update...
```

</Accordion>

<a id="epd-disaggregation" />

### 3.4 Encoder/LLM Disaggregation (EPD)

`Dots3NoteForCausalLM` supports both roles of an encoder/LLM-disaggregated deployment:

- **Encoder role** — serve with `--encoder-only`; the instance runs only the vision and audio towers.
- **Language role** — serve with `--language-only`; the instance skips tower construction, leaving the memory to the language model.

See the [EPD guide](../../../docs/advanced_features/epd_disaggregation) for how to wire the roles together.
