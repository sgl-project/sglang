---
title: DiffusionGemma
metatags:
    description: "Deploy DiffusionGemma with SGLang - Google's uniform-state renoising block-diffusion language model (26B-A4B MoE) served with the Gemma4Renoise sampler."
tag: NEW
---

## 1. Model Introduction

DiffusionGemma is a uniform-state (renoising) block-diffusion language model from Google. An encoder builds causal context, and a decoder denoises a fixed-length bidirectional canvas of `canvas_length` tokens. The `Gemma4Renoise` sampler runs `max_denoising_steps` reverse steps over the canvas, feeding the previous step's logits back as self-conditioning and emitting the greedy argmax of the processed logits.

**Key Features:**

- **Uniform-State Renoising**: The canvas starts from random tokens and is refined each step by accepting confident positions and re-noising the rest, with no mask token.
- **Encoder / Decoder Canvas**: The encoder produces causal context KV, the decoder attends bidirectionally over the canvas.
- **Self-Conditioning**: Each step conditions on the previous step's logits.
- **EntropyBound Acceptance**: Each step accepts the lowest-entropy canvas positions within an entropy budget and re-noises the rest.
- **StableAndConfident Stopping**: A canvas stops early once it is stable and confident.
- **MoE Architecture**: The 26B-A4B model uses a Mixture-of-Experts architecture for efficient inference.
- **Multimodal Input**: Accepts text and image inputs (via a ~550M vision encoder) and generates text output.

**Available Models:**

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "40.0%"}} />
    <col style={{width: "30.0%"}} />
    <col style={{width: "30.0%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Model</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Architecture</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>[google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>MoE, uniform-state diffusion (text + image)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>25.2B total / 3.8B active</td>
    </tr>
  </tbody>
</table>

**Architecture Specifications:**

| Spec | Value |
| --- | --- |
| Total Parameters | 25.2B |
| Active Parameters | 3.8B |
| Layers | 30 |
| Sliding Window | 1024 tokens |
| Context Length | Up to 256K tokens |
| Canvas Length | 256 |
| Vocabulary Size | 262K |
| Experts | 8 active / 128 total + 1 shared |
| Supported Modalities | Text, Image |
| Vision Encoder | ~550M parameters |

**License:**

Refer to the model card for license details.

## 2. SGLang Installation

Please refer to the [official SGLang installation guide](../../../docs/get-started/install) for installation instructions.

The checkpoint ships its own modeling code, so `--trust-remote-code` is required when serving.

## 3. Model Deployment

### 3.1 Basic Configuration

The required runtime settings are applied automatically for `Gemma4Renoise` (the Triton attention backend, eager mode, and unchunked prefill, needed because the full-attention head_dim is 512 and the canvas uses bidirectional attention), so a default launch works:

```bash Command
sglang serve \
  --model-path google/diffusiongemma-26B-A4B-it \
  --dllm-algorithm Gemma4Renoise \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000
```

### 3.2 Configuration Tips

**dLLM-Specific Parameters:**

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Parameter</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Description</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Recommended Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--dllm-algorithm`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Diffusion decoding algorithm</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>`Gemma4Renoise`</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--trust-remote-code`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Required to load the checkpoint's modeling code</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Always enabled</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--dllm-algorithm-config`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Optional YAML overriding the renoise schedule</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Checkpoint defaults</td>
    </tr>
  </tbody>
</table>

The attention backend, eager mode, and unchunked prefill are selected automatically for `Gemma4Renoise`, so they do not need to be passed on the command line.

Sampling is governed by the renoise schedule. Request-level `logprobs`, penalties, `logit_bias`, and grammar / structured output (`json_schema` / `regex` / `ebnf` / `structural_tag`) are not applied and are rejected with a 400. Core sampling controls (`temperature`, `top_k`, `top_p`) are accepted but have no effect. Streaming is block-level: one fully-denoised canvas per chunk.

**Gemma4Renoise Config** (defaults follow the checkpoint's `generation_config.json`):

```yaml Config
# Number of reverse denoising steps per canvas.
max_denoising_steps: 48
# Optional. Makes the renoise sampling reproducible (also shared across TP ranks).
seed: 1234
sampler_config:
  # Entropy budget. Accept the lowest-entropy canvas positions within this bound each step (the rest are re-noised).
  entropy_bound: 0.1
# Linear temperature schedule applied over the denoising steps.
temperature_schedule:
  t_min: 0.4
  t_max: 0.8
# Stop early once the canvas is stable and confident.
stopping_config:
  confidence_threshold: 0.005
  stability_threshold: 1
```

## 4. Model Invocation

### 4.1 Deployment

Start the server with the command from [Section 3.1](#3-1-basic-configuration).

### 4.2 Basic Usage

```python Example
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="google/diffusiongemma-26B-A4B-it",
    messages=[
        {"role": "user", "content": "What are the key differences between TCP and UDP?"}
    ],
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### 4.3 Streaming

Streaming emits one fully-denoised canvas per chunk.

```python Example
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="google/diffusiongemma-26B-A4B-it",
    messages=[
        {"role": "user", "content": "Write a Python function to compute the Fibonacci sequence."}
    ],
    max_tokens=2048,
    stream=True
)

for chunk in response:
    if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

print()
```

## 5. Benchmark

### 5.1 Speed Benchmark

Not benchmarked for speed.

### 5.2 Accuracy Benchmark

Full test splits, every item scored (no failed-request exclusions). Text MCQ benchmarks use greedy generate-and-parse, MATH uses boxed-answer extraction plus sympy equivalence. MMLU, ARC-Challenge, and MATH-500 are the mean of two independent server launches.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "50.0%"}} />
    <col style={{width: "50.0%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Benchmark</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>GSM8K</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>95.4%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>ARC-Challenge</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>91.6%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>HumanEval</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>92.7% pass@1</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>MMLU</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>76.2%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>MMLU-Pro</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>73.7%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>GSM-Symbolic</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>92.2%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>MATH-500</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>72.1%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>AIME-2026</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>10.0%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>HMMT-Feb-2025</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>10.0%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>GPQA-main</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>59.2%</td>
    </tr>
  </tbody>
</table>

Multimodal, full standard split per task (MMMU / MMMU-Pro / MMStar / AI2D as multiple-choice, MathVista testmini, DocVQA by ANLS, ChartQA by relaxed accuracy):

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "50.0%"}} />
    <col style={{width: "50.0%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Multimodal benchmark</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>MMMU (val, MC)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>64.9%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>MMMU-Pro (standard 10-opt, MC)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>57.3%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>MathVista (testmini)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>68.4%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>DocVQA (val)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>85.9%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>ChartQA (test)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>61.7%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>AI2D (test)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>78.7%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>MMStar (val)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>65.9%</td>
    </tr>
  </tbody>
</table>
