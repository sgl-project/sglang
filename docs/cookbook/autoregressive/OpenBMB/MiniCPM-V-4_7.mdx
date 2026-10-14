---
title: MiniCPM-V 4.7
description: "Deploy OpenBMB MiniCPM-V 4.7 with SGLang — 1B dense and 35B-A3B Mixture-of-Experts vision-language models on a hybrid Gated DeltaNet backbone, in BF16, FP8, or INT4, with DSpark speculative decoding on a single GB300 or H200."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

MiniCPM-V 4.7 support comes from [sgl-project/sglang#41993](https://github.com/sgl-project/sglang/pull/41993); both paths below use that PR's code. For all methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install). The two paths below match the **Python / Docker** toggle in the command panel.

<Tabs>

<Tab title="Python (from source)">

```bash Command
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin pull/41993/head:minicpm-v-4.7
git checkout minicpm-v-4.7
pip install --upgrade pip
pip install -e "python"
```

Then run the **Python** output of the command panel below in that environment.

</Tab>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:dev-minicpm-v-4.7
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick the hardware, variant, and quantization to generate the launch command. Every recipe runs on a single GPU and is verified on GB300 and H200; the H100 recipes (1B, FP8, INT4) are unverified. The optional **DSPARK** row adds speculative decoding; see §2 for where it helps.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/openbmb/minicpm-v-4.7.jsx";
import { benchmarks } from "/src/snippets/configs/openbmb/minicpm-v-4.7-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

The Playground is where you experiment with **SGLang features beyond the recipes above**. The Deploy panel emits this model's validated launch recipes; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

MiniCPM-V 4.7 is the latest model in OpenBMB's MiniCPM-V series, covering single-image, multi-image, and video understanding. It ships in two sizes: a 1B dense model and a 35B-A3B Mixture-of-Experts model, the series' first MoE.

Both sizes use a Qwen3.5-style hybrid language backbone in which every fourth layer is full attention and the rest are Gated DeltaNet linear attention: 24 layers for 1B, and 40 layers with 256 experts (8 active per token) for 35B-A3B. Native context length is 262,144 tokens. The vision encoder compresses visual tokens early inside the ViT, slices high-resolution images into up to 9 tiles, and supports mixed 4x/16x visual token compression (see §3.3). Thinking is on by default.

**License:** [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "42%"}} />
    <col style={{width: "22%"}} />
    <col style={{width: "36%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Checkpoint</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Precision</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Role on this page</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/openbmb/MiniCPM-V-4.7-1B">openbmb/MiniCPM-V-4.7-1B</a></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>BF16</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>1B dense base (1.55B parameters including the vision encoder).</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.05)"}}><a href="https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B">openbmb/MiniCPM-V-4.7-35B-A3B</a></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>BF16</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>35B-A3B MoE base (35.2B total parameters, about 3B active).</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-FP8">openbmb/MiniCPM-V-4.7-35B-A3B-FP8</a></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>FP8 W8A8 (compressed-tensors)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>35B-A3B FP8.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.05)"}}><a href="https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-int4">openbmb/MiniCPM-V-4.7-35B-A3B-int4</a></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>INT4 W4A16, group size 128 (compressed-tensors)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>35B-A3B INT4.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-DSpark">openbmb/MiniCPM-V-4.7-1B-DSpark</a></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>BF16 draft</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Draft model for the 1B <strong>DSPARK</strong> row; not served on its own.</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.05)"}}><a href="https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark">openbmb/MiniCPM-V-4.7-35B-A3B-DSpark</a></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>BF16 draft</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Draft model for the 35B-A3B <strong>DSPARK</strong> row; not served on its own.</td>
    </tr>
  </tbody>
</table>

OpenBMB also publishes AWQ, GPTQ, and GGUF exports and a 1B int4 checkpoint. They are not part of the matrix above.

**Recommended generation:** `temperature=1.0`, `top_p=0.95`, `top_k=20`, from the 35B-A3B checkpoints' `generation_config.json`. SGLang applies these defaults, so do not hardcode them in client code.

**Resources:** [Hugging Face (1B)](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) · [Hugging Face (35B-A3B)](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) · [OpenBMB documentation](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/minicpm_v4dot7_en.md) · [SGLang model support (#41993)](https://github.com/sgl-project/sglang/pull/41993).

## 2. Configuration Tips

- Every generated command carries `--mamba-radix-cache-strategy extra_buffer`, the radix-cache strategy for the Gated DeltaNet hybrid backbone. The FP8 and INT4 commands also carry `--tokenizer-path openbmb/MiniCPM-V-4.7-35B-A3B` (see the note below).
- Every recipe runs on one GPU. On H100 the 35B-A3B BF16 checkpoint (about 66 GB of weights) does not fit the default static memory fraction of an 80 GB card; use FP8 or INT4 there.
- Pick the quantization by platform. On H200, quantization is a speed win: at concurrency 16, INT4 and FP8 reach 15,808 and 15,280 tokens/s per GPU against 12,103 for BF16, and INT4 has the fastest 35B-A3B decode (TPOT 3.47 ms at concurrency 1). On GB300, BF16 and FP8 decode at about the same speed at concurrency 1 (TPOT 4.36 vs 4.44 ms), and FP8 pulls ahead at concurrency 16 (16,059 vs 14,094 tokens/s per GPU). INT4 is slower on GB300 at every concurrency (TPOT 8.56 ms at concurrency 1), so use it there only to save memory.
- SGLang detects FP8 and INT4 from the checkpoint's `quantization_config`; no `--quantization` flag is needed.
- The **DSPARK** row adds `--speculative-algorithm DSPARK` and `--speculative-draft-model-path` pointing at the variant's DSpark checkpoint. On GB300, turn it on only for 35B-A3B BF16 and FP8 at low concurrency; on H200, turn it on for 35B-A3B BF16 at both measured concurrencies and for FP8 and INT4 at low concurrency; leave it off for 1B. Matched sharegpt runs back this (TPOT ms, DSPARK off → on):

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "20%"}} />
    <col style={{width: "20%"}} />
    <col style={{width: "20%"}} />
    <col style={{width: "20%"}} />
    <col style={{width: "20%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Configuration</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>GB300, conc 1</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>GB300, conc 16</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>H200, conc 1</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>H200, conc 16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>1B BF16</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>1.75 → 2.02</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>4.00 → 5.73</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>1.56 → 1.65</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>4.43 → 6.27</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>35B-A3B BF16</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>2.99 → 2.80</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>11.17 → 13.69</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>3.56 → 3.02</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>13.51 → 11.92</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>35B-A3B FP8</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>3.08 → 2.76</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>10.53 → 12.60</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>3.42 → 2.61</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>11.11 → 12.40</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>35B-A3B INT4</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>7.19 → 11.05</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>47.92 → 54.80</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>3.34 → 2.38</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>9.33 → 11.20</td>
    </tr>
  </tbody>
</table>

- Benchmark speculative decoding on real text. Random-token prompts collapse draft acceptance, so they say nothing about DSPARK. For a real-workload reference, GSM8K accept length is about 2.6 on 1B, 4.1 on 35B-A3B BF16, 4.0 on FP8, and 3.8 on INT4.
- Every command carries `--reasoning-parser auto --tool-call-parser auto`. SGLang resolves both from the chat template, to `qwen3` for reasoning and `qwen3_coder` for tool calls, so the thinking trace lands in `message.reasoning_content` and tool calls in `message.tool_calls` (see §3.1 and §3.2). To turn either off, toggle it in the **Parsers** card of the [Playground](#playground).
- Thinking is on by default. To get a direct answer for one request, pass `"chat_template_kwargs": {"enable_thinking": false}` (in the OpenAI Python client, `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`); `reasoning_content` then comes back empty.
- Image and video requests need no extra server flags.

<Note>
  The public FP8 and INT4 repos' `tokenizer_config.json` declares `auto_map.AutoProcessor = processing_minicpmv4_7.MiniCPMV4_7Processor`, but the repos do not ship that file, so SGLang cannot load their processor. The FP8 and INT4 commands therefore load the tokenizer and processor from the base repo with `--tokenizer-path openbmb/MiniCPM-V-4.7-35B-A3B`, which has the same `tokenizer.json` and chat template. Keep the flag until OpenBMB updates the FP8 and INT4 repos to ship the file or drop the `auto_map` entry.
</Note>

## 3. Advanced Usage

### 3.1 Reasoning

MiniCPM-V 4.7 thinks by default. The reasoning parser in every generated command returns the thinking trace in `message.reasoning_content` and the final answer in `message.content`.

<Accordion title="Reasoning (Python)">

```python Example
import requests

resp = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "openbmb/MiniCPM-V-4.7-35B-A3B",
        "messages": [{"role": "user", "content": "What is 23*17?"}],
    },
).json()
choice = resp["choices"][0]
print("finish_reason:", choice["finish_reason"])
print("--- reasoning_content ---")
print(choice["message"]["reasoning_content"])
print("--- content ---")
print(choice["message"]["content"])
```

</Accordion>

<Accordion title="Example Output">

```text Output
finish_reason: stop
--- reasoning_content ---
The user is asking for the product of 23 and 17. I need to compute 23 * 17.

Let me calculate:
23 * 17
I can break it down:
23 * 10 = 230
23 * 7 = 161
Now add: 230 + 161 = 391

Wait, let me double-check:
23 * 17
23 * (10 + 7) = 23*10 + 23*7
23*10 = 230
23*7: 20*7=140, 3*7=21, so 140+21=161
Now 230 + 161 = 391

So the product is 391.

--- content ---


To find 23 × 17, I'll calculate the multiplication:

23 × 17 = 391

Answer: 391
```

</Accordion>

<Note>
  On GSM8K-style prompts that ask for the answer in `\boxed{}`, the model can write the final answer inside the thinking block and stop without emitting `</think>`, so the reasoning parser leaves `content` empty. Clients that read only `content`, such as accuracy-eval harnesses, should either serve without `--reasoning-parser` or also read `reasoning_content`.
</Note>

### 3.2 Tool calling

The tool-call parser in every generated command returns calls in `message.tool_calls`, with the arguments as a JSON string. Because thinking is on, print `reasoning_content` and `content` alongside the calls.

<Accordion title="Tool calling (Python)">

```python Example
import json

import requests

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]

resp = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "openbmb/MiniCPM-V-4.7-35B-A3B",
        "messages": [{"role": "user", "content": "What is the weather in Beijing today?"}],
        "tools": tools,
    },
).json()
choice = resp["choices"][0]
msg = choice["message"]
print("finish_reason:", choice["finish_reason"])
print("--- reasoning_content ---")
print(msg["reasoning_content"])
print("--- content ---")
print(repr(msg["content"]))
print("--- tool_calls ---")
print(json.dumps(msg["tool_calls"], indent=1))
```

</Accordion>

<Accordion title="Example Output">

```text Output
finish_reason: tool_calls
--- reasoning_content ---
The user is asking about the weather in Beijing. I need to call the get_weather function with the city parameter set to Beijing.

--- content ---
'\n\n'
--- tool_calls ---
[
 {
  "id": "call_592d3b19eca4432eaf1a89f1",
  "index": 0,
  "type": "function",
  "function": {
   "name": "get_weather",
   "arguments": "{\"city\": \"Beijing\"}"
  }
 }
]
```

</Accordion>

To get the final answer, append the assistant message and a `tool` role message with the tool's result, then call the API again with the same `tools` list.

### 3.3 Per-request image processing

OpenAI-compatible `/v1/chat/completions` and `/v1/completions` requests accept `mm_processor_kwargs`, which overrides the image processor for that request only:

- `downsample_mode`: `"16x"` (default) or `"4x"`. `"4x"` keeps more visual tokens per tile.
- `max_slice_nums`: the maximum number of high-resolution slices. `1` sends the image without slicing.

<Accordion title="Image token budget (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

IMAGE_URL = "https://www.ilankelman.org/stopsigns/australia.jpg"

for overrides in ({}, {"downsample_mode": "4x"}, {"max_slice_nums": 1}):
    resp = client.chat.completions.create(
        model="openbmb/MiniCPM-V-4.7-35B-A3B",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }],
        extra_body={"mm_processor_kwargs": overrides} if overrides else None,
    )
    print(overrides or "default", "prompt_tokens =", resp.usage.prompt_tokens)
```

</Accordion>

Prompt token counts from the validation probes. The counts depend on the image sizes and the text prompt, so the example above, which uses a different image, returns different values. The first two rows send the same image and text prompt and differ only in `downsample_mode`.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "46%"}} />
    <col style={{width: "34%"}} />
    <col style={{width: "20%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left",  padding: "10px 12px", fontWeight: 700}}>Input</th>
      <th style={{textAlign: "left",  padding: "10px 12px", fontWeight: 700}}><code>mm_processor_kwargs</code></th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Prompt tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>One 1568×448 image</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>none (<code>"16x"</code>)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>355</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>One 1568×448 image</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>{`{"downsample_mode": "4x"}`}</code></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>1291</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Two images (1568×448 and 768×384) in one request</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>none (<code>"16x"</code>)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>556</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>One 768×384 image</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}><code>{`{"max_slice_nums": 1}`}</code></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>93</td>
    </tr>
  </tbody>
</table>

Overrides do not leak between requests: alternating `"16x"` and `"4x"` on the same image returns the matching token count and a correct answer each time.

For more API examples, see the [SGLang Basic Usage Guide](/docs/basic_usage/send_request).
