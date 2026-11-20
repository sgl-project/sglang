---
title: Hy4
description: "Deploy Tencent Hy4 with SGLang — a 770B-parameter MoE vision-language model with DeepSeek Sparse Attention and DFly2 speculative decoding, in FP8 on H200, B200, B300 and GB300."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all methods and hardware platforms, see the [official SGLang installation guide](/docs/get-started/install).

<Tabs>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:hy4
```

For how to launch the image, see [Install → Method 3: Using Docker](/docs/get-started/install#method-3-using-docker) and use the `hy4` tag; the command generator below supplies the `sglang serve` command. Run its **Python** output inside that container, or its **Docker** output directly on the host. The **Docker** output passes `HF_TOKEN`; set it in **⚙ Env** before copying.

</Tab>

</Tabs>

</Accordion>

Pick your hardware and strategy to generate the launch command. Both strategies serve the FP8 checkpoint at TP8 on 8× H200/B200 or TP4 on 4× B300/GB300 and use automatic memory sizing:

- **Low-Latency** — DFly2 speculative decoding with the draft layers bundled in the checkpoint. Pick it for interactive traffic such as single-user chat and agents that decode greedily; see [DFly2 requirements](#dfly2-requirements) for the request restrictions.
- **High-Throughput** — no speculative decoding. Pick it for multi-user and batch traffic, for sampled decoding and grammar-constrained tool calls, and as the base for DP-Attention, context parallelism and the FP8 KV cache.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/tencent/hy4.jsx";
import { benchmarks } from "/src/snippets/configs/tencent/hy4-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Checkpoints

<table style={{width: "100%", borderCollapse: "collapse"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #0052d9"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.02)"}}>Checkpoint</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.05)"}}>Weights</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, backgroundColor: "rgba(255,255,255,0.02)"}}>Recipes on this page</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><code>tencent/Hy4-FP8</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>~786 GB, block FP8</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>8× H200/B200 at TP8, 4× B300/GB300 at TP4</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><code>tencent/Hy4</code></td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>~1.55 TB, BF16</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>None: the weights exceed TP8 on H200/B200 and TP4 on B300/GB300</td>
    </tr>
  </tbody>
</table>

`tencent/Hy4-FP8` stores the routed experts, the attention output and gate projections, and most of the DFly2 layers' linear projections in block FP8: e4m3 weights in 128×128 blocks with dynamically quantized group-128 activations. The remaining attention projections, the DSA indexer, the first layer's attention and dense MLP, the shared experts, `lm_head` and the vision tower stay in BF16. SGLang reads the quantization from the checkpoint, so the recipes pass no `--quantization` flag.

Both checkpoints declare a 1,048,576-token context. No recipe pins `--context-length`, so the server uses the declared length; the KV pool, not the declared length, bounds how many tokens fit at once.

## Model capabilities

Hy4 pairs a 770B-parameter language backbone (MLA + DSA sparse attention + 256-expert MoE + iHC) with a 40-layer vision transformer and a Conv2d spatial-merge adapter, for image and video understanding with dynamic resolution (32-pixel alignment, 512²–2048² budget) and 3-channel mRoPE position encoding. The 770B count excludes the ViT, the adapter and the DFly2 layers.

- **Images**: single and multiple images per request, in any position relative to text. Both the OpenAI chat API (`image_url` parts) and raw prompts with `<image>` placeholders work.
- **Video**: sampled-frame video input (fps=2) with per-frame timestamp tokens. Long or high-resolution clips can exhaust the KV pool, so leave headroom for long videos.
- **Tool calling + reasoning**: `--reasoning-parser auto --tool-call-parser auto` resolve to the `hunyuan` parsers. `tool_choice: "required"`, a named function, and `"auto"` with `strict` tools are enforced by a native structural grammar: the call envelope, tool names and argument keys are constrained, while argument values are free text and are not validated against the schema. A named function or `parallel_tool_calls: false` allows exactly one call. DFly2 rejects these grammar-constrained requests; serve them from the High-Throughput recipe.
- **DFly2 speculative decoding**: the checkpoint bundles five DFly2 draft layers. One draft forward runs the bonus token, mixed with the target's final hidden state, plus six mask tokens, and one target pass verifies the seven-token block. Each draft layer keeps its KV in the pool of a target layer and reuses a target indexer's sparse-attention top-k, so the draft builds no indexer and no KV pool of its own. The seven-token block comes from the checkpoint's `dfly_block_size`, so the recipes pass no block-size flag. With the server defaults, the decode CUDA graph resolves to `full` and captures the draft forward together with the target verify, the target prefill replays through a `breakable` CUDA graph, and overlap scheduling stays on.

## Playground

The Playground adds knobs on top of the selected Deploy cell: DP-Attention and prefill context parallelism (CP2) on the High-Throughput recipe, the hierarchical KV cache, and the speculative preset. Everything else is inherited from the base recipe.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## DFly2 requirements

The server validates DFly2 before loading weights and refuses to start with any of:

- a decode CUDA graph backend other than `full` or `disabled`, a prefill CUDA graph backend other than `breakable` or `disabled`, two-batch overlap or mixed chunked prefill;
- DP-Attention, pipeline parallelism, prefill context parallelism (`--attn-cp-size` above 1), decode context parallelism (`--dcp-size` above 1) or PD disaggregation;
- a model dtype other than BF16, or a KV cache dtype other than BF16;
- LoRA;
- a separate draft checkpoint (`--speculative-draft-model-path`), a draft attention backend, or a draft quantization or KV cache dtype that differs from the target's;
- `--speculative-num-steps` or `--speculative-eagle-topk` other than 1;
- a TP size that does not divide the draft's attention heads and MLP widths (TP 4 and TP 8 do).

DFly2 verifies drafts greedily. A request that sets none of `temperature`, `top_k`, `top_p` and `min_p` resolves to `temperature: 0`, `top_k: 1`. Setting any one of those fields is an explicit sampling choice, and the fields it leaves unset keep the checkpoint's `generation_config.json` values, including `temperature: 0.9`: a request with only `top_p: 1` samples at temperature 0.9 and is rejected. To combine sampling fields with DFly2, also send `temperature: 0` or `top_k: 1`; otherwise the request fails with HTTP 400 and the message `DFLY speculative decoding verifies greedily; send temperature=0 or top_k=1.` DFly2 rejects non-greedy sampling, grammar-constrained decoding, frequency, presence or repetition penalties, a minimum token count, logit bias, custom logit processors, returned logprobs and returned hidden states.

DFly2 runs on CUDA only and needs the default `flashmla_sparse` DSA decode backend. Hierarchical KV cache is supported with DFly2.

## Optional features

These are opt-in additions to the Deploy recipes.

- **Hierarchical KV cache (HiCache)**: enable it on either recipe in the Playground's **HiCache** card, which adds `--enable-hierarchical-cache --hicache-ratio 2 --hicache-size 0 --hicache-write-policy write_through`. Cached prefixes are written through to a host pool twice the size of each rank's device KV pool, and a prefix evicted from the GPU reloads from host memory on its next hit. The host pool holds both the MLA latent KV and the DSA indexer keys, so budget host RAM for it on every rank. HiCache is refused with decode context parallelism.
- **DP-Attention** (High-Throughput recipe): add `--attn-dp-size` with a degree that divides TP (2, 4 or 8 at TP8; 2 or 4 at TP4). The server refuses DP-Attention with DFly2 and with prefill context parallelism.
- **Prefill context parallelism** (High-Throughput recipe): add `--enable-prefill-cp --cp-strategy interleave --attn-cp-size 2` for long prompts with text, images and video. The server refuses prefill CP with speculative decoding (including DFly2), DP-Attention, pipeline parallelism, PD disaggregation and `--cp-strategy zigzag`.
- **Decode context parallelism** (High-Throughput recipe, B200/B300/GB300): on SM100, SM103 and SM107 GPUs, `--dcp-size 2` stripes the MLA latent KV across each rank pair by position while the DSA indexer keys stay replicated, raising KV capacity at the cost of extra decode communication. Both DSA backends switch to `trtllm`, and the prefill CUDA graph is disabled. On H200 (SM90) and SM12x GPUs the server refuses `--dcp-size` at startup: the DCP merge needs the softmax LSE that only the TRT-LLM-gen sparse MLA kernel returns with learnable attention sinks, and that kernel runs only on SM100, SM103 and SM107. DCP is refused with DFly2, pipeline parallelism, DP-Attention, prefill CP, HiCache, the FP8 KV cache and PD disaggregation. See [decode context parallelism](/docs/advanced_features/dcp).
- **FP8 KV cache** (High-Throughput recipe, B200/B300/GB300): the KV cache is BF16 by default on every GPU. On SM100, SM103 and SM107, `--kv-cache-dtype fp8_e4m3` stores it in FP8 and resolves both DSA backends to `trtllm`. The server refuses the FP8 KV cache on SM90 and SM12x, with decode or prefill context parallelism, and with DFly2.

## Usage examples

The examples use the OpenAI Python client against either recipe on port 30000. They set no sampling fields, so the Low-Latency recipe decodes greedily and the High-Throughput recipe samples with the checkpoint's `generation_config.json` defaults. The streaming and video outputs come from the H200 High-Throughput recipe and the other outputs from the H200 Low-Latency recipe; on the High-Throughput recipe the wording varies from run to run.

### Chat

Thinking is on by default. To answer directly, pass `reasoning_effort: "no_think"` through `chat_template_kwargs`:

<Accordion title="Chat without thinking (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="tencent/Hy4-FP8",
    messages=[{"role": "user", "content": "Hi"}],
    max_tokens=32,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
)

print(response.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Hello! How can I help you today? 😊
```

</Accordion>

### Reasoning

The Deploy recipes pass `--reasoning-parser auto`, which resolves to the `hunyuan` parser. It returns the thinking in `message.reasoning_content` and the answer in `message.content`:

<Accordion title="Reasoning (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="tencent/Hy4-FP8",
    messages=[{
        "role": "user",
        "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Put the final answer after '####'.",
    }],
    max_tokens=2048,
)

message = response.choices[0].message
print("=============== Thinking =================")
print(message.reasoning_content)
print("=============== Content =================")
print(message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
=============== Thinking =================
We need answer simple word problem. Need reason. Need final with answer after '####'. Need be careful: "sold clips to 48 of her friends in April" Ambiguity: sold clips to 48 friends, not necessarily 48 clips? Usually "sold clips to 48 of her friends" means sold 48 clips? Need parse. Then "sold half as many clips in May" means half as many clips as in April. Need total. If April = 48 clips, May = 24, total = 72. Need final answer after '####'. Could include concise explanation. Need ensure final answer after '####'. Maybe "April: 48 clips. May: half of 48 = 24. Total = 48 + 24 = 72. #### 72". Need maybe "#### 72" final answer after marker. Need not overcomplicate. Need consider if "to 48 of her friends" could mean 48 friends, each bought one clip? In elementary word problems, "sold clips to 48 of her friends" likely means 48 clips. Need answer 72. Need final.
=============== Content =================
April: 48 clips
May: half of 48 = 24 clips
Total: 48 + 24 = 72 clips

#### 72
```

</Accordion>

With `stream=True`, the thinking arrives in `delta.reasoning_content` and the answer in `delta.content`:

<Accordion title="Streaming with thinking (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="tencent/Hy4-FP8",
    messages=[{
        "role": "user",
        "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Put the final answer after '####'.",
    }],
    max_tokens=2048,
    stream=True,
)

thinking_started = False
answer_started = False
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):
        if not thinking_started:
            print("=============== Thinking =================", flush=True)
            thinking_started = True
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        if not answer_started:
            print("\n=============== Content =================", flush=True)
            answer_started = True
        print(delta.content, end="", flush=True)
print()
```

</Accordion>

<Accordion title="Example Output">

```text Output
=============== Thinking =================
We need answer word problem. Need think carefully.

Problem: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Put the final answer after '####'."

Need parse: "sold clips to 48 of her friends in April" Ambiguous: sold clips to 48 friends, not necessarily 48 clips. But likely intended "sold 48 clips" or "sold clips to 48 friends" maybe each friend one clip? Need answer from typical arithmetic: April 48, May half as many = 24, total 72. Need maybe note "assuming one clip per friend"? Let's analyze.

The phrase "sold clips to 48 of her friends in April" means quantity of friends, not clips. However such grade-school word problems often have imprecise wording: "sold clips to 48 of her friends" implies 48 clips if each friend bought one clip. Then "sold half as many clips in May" means half of 48 clips = 24 clips. Altogether clips = 48 + 24 = 72. If strict, cannot determine from friend count; but user likely expects arithmetic. Need decide.

Question: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" It says "sold clips to 48 of her friends in April" not "sold 48 clips". If asked "How many clips" and no detail per friend, there is insufficient info. But most benchmark datasets assume "to 48 of her friends" equals 48 clips? Let's consider if there is a trick? Need maybe answer "72" because "half as many clips" references 48 clips. The phrase "sold clips to 48 of her friends" could be interpreted as "sold [hair] clips [one] to 48 friends" by common core word problem. They may use "clips to 48 of her friends" as count of clips sold, e.g., "sold clips to 48 of her friends" means 48 clips if each friend got one. These tasks typically expect final numeric. Need provide final answer after '####'. We can include calculation maybe concise, final answer after marker. Need ensure final answer after '####' exactly.

Need craft final: "In May she sold half of 48, which is 24. Altogether: 48 + 24 = 72. #### 72" Need "final answer after '####'" maybe they want '#### 72' at end. Could be just "#### 72" to be safe. Let's prepare final with minimal calculation and marker. Need no mention ambiguity unless maybe avoid, as final should answer. Could include "Assuming one clip per friend"?? That might confuse. But if strict problem maybe maybe answer insufficient. Need decide.

Let's inspect known style from arithmetic word problems. The wording "sold clips to 48 of her friends in April" maybe from a dataset where answer expected is 72. They ask "half as many clips in May." In datasets, "sold clips to 48 of her friends" often means "sold 48 clips" (they conflated). There is maybe a "misleading" phrase? Let's reason.

If "Natalia sold clips to 48 of her friends in April" - "clips" plural, not "a clip". She could have sold multiple clips to each of 48 friends. Then "half as many clips in May" — half as many as what? It references "as many clips" but no number of clips in April is given; only number of friends. A mathematically rigorous answer: cannot determine. But elementary word problem expects assume "one clip per friend" or probably meant "sold 48 clips". The instruction "Put final answer after '####'" suggests benchmark expects numeric final answer. If answer "cannot be determined" maybe marked wrong. Need optimize.

Need maybe include "assuming each friend bought one clip" and answer 72? But "final answer after ####" maybe final numeric. Could say "Assuming one clip per friend, May = 48/2 = 24; total = 48+24=72. #### 72" That hedges ambiguity while satisfying expected. But if want strict, final answer after #### should be "Cannot be determined" not 72. Which is more likely? Let's consider dataset context. The user asks a simple word problem; no mention of trick. It's likely from "GSM8K" or similar? Actually GSM8K problem: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" This is a known GSM8K problem! I recall GSM8K first example: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" The known answer is 72. Yes, in GSM8K, solution: "In May, Natalia sold 48/2 = 24 clips. In April and May, she sold 48+24=72 clips." So they interpret 48 friends as 48 clips. Good. Need answer 72.

Need final with calculation and marker. Ensure final answer after '####'. Maybe include "Natalia sold half of 48 in May: 24. Total: 48 + 24 = 72. #### 72" This satisfies. Need no extra after final. Use "#### 72" after explanation. The final answer after '####' could be on same line. Need maybe "#### 72" exactly.

Need not mention. Done.
=============== Content =================
In May, Natalia sold \(48 \div 2 = 24\) clips.

Altogether: \(48 + 24 = 72\)

#### 72
```

</Accordion>

### Images

Send images as `image_url` content parts, as URLs or base64 `data:` URIs, in any position relative to the text. The image examples draw their test images locally with Pillow, so save this helper next to them as `hy4_images.py`:

<Accordion title="Test image helper (Python)">

```python hy4_images.py
import base64
import io

from PIL import Image, ImageDraw, ImageFont


def make_image(shape, color, label):
    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    box = (100, 60, 348, 308)
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    else:
        draw.polygon([(224, 60), (348, 308), (100, 308)], fill=color)
    draw.text((110, 350), label, fill="black", font=ImageFont.truetype("DejaVuSans-Bold.ttf", 40))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
```

</Accordion>

<Accordion title="Single image (Python)">

```python Example
from openai import OpenAI

from hy4_images import make_image

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="tencent/Hy4-FP8",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": make_image("circle", "red", "RED 7231")}},
            {"type": "text", "text": "Describe the shape, its color and the text in the image."},
        ],
    }],
    max_tokens=128,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
)

print(response.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
The shape in the image is a **circle** (or a round disc). Its color is a bright, vivid **red** (specifically labeled as "RED 7231"). Below the red circle, the text reads **"RED 7231"** in bold, black, uppercase letters (with a space between "RED" and "7231").
```

</Accordion>

<Accordion title="Multiple images (Python)">

```python Example
from openai import OpenAI

from hy4_images import make_image

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="tencent/Hy4-FP8",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": make_image("circle", "red", "RED 7231")}},
            {"type": "image_url", "image_url": {"url": make_image("triangle", "green", "GREEN 5582")}},
            {"type": "text", "text": "There are two images. For each image, in order, give the text written under it, one per line as 'Image k: <text>'."},
        ],
    }],
    max_tokens=128,
    extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
)

print(response.choices[0].message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Image 1: RED 7231
Image 2: GREEN 5582
```

</Accordion>

### Video

Send a clip as a `video_url` content part. The server samples frames at 2 fps and adds a timestamp to each frame, so no client-side frame extraction is needed:

<Accordion title="Video (Python)">

```python Example
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="tencent/Hy4-FP8",
    messages=[{
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": "https://github.com/sgl-project/sgl-test-files/raw/refs/heads/main/videos/jobs_presenting_ipod.mp4"}},
            {"type": "text", "text": "What is happening in this video? Answer in two sentences."},
        ],
    }],
    max_tokens=2048,
)

message = response.choices[0].message
print("Reasoning:", message.reasoning_content)
print("Content:", message.content)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: The user wants a two-sentence summary of what is happening in the video.

1. **Analyze the video content**:
   - It's a clip of Steve Jobs (an Apple keynote presentation, likely the iPod Nano introduction event, with captions/overlay from Evan Carmichael).
   - On the giant screen behind him, there's an image of blue jeans with the small "watch pocket" / coin pocket.
   - Subtitles: "Anyone know what this pocket is for?", "I've always wondered that", "well now we know because this is the new iPod Nano.o" (typo in caption "iPod Nano.o" / "iPod Nano").
   - Jobs reveals the completely redesigned, ultra-thin, tall iPod Nano by pulling it out of his own jeans' small coin pocket (and showing it on screen being pulled out of the giant jeans pocket / holding it up).
   - He then holds up the incredibly thin profile of the new iPod Nano to the audience/camera while the screen zooms in on its slim side profile.

2. **Draft the summary**:
   - Sentence 1: In this classic Apple keynote clip, Steve Jobs stands on stage in front of a giant screen showing the small coin pocket on a pair of jeans, joking about its purpose.
   - Sentence 2: He reveals the ultra-thin new iPod Nano by pulling it from that very pocket and holding up its remarkably slim profile for the audience to see.

3. **Refine for conciseness, accuracy, and flow (exactly two sentences)**:
   "In this Apple keynote presentation, Steve Jobs addresses the mystery of the tiny coin pocket in blue jeans displayed on the screen behind him. He reveals that its purpose is to hold the newly redesigned, incredibly thin iPod Nano, pulling the device out and showing its slim profile to the audience."

4. **Check against constraints**:
   - Two sentences? Yes.
   - Accurate? Yes (the coin pocket / watch pocket, Steve Jobs, iPod Nano launch, thinness demonstration).
Content: In this classic Apple keynote presentation, Steve Jobs addresses the mystery of the tiny coin pocket on a pair of jeans displayed on the giant screen behind him. He reveals that it is made for the newly redesigned iPod Nano, pulling the device out of his own pocket and holding up its remarkably thin profile for the audience to see.
```

</Accordion>

### Tool calling

The Deploy recipes pass `--tool-call-parser auto`, which resolves to the `hunyuan` parser and returns calls in `message.tool_calls`. The model thinks before calling, so print `reasoning_content` too:

<Accordion title="Tool calling (Python)">

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
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="tencent/Hy4-FP8",
    messages=[{"role": "user", "content": "What's the weather like in Paris right now? Use the tool."}],
    tools=tools,
    tool_choice="auto",
    max_tokens=1024,
)

message = response.choices[0].message
print("Reasoning:", message.reasoning_content)
print("Content:", message.content)
for call in message.tool_calls or []:
    print("Tool call:", call.function.name, call.function.arguments)
print("Finish reason:", response.choices[0].finish_reason)
```

</Accordion>

<Accordion title="Example Output">

```text Output
Reasoning: The user wants to know the weather in Paris right now and explicitly asks to use the tool. I need to call the get_weather function with city="Paris". Let me do that.
Content: I'll check the current weather in Paris for you.
Tool call: get_weather {"city": "Paris"}
Finish reason: tool_calls
```

</Accordion>

`tool_choice: "required"`, a named function and `strict` tools are enforced by a structural grammar, as are `response_format` and `regex`. Send these requests to the High-Throughput recipe: the Low-Latency recipe rejects them with HTTP 400 and the message `DFLY speculative decoding does not support grammar decoding.`

## Notes and limitations

- **Memory fraction**: both strategies use the server's automatic memory sizing. If you override memory settings for vision or long-context peaks, leave enough room for the KV pool and graph capture.
- **DSA indexer fusion**: on CUDA the DSA indexer runs fused by default, and `SGLANG_DISABLE_DSA_INDEXER_FUSION=1` selects the unfused path. The two modes store indexer keys in different bases and the HiCache storage key does not distinguish them, so with a persistent L3 backend (for example `--hicache-storage-backend file`) use a fresh storage directory or namespace whenever the fusion mode changes, and never share one L3 namespace between fused and unfused workers.
- **Checkpoint config**: the vision tower always runs unquantized, and the loader rejects quantized vision tensors and `tie_word_embeddings: true`. Vendor `soft_logits_capping` / `soft_logits_capping_logits` maps to the final logit softcap.
- **Video**: there is no video accuracy benchmark for these recipes.
