---
title: HiDream-O1-Image
description: "Deploy HiDream-O1-Image with SGLang Diffusion."
metatags:
    description: "Serve HiDream-O1-Image for pixel-space text-to-image generation with SGLang Diffusion."
tag: NEW
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/HiDream/hidream-o1-image.jsx';

<DiffusionModelTags tags={["image", "text-to-image", "pixel-space", "8B UiT"]} />

## 1. Quick start

Follow the [SGLang Diffusion installation guide](/docs/sglang-diffusion/installation),
then choose a recipe below. TP 1 and TP 2 on a single H200 are the verified shapes.

<Deployment config={config} />

## 2. Model capabilities

[HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image) is an 8B pixel-level Unified Transformer (UiT). Unlike the usual latent-diffusion layout, it has neither a VAE nor a separate text encoder: one Qwen3-VL backbone embeds the prompt and predicts clean 32x32 RGB patches directly, and the flow-matching head reads its timestep from a dedicated token in the prompt. It loads through SGLang's native `HiDreamO1ImagePipeline`.

Choose it for text-to-image at its trained 2K-class resolutions, including legible in-image text. The checkpoint ships a stock `qwen3_vl` config with no `model_index.json`, so the pipeline loads the transformer itself; the tradeoff is a narrower feature surface: bf16 only, one prompt per request, no sequence parallelism, and monolithic serving only.

| Model | Hugging Face model ID | Notes |
| --- | --- | --- |
| HiDream-O1-Image | `HiDream-ai/HiDream-O1-Image` | Text-to-image, bf16 only |

## 3. Generate an image

One-shot generation from the CLI:

```bash Command
sglang generate \
  --model-path HiDream-ai/HiDream-O1-Image \
  --prompt "A cat is sitting next to a sign that says 'HiDream-O1'" \
  --num-inference-steps 50 \
  --guidance-scale 5.0 \
  --save-output
```

Or through the OpenAI-compatible image generation API once the server is up:

```python Python
import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:30010/v1")

response = client.images.generate(
    model="HiDream-ai/HiDream-O1-Image",
    prompt="A cinematic photo of a quiet lakeside cabin at sunrise",
    n=1,
    response_format="b64_json",
)

image_bytes = base64.b64decode(response.data[0].b64_json)
with open("hidream_o1.png", "wb") as f:
    f.write(image_bytes)
```

## 4. Configuration tips

- The model is released in bfloat16 only; `--dit-precision` / `--component-precisions` overrides are rejected rather than silently downcast.
- The trained resolutions are 2048x2048, 2304x1728, 2560x1440, 2496x1664, 3104x1312, 2304x1792 and their transposes. Nothing snaps: an off-grid size is warned about in the server log and then rendered exactly as requested, and because pixel patches carry no scale invariance the quality degrades badly.
- Sequence parallelism is unsupported: the prompt span is causal while the image span is bidirectional, which needs a dense `[B, 1, S, S]` mask that cannot be sharded along the sequence. Use `--tp-size` for multi-GPU instead. `--cfg-parallel-size 2` is accepted at the default `--guidance-scale 5.0`, since it splits the two CFG branches across GPUs, but it has not been verified end to end.
- Encoder/denoiser/decoder disaggregation does not apply — one component does all three, so the pipeline only runs in the monolithic role.
- The defaults are the reference `full` recipe: 50 steps, `--guidance-scale 5.0`, shift 3.0 (fixed). Classifier-free guidance costs a second forward pass per step, so `--guidance-scale 1.0` halves the step cost at some quality loss.
- Noise is drawn on the host, so a given `--seed` reproduces the same image on any accelerator.

## 5. Run in ComfyUI

import { ComfyUISupport } from '/src/snippets/diffusion/comfyui-support.jsx';

<ComfyUISupport model="image" note="HiDream-O1-Image has no VAE and no integrated KSampler executor. Use server mode; the plugin workflow has not been validated for this model." />
