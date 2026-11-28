---
title: JoyAI-Image-Edit
description: "Deploy JoyAI-Image-Edit with SGLang Diffusion for instruction-guided and spatial image editing."
metatags:
    description: "Serve JoyAI-Image-Edit for instruction-guided image editing, object moves, rotations, and camera changes with SGLang Diffusion."
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/JoyAI/joyai-image-edit.jsx';

<DiffusionModelTags tags={["image editing", "single-image edit", "spatial editing", "camera control", "40 steps"]} />

## 1. Quick start

Install SGLang with diffusion support by following the [SGLang Diffusion installation guide](/docs/sglang-diffusion/installation), then start the server and send an edit request.

<Deployment config={config} />

The request returns a base64 PNG; the command above writes it to `output.png`.
The server keeps the model defaults of 40 steps and guidance 4.0, and picks the
output size from the input image's aspect ratio.

## 2. Model capabilities

[JoyAI-Image-Edit](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers) from JD edits one input image from a text instruction. A Qwen3-VL encoder reads the image and the instruction together, so the model handles edits that depend on where things are: moving an object into a marked region, rotating an object to a named view, or changing the camera angle while keeping the scene.

Choose it when the edit is spatial rather than stylistic. The spatial modes work best with the prompt templates below. One GPU needs about 59 GiB of memory, so plan for an 80 GB-class card or larger.

## 3. Deployment details

On two GPUs the recommended layout is CFG parallelism: the conditional and
unconditional branches run on separate GPUs, and each GPU holds a full copy of
the model. On H200 it halves the request time and was faster than Ulysses
sequence parallelism on the same two GPUs. Both two-GPU layouts change
floating-point rounding, so a fixed seed gives a visually equivalent image that
is not bit-identical to the one-GPU output.

| Hardware | GPUs | Layout | Edit request (1024x1024, 40 steps) | Peak memory / GPU |
| --- | --- | --- | --- | --- |
| H200 | 1 | Single GPU | 26.7 s | 58.7 GiB |
| H200 | 2 | CFG parallel (recommended) | 13.5 s | 51.4 GiB |
| H200 | 2 | Ulysses 2 | 16.0 s | 53.7 GiB |

Measured with `sglang serve` on SGLang `main` at `385262cabbae1`: median of three warmed
`/v1/images/edits` requests with a fixed input image, prompt, and seed; the time
is the full HTTP request, and memory is the `nvidia-smi` peak. The command builder marks other hardware and GPU
counts as unverified.

See the [CLI reference](/docs/sglang-diffusion/api/cli) for the remaining server flags.

## 4. Request examples

### 4.1 Offline generation

```bash Command
sglang generate --model-path jdopensource/JoyAI-Image-Edit-Diffusers \
  --image-path input.png \
  --prompt "Remove the construction structure from the top of the crane." \
  --seed 0 --save-output
```

### 4.2 Spatial editing prompts

JoyAI-Image-Edit was trained with three spatial prompt patterns. Keep the
wording close to these templates.

| Edit | Prompt template |
| --- | --- |
| Object move | `Move the <object> into the red box and finally remove the red box.` (draw the red box on the input image) |
| Object rotation | `Rotate the <object> to show the <view> side view.` with `<view>` one of `front`, `right`, `left`, `rear`, `front right`, `front left`, `rear right`, `rear left` |
| Camera control | `Move the camera.` followed by lines `- Camera rotation: Yaw <deg>°, Pitch <deg>°.`, `- Camera zoom: in/out/unchanged.`, `- Keep the 3D scene static; only change the viewpoint.` |

### 4.3 OpenAI Python client

```python Example
import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:30000/v1")
response = client.images.edit(
    model="jdopensource/JoyAI-Image-Edit-Diffusers",
    image=open("input.png", "rb"),
    prompt="Rotate the dog to show the left side view.",
    response_format="b64_json",
)
with open("output.png", "wb") as f:
    f.write(base64.b64decode(response.data[0].b64_json))
```

## 5. Run in ComfyUI

import { ComfyUISupport } from '/src/snippets/diffusion/comfyui-support.jsx';

<ComfyUISupport model="image" />
