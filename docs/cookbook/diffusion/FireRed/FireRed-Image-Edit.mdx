---
title: FireRed-Image-Edit
description: "Deploy FireRed-Image-Edit 1.0 and 1.1 with SGLang Diffusion for general-purpose and multi-image editing."
metatags:
    description: "Serve FireRed-Image-Edit 1.0 and 1.1 for identity-preserving, multi-image, text-style, and photo-restoration edits with SGLang Diffusion."
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/FireRedTeam/firered-image-edit.jsx';

<DiffusionModelTags tags={["image editing", "multi-image edit", "identity consistency", "Qwen-Image-Edit architecture", "40 steps"]} />

## 1. Quick start

Install SGLang with diffusion support by following the [SGLang Diffusion installation guide](/docs/sglang-diffusion/installation), then start the server and send an edit request.

<Deployment config={config} />

The request returns a base64 PNG; the command above writes it to `output.png`.
The server keeps the model defaults of 40 steps and guidance 4.0. Each
checkpoint needs its own server.

## 2. Model capabilities

[FireRed-Image-Edit](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1) from FireRedTeam is a general-purpose image editor. It is strong at keeping a person recognizable across edits, preserving the style of existing text, restoring old photos, and combining elements from several input images, such as a virtual try-on.

Choose 1.1 for new deployments: it improves identity consistency and multi-image conditioning over 1.0 and uses the same architecture, so the serving commands are identical. FireRed runs on SGLang's native Qwen-Image-Edit-Plus pipeline, so the Qwen-Image-Edit runtime options apply to it.

| Checkpoint | Best fit | Boundary |
| --- | --- | --- |
| `FireRedTeam/FireRed-Image-Edit-1.1` | New deployments, portrait and multi-image edits | Same latency and memory as 1.0 on H200 |
| `FireRedTeam/FireRed-Image-Edit-1.0` | Reproducing results produced with 1.0 | Superseded by 1.1 |

## 3. Deployment details

On two GPUs the recommended layout is CFG parallelism: the conditional and
unconditional branches run on separate GPUs, and each GPU holds a full copy of
the model. On H200 it halves the request time and was faster than Ulysses
sequence parallelism on the same two GPUs. Both two-GPU layouts change
floating-point rounding, so a fixed seed gives a visually equivalent image that
is not bit-identical to the one-GPU output.

| Checkpoint | Hardware | GPUs | Layout | Edit request (1024x1024, 40 steps) | Peak memory / GPU |
| --- | --- | --- | --- | --- | --- |
| 1.1 | H200 | 1 | Single GPU | 21.1 s | 59.9 GiB |
| 1.1 | H200 | 2 | CFG parallel (recommended) | 10.7 s | 62.7 GiB |
| 1.1 | H200 | 2 | Ulysses 2 | 12.4 s | 64.4 GiB |
| 1.0 | H200 | 1 | Single GPU | 21.0 s | 59.9 GiB |
| 1.0 | H200 | 2 | CFG parallel (recommended) | 10.7 s | 62.7 GiB |

Measured with `sglang serve` on SGLang `main` at `385262cabbae1`: median of three warmed
`/v1/images/edits` requests with a fixed input image, prompt, and seed; the time
is the full HTTP request, and memory is the `nvidia-smi` peak. A request with
two input images took 40.3 s on one H200. The command builder marks other hardware and GPU
counts as unverified.

See the [CLI reference](/docs/sglang-diffusion/api/cli) for the remaining server flags.

## 4. Request examples

### 4.1 Offline generation

```bash Command
sglang generate --model-path FireRedTeam/FireRed-Image-Edit-1.1 \
  --image-path input.png \
  --prompt "Restore this old photo and colorize it naturally." \
  --width 1024 --height 1024 --seed 0 --save-output
```

This offline command has not been run end to end in the current release; the
HTTP server above is the verified path.

### 4.2 Multiple input images

Send each input image as a separate `image` field and refer to them in the
prompt.

```bash Command
curl -sS --fail-with-body http://127.0.0.1:30000/v1/images/edits \
  -F "image=@person.png" \
  -F "image=@jacket.png" \
  --form-string 'prompt=Dress the person in the first image in the jacket from the second image.' \
  --form-string seed=0 \
  --form-string response_format=b64_json \
  | python3 -c "import base64,json,sys; open('output.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

## 5. Run in ComfyUI

import { ComfyUISupport } from '/src/snippets/diffusion/comfyui-support.jsx';

<ComfyUISupport model="image" />
