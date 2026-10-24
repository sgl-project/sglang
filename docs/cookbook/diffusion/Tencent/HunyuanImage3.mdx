---
title: HunyuanImage-3.0-Instruct
description: Deploy HunyuanImage-3.0-Instruct for text-to-image generation and reference-image editing.
tag: NEW
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';

<DiffusionModelTags tags={["image", "text-to-image", "reference-image editing", "80B MoE", "13B active"]} />

## 1. Quick start

Follow the [SGLang Diffusion installation guide](/docs/sglang-diffusion/installation). The command builder below targets Linux/CUDA; use the [Ascend installation guide](/docs/hardware-platforms/ascend-npus/getting-started/installation) for NPU environments. Exact deployment recipes remain unverified unless explicitly marked otherwise.

```bash
uv pip install -e 'python[diffusion]'
```

import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/tencent/hunyuan-image3.jsx';

<Deployment config={config} />

## 2. Model capabilities

[HunyuanImage-3.0-Instruct](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct) combines a text-and-image transformer with diffusion image generation. It supports text-to-image requests and instruction-guided editing with reference images. The native pipeline reuses the transformer's text prefix during denoising and uses a SigLIP2 encoder plus a convolutional VAE for image conditioning.

Choose it when one checkpoint must handle both generation and reference-based editing. Its 80B total parameters, despite approximately 13B active per token, make weight residency a major deployment constraint. This recipe covers the Instruct checkpoint, not the separate distilled release. Start with tensor parallelism and the default 50-step, guidance-2.5 sampling profile before testing acceleration options.

## 3. Reference-image editing

Use the same server with the images edit endpoint. Repeat `image[]` for multiple reference images. The file paths below are local to the client.

```bash
curl -sS http://localhost:30010/v1/images/edits \
  -F model=tencent/HunyuanImage-3.0-Instruct \
  -F 'prompt=Change the background to a quiet seaside town; preserve the subject.' \
  -F 'image[]=@reference.png' \
  -F num_inference_steps=50 \
  -F guidance_scale=2.5 \
  -F seed=42 \
  -F response_format=b64_json
```

For offline generation, pass the same model and topology to `sglang generate`:

```bash
sglang generate --model-path tencent/HunyuanImage-3.0-Instruct \
  --num-gpus 2 --tp-size 2 --ulysses-degree 1 \
  --prompt "Change the background to a quiet seaside town; preserve the subject." \
  --image-path reference.png --num-inference-steps 50 --guidance-scale 2.5 \
  --seed 42 --save-output
```

## 4. Request semantics

- **Resolution:** `width` and `height` select an aspect ratio by default, not an exact generation canvas. The processor selects a native bucket; decoding crops or pads it. Editing without an explicit size uses the first reference image's dimensions. Use `output_size_mode="exact_size"` when exact output dimensions matter; this resamples the decoded result rather than changing its native bucket.
- **Prompting:** `bot_task="image"` and `system_prompt="en_unified"` are the defaults. Other `bot_task` values change the response prefix. They do not run a separate autoregressive reasoning or recaptioning loop. `cot_text` accepts an already-generated reasoning/recaption prefix.
- **Multiple outputs:** `n` on the HTTP API or `--num-outputs-per-prompt` offline controls output count. More outputs require additional activation memory.

## 5. Deployment boundaries

For an Ascend environment with four visible NPU devices and sufficient aggregate memory, the equivalent TP4 launch is:

```bash
sglang serve --model-path tencent/HunyuanImage-3.0-Instruct \
  --num-gpus 4 --tp-size 4 --ulysses-degree 1 --port 30010
```

Here `--num-gpus` counts visible accelerator devices, including NPUs. Follow the platform installation guide for compatible PyTorch/CANN packages; do not install the CUDA extra into an NPU environment.

Use TP to shard transformer weights; keep Ulysses and Ring at 1. This pipeline does not implement sequence-parallel attention or CFG-parallel execution. The vision encoder and VAE remain replicated, so increasing TP does not divide every component's memory usage. Approximately 160 GB of BF16 transformer weights is only a capacity estimate, not a peak-VRAM measurement: leave room for those components, activations, and loading buffers.

Compatible requests can be grouped with `--batching-max-size` and `--batching-delay-ms`. Begin at batch size 1, then validate peak memory at the largest reference-image count and output size you intend to serve. Different prompt/seed values can share a batch; incompatible conditioning and sampling settings cannot.

The pipeline has a model-specific Cache-DiT adapter. Caching reuses intermediate results and is not numerically lossless; measure quality and latency on your workload before enabling it. See [Cache-DiT](/docs/sglang-diffusion/cache_dit). SP, CFG parallel, quantization, and full-checkpoint CUDA performance are not claimed as verified by this recipe.
