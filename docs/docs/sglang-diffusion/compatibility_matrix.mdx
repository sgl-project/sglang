---
title: "Supported Models"
description: "Browse model families and public checkpoints supported by SGLang Diffusion."
---

import { DiffusionModelCatalog } from '/src/snippets/diffusion/model-catalog.jsx';

## Ming-Image integration status

Native support covers `inclusionAI/Ming-Image-0.1-Design` (generation and
single-image editing) and `inclusionAI/Ming-Image-0.1-Design-Layer` (ordered RGBA
layers). The checkpoints are not interchangeable. No remote modeling code is
used; checkpoint directories do not need conversion to a Diffusers layout.

H200 full-checkpoint checks include 2048-square generation, 1024-square
four-layer decomposition, and repeated 512-square HTTP generation and editing.
The HTTP checks cover cold starts, request warmup, layerwise offload, opt-in VAE
tiling, and one or two sequential generations per request,
requiring identical pixels across repeated requests with fixed inputs.

Two-H200 functional checks cover DiT TP2, Ulysses2, Ring2 with FlashAttention,
CFG parallelism for Design-Layer, encoder folding, and spatial VAE decoding.
Single-GPU checks cover DiT and encoder layerwise offload, Cache-DiT,
SageAttention, actual breakable CUDA graph replay, and dynamic LoRA load/remove
with a synthetic adapter. These checks do not establish quality equivalence for lossy
optimizations or bit-identical output across parallel topologies. Quantized
checkpoints and other GPU families remain unverified.

Run the opt-in HTTP regression tests on a CUDA host with sufficient memory:

```bash
SGLANG_TEST_MING_IMAGE=1 python -m pytest -q \
  python/sglang/multimodal_gen/test/server/test_server_ming_image.py
```

See the [Ming-Image cookbook](/cookbook/diffusion/inclusionAI/Ming-Image) for
launch commands, request parameters, and model-specific limits.

## Supported model inventory

Use a listed checkpoint as `--model-path` with `sglang generate` or
`sglang serve`. This registry-backed list contains known public entry points;
family detection may also support compatible local directories. Open the linked
Cookbook recipe for launch commands, optimizations, adapters, and model-specific
notes.

<Tabs>
  <Tab title="Image and 3D">
    <DiffusionModelCatalog category="image" />
  </Tab>
  <Tab title="Video and audio">
    <DiffusionModelCatalog category="video" />
  </Tab>
  <Tab title="World and action">
    <DiffusionModelCatalog category="world" />
  </Tab>
</Tabs>

## Qwen-Image 2.1 integration status

Qwen-Image 2.1 uses a separate pipeline from older Qwen-Image models. See its
[cookbook](/cookbook/diffusion/Qwen-Image/Qwen-Image-2.1) for the public checkpoint,
hardware recipes, and the validated scope of parallelism, quantization, and offload.
