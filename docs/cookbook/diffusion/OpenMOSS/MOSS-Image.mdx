---
title: MOSS-Image
description: "Serve MOSS-Image with native Qwen3.5 conditioning, FLUX.2 decoding, and request-level guidance."
tag: NEW
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/OpenMOSS/moss-image.jsx';

<DiffusionModelTags tags={["image", "text-to-image", "Qwen3.5 conditioning", "FLUX.2 decoder"]} />

## 1. Quick start

Use a [SGLang Diffusion Docker environment](/docs/sglang-diffusion/installation).
Until your image includes this integration, install the model-support PR inside
the container:

```bash Install model support
pip install 'sglang[diffusion] @ git+https://github.com/sgl-project/sglang.git@refs/pull/44325/head#subdirectory=python'
```

You need access to the selected Hugging Face checkpoint. The picker distinguishes
available implementation paths from hardware recipes that have not been measured.

<Deployment config={config} />

## 2. Model capabilities

MOSS-Image is a text-to-image model using Qwen3.5 text conditioning and a FLUX.2
image decoder. Its native pipeline accepts natural-language prompts and structured
layout descriptions through the same Images API. Structured prompts are passed
to the text encoder as text, not interpreted as a separate rendering language.

Choose Base for reproducing the reference comparison. The SFT identifier resolves
to the same pipeline family, but its checkpoint has not been validated here.
Generation is supported; image editing is not implemented. Keep the scheduler,
guidance recipe, attention backend, and precision fixed when comparing outputs.

## 3. Guidance and precision

Defaults are 1024 x 1024, 50 steps, flow shift 5, CFG 3, and an empty negative
prompt. Width and height must be divisible by 16. DiT and text-encoder weights
use BF16; latents, CFG arithmetic, scheduler updates, and VAE decode use FP32.
Autocast is disabled by the pipeline configuration; no extra CLI flag is needed.

The default two-pass recipe skips unconditional layer 22 inside the
`slg_start=0.01`, `slg_stop=1.0` window. It leaves the first step at full CFG.
Set `uncond_skip_layers=[]` for ordinary CFG. Three-pass SLG adds a skipped
conditional prediction and changes the sampling algorithm; it is not a lossless
speed optimization.

```json Three-pass request overrides
{
  "guidance_scale": 5,
  "uncond_skip_layers": [],
  "slg_layers": [22],
  "slg_scale": 3,
  "slg_start": 0.01,
  "slg_stop": 1.0
}
```

For the official pipeline's start-at-zero behavior, explicitly set
`slg_start=0.0` and `uncond_skip_layers=[22]`, including when using three-pass
SLG. Guidance fields are request-scoped and do not require a server restart.

`quality=exact` preserves the native rounding policy, but does not make different
attention backends identical. `quality=lossless` can additionally enable shared
VAE fast paths. Compare each configuration against its matched reference rather
than treating either quality name as a cross-backend guarantee.

## 4. Deployment features

Component and layerwise offload reuse the standard residency manager. Start with
Auto placement; use `--component-residency all=resident` only when weights and
request activations fit. Offload saves VRAM at the cost of host RAM and transfers.
The text encoder also supports layerwise offload through its native layer list.

Sequence parallelism uses the shared shard/pad/gather helpers and USP attention;
the text refiner stays replicated. Ulysses degree must divide the 24 DiT heads.
Ring requires a backend exposing softmax LSE, such as FlashAttention. CFG
parallelism uses the shared branch dispatcher, including the optional third SLG
branch. Multi-GPU checkpoint recipes remain experimental until measured on the
selected hardware and workload.

TP, Cache-DiT, quantization, LoRA, compilation, and breakable CUDA graphs do not
have validated recipes in this integration. Do not assume that a framework-wide
flag establishes model support. See [performance optimization](/docs/sglang-diffusion/performance-optimization)
for the shared feature contracts.

## 5. Batching

Dynamic batching uses the shared scheduler and is opt-in through the maximum
batch size. For example, on an H200:

```bash
sglang serve --model-path OpenMOSS-Team/MOSS-Image-Base \
  --batching-max-size 4 --batching-delay-ms 20
```

Compatible requests may have different prompts and seeds; resolution, step count,
negative prompt, and guidance settings must match. The maximum size limits total
outputs, not just HTTP requests. A request can also use `n` to generate multiple
images from one prompt. Higher batch sizes need more activation VRAM and can
increase individual-request latency; the default remains one output at a time.

At `quality=lossless` (the default) or `high`, the DiT groups samples by their actual text length
and restores their original output order. Different lengths run in separate
groups without padding or switching attention backends. Empty negative prompts
can share one group even when positive prompts differ. The text refiner retains
per-sample GEMM shapes to reduce numerical drift. Batched DiT GEMM shapes can
change floating-point rounding, which denoising can amplify into visible changes;
cross-batch-size pixel identity is not guaranteed. `quality=exact` retains the
per-sample DiT path instead. Neither mode promises batch-invariant text encoding
or VAE decoding. Benchmark your prompts and batch sizes before deployment.

On one H200, four concurrent requests with different prompts and seeds 42-45 measured
the following warmed HTTP throughput. Both paths use BF16 DiT/text encoder,
FP32 VAE, default attention backends, `quality=lossless`, resident components,
maximum batch size 4, and a 20 ms batching delay. Loading is excluded; PNG encoding,
transfer, and client decoding are included. Results average three measured waves
at 512 pixels and two at 1024, after one warmup wave per resolution.
The environment used PyTorch 2.14.1+cu130, Transformers 5.19.0, and eager execution.

| Workload | Per-sample DiT | Grouped DiT | Throughput Change |
| --- | --- | --- | --- |
| 512 x 512, 8 steps | 2.51 images/s | 2.92 images/s | +16% |
| 1024 x 1024, 20 steps | 0.293 images/s | 0.296 images/s | Essentially unchanged |

At 512 pixels, agreement with the per-sample DiT path ranged from 28.3 to 46.2 dB
PSNR and 0.939 to 0.996 SSIM. All four 1024-pixel outputs were pixel-identical.
These measure output agreement, not semantic quality or a guarantee for other
prompts. Visible differences remain possible; use `quality=exact`
to retain per-sample DiT execution. Fixed-batch repeated requests are checked
separately from cross-batch agreement.

## 6. Run in ComfyUI

import { ComfyUISupport } from '/src/snippets/diffusion/comfyui-support.jsx';

<ComfyUISupport model="image" />
