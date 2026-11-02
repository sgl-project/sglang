---
title: Qwen-Image 2.1 and Turbo
description: "Run Qwen-Image 2.1 and its eight-step Turbo variant with SGLang Diffusion: text-to-image, reference-image editing, and transparent RGBA output."
tag: NEW
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/Qwen/qwen-image-2.1.jsx';

<DiffusionModelTags tags={["RGBA image", "text-to-image", "image editing", "multi-image references", "block-causal attention"]} />

## 1. Quick start

Use a nightly Docker image containing this integration, or install from source,
following the [SGLang Diffusion installation guide](/docs/sglang-diffusion/installation).
Run the commands below inside that environment. Select **Checkpoint weights** for
`Qwen/Qwen-Image-2.1` or `Qwen/Qwen-Image-2.1-Turbo`; you can also set a local
checkpoint directory under **Variables**. The recipes
target NVIDIA CUDA on Linux; the picker marks which workloads have
been verified with the full checkpoint.

<Deployment config={config} />

Use **Setup** to select text-to-image, single-image editing, or multi-image
editing. **Server** controls placement, attention, encoder scheduling, VAE
tiling, graph execution, and request batching. **Request** controls the background, resolution,
steps, and output count. Set reference PNG paths under **Variables**; edits
upload files from the machine running cURL, so they need not exist on the server.

Hardware selection applies the recommended placement for that GPU. H200,
B200, RTX PRO 6000 96GB, and DGX Spark keep weights resident; RTX 5090 and RTX 4090
offload selected components to fit the full pipeline.
Untested topologies and feature combinations remain selectable and are labeled
**Unverified**. Invalid topology combinations disable Copy. This integration
currently uses the Python/source command; no published Docker image is verified.

Both request modes return base64 PNGs. To save all returned images, append
`> response.json` to the request command, then run:

```bash Command
python - <<'PY'
import base64
import json
from pathlib import Path

for i, item in enumerate(json.loads(Path("response.json").read_text())["data"]):
    Path(f"output-{i}.png").write_bytes(base64.b64decode(item["b64_json"]))
PY
```

### Recommended hardware settings

The picker defaults to native BF16/FP32 precision, exact attention, eager
execution, and full-image VAE decoding.
Commands omit default values, including one GPU, encoder auto scheduling, and
batch size one. Explicit placement and attention overrides preserve each recipe.

| GPU | Placement / attention | Generation | Edit | Peak VRAM |
| --- | --- | --- | --- | --- |
| H200 141GB | Resident / FlashAttention | 4.48 s | 5.29 s | 38.4 GiB |
| B200 192GB | Resident / FlashAttention | 2.46 s | 3.02 s | 38.5 GiB |
| RTX PRO 6000 96GB | Resident / Torch SDPA | 8.03 s | 9.63 s | 38.4 GiB |
| RTX 4090 24GB | DiT and VAE resident, encoder layerwise offload / FlashAttention | 18.68 s | 21.68 s | 22.7 GiB |
| DGX Spark 128GB unified | Resident / Torch SDPA | 35.36 s | 42.23 s | — (unified) |

Measured with the original checkpoint on 2026-09-20 at 1024×1024, 40 steps, CFG 1, and one RGBA PNG per
request. Times are median HTTP latency after warmup, including PNG serialization
and excluding startup; VRAM is the sampled request-phase peak. Prompts and
software versions affect both latency and memory use.

On one RTX 5090, native-precision, eager text-to-image or single-reference edits
with one output and batching off keep the DiT and VAE resident and stream the
encoder with Torch SDPA. Multiple reference images, multiple outputs, or request
batching select DiT layerwise offload instead; use the updated Server command
when changing these settings. The picker identifies which exact combinations
are verified. Both RTX 5090 and RTX PRO 6000 use SDPA when FlashAttention is
selected in this runtime. CPU offload requires host RAM.

### DGX Spark

Select **DGX Spark** for one GB10 GPU on Linux ARM64 with CUDA 13. Use the
source installation above. The recommended configuration keeps all components
resident, uses native BF16/FP32 precision, and lets the runtime select Torch SDPA:

```bash Command
sglang serve \
  --model-path Qwen/Qwen-Image-2.1 \
  --performance-mode speed
```

The [128 GB unified memory](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
is shared by the CPU and GPU. CPU offload is unnecessary for the verified
single-image 1024×1024 workload. Keep full-image VAE decoding and eager execution.
Generation, editing, transparent generation, and transparent editing were
verified with PyTorch 2.13.0+cu130. Spark reports no separate VRAM usage in `nvidia-smi`.
This recipe covers one Spark; multi-node deployment and batching remain unverified.

### Batching

Keep **Request batching → Off** and **Outputs → 1** for interactive use.
Batching increases individual request latency and does not guarantee higher
throughput. Measure your workload before enabling it.

Cross-request batching merges compatible text-to-image requests. Image edits
run separately; **Outputs** controls multiple images within one request.
On RTX 4090, selecting multiple outputs or request batching switches to DiT
layerwise offload for memory headroom. Restart with the updated **Server** command.

Batching preserves native precision but can change floating-point rounding and
output pixels, even with the same seed. See
[Inference batching](/docs/sglang-diffusion/dynamic_batching) for admission rules
and metrics.

### Turbo sampling

Select **Qwen-Image 2.1 Turbo** under **Setup → Checkpoint weights** and restart
with the generated Server command. Requests use the checkpoint's preset eight-step
sigma grid automatically; omit `num_inference_steps`. Setting that field to 8
on the original checkpoint does not reproduce Turbo.

For offline generation:

```bash Command
sglang generate --model-path Qwen/Qwen-Image-2.1-Turbo \
  --prompt "A ceramic teapot on a wooden table in soft morning light"
```

The preset comes from `sample_sigmas` in `model_index.json`, with scheduler
settings loaded from `scheduler/scheduler_config.json`. Preserve both when copying
or exporting the checkpoint. Changing only `num_inference_steps` does not override
the preset. Custom grids are experimental and can reduce quality; the picker keeps
Turbo on its published schedule. The default CFG scale is 1 for both checkpoints.
See the [Turbo model card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo).

Turbo was verified at 1024×1024 with native precision and eager execution on
H200: single-GPU generation, editing, transparent generation and editing,
multi-image editing, and two outputs; TP×2 generation and editing also passed.
The picker distinguishes these from untested Turbo hardware and combinations.

The hardware latency table above describes the original 40-step checkpoint.
Turbo has the same architecture and weight-memory footprint; fewer denoising steps
do not eliminate encoder or VAE costs. Existing quantization and acceleration
results for the base checkpoint do not establish Turbo quality or speed.

## 2. Model capabilities

Qwen-Image 2.1 supports text-to-image generation, single- and multi-image editing,
and RGBA output. Both the original and eight-step Turbo checkpoints support all
these modes. Choose Turbo for shorter denoising runs; compare the two checkpoints
on your prompts when fidelity is the priority.

For multi-round editing, send the previous output as the next reference image.
The server does not retain conversation state. Repeated prompts and reference
images can reuse exact encoder outputs and VAE posteriors through the default
[conditioning cache](/docs/sglang-diffusion/caching-acceleration#conditioning-cache).
Changing the prompt or image recomputes their joint embedding; unchanged images
can still reuse vision features. Condition-prefix KV reuse stays within each
request. Use `--disable-conditioning-cache` to turn off cross-request reuse.

## 3. Checkpoint layout

The checkpoint directory must contain `model_index.json` and the `processor`,
`text_encoder`, `transformer`, `vae`, and `scheduler` subdirectories. The processor
includes the Qwen3-VL tokenizer assets; no separate tokenizer directory is needed.
Use `--model-id Qwen-Image-2.1` when your local checkpoint directory has another
name. Older Qwen-Image and Qwen-Image-Edit transformer/VAE weights are incompatible.

Keep SGLang's installed dependencies. Its native encoder preserves the
reference's Transformers 4.57.3 conditioning semantics without requiring a
runtime-wide downgrade.

### Transparent PNG output

Choose **Transparent / alpha** under Request and describe an isolated subject
on a transparent background in the prompt. The picker adds this instruction
and selects PNG. `background: "transparent"` alone does not change conditioning
or remove the background; JPEG cannot retain alpha.

PNG references retain their alpha channel during editing; RGB references
receive an opaque alpha channel. The model predicts continuous alpha values,
including partly transparent edges, without thresholding or background removal.

## 4. Offline requests

Defaults are 1024×1024, CFG 1, and seed 42; output saving is enabled.
The original checkpoint defaults to 40 steps; Turbo uses its preset 8-step grid.
For GPUs that need offload, also pass the placement flags from the picker.

### Text-to-image

```bash Command
sglang generate \
  --model-path Qwen/Qwen-Image-2.1 \
  --prompt "A capybara reading a book by candlelight"
```

### Image-conditioned editing

```bash Command
sglang generate \
  --model-path Qwen/Qwen-Image-2.1 \
  --image-path /path/to/input.png \
  --prompt "Move the scene to a snowy mountain at sunrise"
```

Height and width must be positive multiples of 32. Reference images preserve
their aspect ratio and are resized to approximately the requested output area;
the same resized image feeds the VLM and VAE. Image labels are deterministic
(`Picture 1`, `Picture 2`, and so on). Multiple outputs receive independent
noise seeds and independent prefix caches.

## 5. Runtime features

Both checkpoints use Euler flow matching with CFG disabled by default: 40 steps
for the original, or the preset 8-step grid for Turbo. For CFG, provide
`--negative-prompt` and `--guidance-scale` greater than one. The API requires a
text prompt; precomputed embeddings alone are insufficient.

- **Parallelism:** TP, Ulysses, Ring, CFG parallelism, and encoder folding are
  available in the picker. The target token count, `(height / 16) × (width / 16)`,
  must be divisible by the SP degree. Ring requires FlashAttention or SageAttention.
- **Memory:** use the hardware's recommended placement. **All components
  layerwise** also streams encoder and VAE blocks, trading transfers for lower
  device memory.
- **VAE:** full-image decoding is the default except on AMD gfx1151 (Strix Halo).
  Tiling can change pixels near boundaries. With two or more GPUs, **Spatial shard**
  distributes full-image decoding without enabling tiling; floating-point
  rounding can still differ.

On gfx1151, VAE tiling defaults to enabled to avoid reported full-frame decode
hangs at 896px and higher. Explicit CLI, pipeline config, and Python API settings
take precedence: `--vae-tiling false` disables tiling but may encounter the hang.
Defaults on NVIDIA CUDA and other platforms are unchanged.

See the [performance guide](/docs/sglang-diffusion/performance-optimization)
for shared runtime options.

### Validation boundaries

The picker marks exact tested HTTP deployments, not every combination of the
features below. Functional checks do not establish quality for lossy settings.
The feature checks below cover the original checkpoint; see
[Turbo sampling](#turbo-sampling) for the Turbo validation scope.

- **Parallelism and offload:** H200/B200 full-checkpoint checks cover TP,
  Ulysses, Ring with FlashAttention, CFG parallelism, encoder folding, and
  all-component layerwise offload. Spatial VAE sharding was checked on two B200
  GPUs; floating-point rounding can differ from full-image decoding.
- **Disaggregation:** encoder, denoiser, and decoder roles were checked on three
  B200 GPUs with same-host Mooncake TCP at 512px/4 steps. Multi-host RDMA and
  multi-rank roles remain unverified for this model.
- **Quantized exports:** serialized FP8, native-name Q4_0 GGUF, and calibrated
  ModelOpt NVFP4 component exports were checked on B200, including TP2 with
  encoder folding and separate offload checks. A community ComfyUI-GGUF Q4_K_M
  DiT, which also quantizes the projections outside the blocks, was checked for
  text-to-image on one RTX 5090. These results do not establish support for
  arbitrary community exports, other GGUF types, or other hardware.
  The NVFP4 export used six calibration requests; assess image and alpha quality
  on your workload.
- **LoRA:** synthetic Diffusers-format adapters were checked on one B200 and
  TP2 with encoder folding. Trained-adapter quality remains unverified.

### Quantization

Native precision is the default. Quantization changes image and alpha values;
check quality on your own prompts and reference images. Set compatible component
paths under **Variables** when choosing an exported format. Adding quantization
metadata to native weights does not convert them.

For online FP8, use `--component-quantizations.transformer fp8`,
`--component-quantizations.text_encoder fp8`, or both.

### Serialized FP8 components

Select a **Serialized FP8** option and set the exported component directories.
Each directory needs its architecture `config.json`, weights, and quantization
metadata. Use `--component-paths.transformer` and/or
`--component-paths.text_encoder`; omit online quantization flags.
See the [quantization guide](/docs/sglang-diffusion/quantization) for formats.

### GGUF components

Select a **GGUF** option and set the `.gguf` files. The picker uses
`--component-weights-paths.transformer` and/or
`--component-weights-paths.text_encoder`, retaining architecture configs from the
base checkpoint. Each file must contain the entire component with native tensor
names. GGUF reduces weight storage but does not guarantee lower latency.
See the [GGUF guide](/docs/sglang-diffusion/quantization#gguf).

### NVFP4 components

NVFP4 requires Blackwell and compatible ModelOpt exports. Select the component
directories using `--component-paths.transformer` and/or
`--component-paths.text_encoder`. Keep the FlashInfer backend at `auto` on
RTX 5090, RTX PRO 6000, and DGX Spark: TensorRT-LLM FP4 GEMM does not support SM12.x.
These GPUs remain unverified for this model's NVFP4 exports. See the
[NVFP4 guide](/docs/sglang-diffusion/quantization#modelopt-nvfp4).

### LoRA and execution options

Use `--lora-path` and `--lora-merge-mode dynamic|merge` or the runtime adapter APIs.
Diffusers adapter keys prefixed with `transformer.` map to the native DiT.

Keep eager execution as the default. Breakable CUDA Graph replay requires
matching resolution and condition-prefix length; unseen shapes run eagerly.
Text buckets alone do not guarantee replay. SageAttention and Cache-DiT can
change numerical results and require quality checks for your workload.

### Cache-DiT

Enable `--enable-cache-dit true` or `SGLANG_CACHE_DIT_ENABLED=true`. 2.1 prefix
KV is per layer: each block slices caches by `_layer_id`. Cache-DiT wraps
`transformer_blocks` and forwards the same extras to every layer; without that
slice, later layers reuse layer 0 and the image collapses to color noise.
See the [Cache-DiT guide](/docs/sglang-diffusion/cache_dit).
