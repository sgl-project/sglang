---
title: FLUX.3
description: Serve FLUX.3 Image Klein text-to-image generation and image editing with SGLang Diffusion.
tag: NEW
metatags:
    description: "Serve FLUX.3 Image Klein, Black Forest Labs' 5.6B text-to-image and image editing model, with SGLang Diffusion."
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/black-forest-labs/flux3.jsx";

<DiffusionModelTags tags={["image", "5.6B DiT", "text-to-image / image editing", "up to 10 references", "Base / step-distilled"]} />

## 1. Quick start

Use a [SGLang Diffusion Docker environment](/docs/sglang-diffusion/installation).
Until a released image contains this integration, install SGLang from `main`
inside it:

```bash
pip install 'sglang[diffusion] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python'
```

<Deployment config={config} />

<Note>
The picker selects `black-forest-labs/flux-3-image-klein-base` or
`black-forest-labs/flux-3-image-klein`, each a self-contained repository. Access
to the selected checkpoint is required. Keep `--model-path` at its repository
root; `--model-variant` selects only the quantized DiT. Both release BF16
checkpoints have single-GPU generation checks; see the validation boundaries
below. Unverified hardware or resource shapes remain selectable but are not
measured recommendations.
</Note>

<Warning>
FLUX.3 Image Klein is released under the
[FLUX Kommunity License v1.0](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md):
non-commercial use, plus commercial output generation for Qualifying Users with
less than US$5M in gross annualized revenue, subject to its terms. The
Qwen3-VL-4B-Instruct text encoder is Apache-2.0. Review the license before
production or commercial use.
</Warning>

For platform-specific installation, see the
[SGLang Diffusion installation guide](/docs/sglang-diffusion/installation).

## 2. Model capabilities

[FLUX.3 Image Klein](https://huggingface.co/collections/black-forest-labs/flux-3-image-klein)
is Black Forest Labs' 5.6B open-weights model for text-to-image generation and
image editing. It takes a prompt and up to 10 reference images and returns one
image; text-to-image, single-reference edits and multi-reference compositions
run through the same DiT, with Qwen3-VL-4B-Instruct as the text encoder. The
5.6B count covers the image transformer only.

Prompts can place elements with bounding boxes on a 0 to 1000 grid, then
recolor, move, replace or remove them; boxes guide placement and scale rather
than masking pixels. The model also renders text inside images. One SGLang
server serves text-to-image and editing, chosen per request. This image
pipeline does not serve the separate FLUX.3 Action robot policy.

| Checkpoint | Training | Steps | Guidance | Built for |
| --- | --- | --- | --- | --- |
| [Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) | step-distilled flow map | 16 | none | generation and editing, self-hosting; the checkpoint behind the BFL API |
| [Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) | undistilled | 50 | 3.5 | fine-tuning, LoRA training, research |

Each repository ships its BF16 DiT with FP8 (`fp8r`) and NVFP4 variants,
together with the text encoder and image autoencoder. Quantization and caching
are separate fidelity/latency tradeoffs, not lossless substitutes for the
selected checkpoint.

| Task | Endpoint | Conditioning |
| --- | --- | --- |
| Text to image | `POST /v1/images/generations` | Text prompt |
| Image to image | `POST /v1/images/edits` | Text prompt and 1 to 10 `image` uploads |

Start the server:

```bash
sglang serve \
  --model-path black-forest-labs/flux-3-image-klein-base \
  --host 0.0.0.0 \
  --port 30010
```

Sampling is set per request (`num_inference_steps`, `guidance_scale`, `size`,
`seed`, `n`); unset fields take the checkpoint's defaults from its
`config.json`. SGLang expands `n` into independently seeded requests and can
pack compatible requests into one DiT forward.

References are prepared as in the
[BFL reference pipeline](https://github.com/black-forest-labs/flux-3-image-klein).
A single reference is
fitted under 4 MP, reflect-padded to the 16-pixel grid and, when you set no
output size, sets the output size; the padding is cropped from the result.
With several references each is fitted under 1 MP and centre-cropped, and the
output takes the first one's size.

### Prompting

The caption controls composition; reference images anchor content but do not
lock structure.

- Write full scene captions covering framing, subject, background and lighting.
  A short prompt gives generic output, and a one-line edit instruction ("make it
  blue") reframes the shot. The [prompt upsampler](#7-prompt-upsampler) expands
  a short request into a full caption.
- For edits, describe the target image as a complete caption of the reference
  with the edit folded in.
- For layout, append a JSON list of elements to the caption. Each box is
  `[top, left, bottom, right]` on a 0 to 1000 grid, and the caption tags each
  element as `<id>`. To edit by box, refer to the n-th reference as
  `<ref_image_n>` and give each element `from`, `src_bbox` and `tgt_bbox`; a
  `null` target removes it. See BFL's
  [bounding box guide](https://docs.bfl.ai/flux_3/flux3_image_bounding_boxes)
  for the full format.

## 3. Checkpoints and precision

| Repository | `--model-variant` | Sampling |
| --- | --- | --- |
| `black-forest-labs/flux-3-image-klein-base` | Omit / `fp8r` / `nvfp4` | Base CFG; defaults from the selected configuration |
| `black-forest-labs/flux-3-image-klein` | Omit / `fp8r` / `nvfp4` | Step-distilled flow map; no CFG |

Both repositories use `flux-3-image-klein` in configuration and safetensors
metadata. This shared identifier does not determine the sampling mode. The
release repositories do not need the pre-release `sd-*` variant names.

The distilled model is a step-distilled flow-map DiT: each step jumps straight
to the next time of the schedule, so it samples without CFG. SGLang tells a
flow map from its weights. It rejects `--enable-cfg-parallel`, a request
`guidance_scale` other than 1, and a non-empty `negative_prompt`. Tensor
parallelism, Ulysses SP and breakable CUDA graphs work as for the base model.

The Base model defaults to CFG 3.5 against an empty negative prompt. Both
variants reuse the package's text encoder and VAE. Keep the checkpoint revision,
precision, attention backend, guidance and sampling parameters fixed when
comparing outputs.
Explicit `guidance_scale=1` uses scale 1, not the package default;
`true_cfg_scale`, when provided, takes precedence over `guidance_scale`.

### Quantized checkpoints

Each repo ships quantized DiTs under `variants/`: select one with
`--model-variant fp8r` or `--model-variant nvfp4` and keep `--model-path`
pointing at the repo root. Text encoder and VAE precision follow their own
package settings; quantizing the DiT does not quantize them. The quantization
comes from the variant's `hf_quant_config.json`, so do not also pass
`--quantization`.

```bash
sglang serve \
  --model-path black-forest-labs/flux-3-image-klein-base \
  --model-variant fp8r \
  --host 0.0.0.0 \
  --port 30010
```

**FP8R** stores rowwise-FP8 weights. On Blackwell (SM10x) the FP8 projections
run FlashInfer PrimsTS GEMMs with fused QK-norm/RoPE and SwiGLU epilogues
(FlashInfer >= 0.7.1); set `SGLANG_ENABLE_FLUX3_PRIMS_TS_AUTOTUNE=1` to
autotune them during warmup. Other GPUs use the generic rowwise-FP8 GEMM.

**NVFP4** stores FP4 (E2M1) weights with FP8 block scales and needs a Blackwell
GPU and a compatible GEMM implementation. SM10x and consumer SM120 use
different kernel paths; a B200 recipe does not establish RTX 5090 performance.
Both quantization modes can change output quality and need workload-specific
comparison against BF16.

These flags remain separately configurable; see the validation boundaries below
before combining quantization with other optimizations.

### LoRA adapters

Use the standard LoRA API with native DiT keys, optionally prefixed by `dit.` or
`transformer.`. BF16 dynamic and merged adapters have a synthetic rank-4 generation
check, including repeatability and exact restoration after unloading; this does
not validate every community adapter. For quantized deployments requiring LoRA,
pass `--lora-path` at startup to select the generic quantized layout. Attaching
LoRA after PrimsTS weight preparation is rejected before replacing any layers.
Quantized LoRA combinations still need separate end-to-end validation.

## 4. Multi-GPU and memory

FLUX.3 shares its DiT with FLUX.3 Action, so the image pipeline supports
tensor parallelism, Ulysses sequence parallelism, CFG parallelism, layerwise
offload and breakable CUDA graphs. The picker covers RTX 5090, RTX PRO 6000,
H100/H200, and B200/B300/GB200/GB300 with one to eight GPUs on one node.
TP times Ulysses degree must equal the GPU count and evenly partition the
24 attention heads. Ring is not admitted by this image recipe.

For example, two-GPU TP uses `--num-gpus 2 --tp-size 2`; two-GPU Ulysses
uses `--num-gpus 2 --tp-size 1 --ulysses-degree 2`. Base also permits CFG
parallelism; distilled checkpoints have no CFG branch to parallelize.
More GPUs do not guarantee lower latency, particularly over PCIe.

<Note>
Under sequence parallelism every rank holds the full DiT, so combining it with
layerwise offload streams the whole model to each GPU every step. TP also
partitions weights, so it is the lower-memory starting point when offloading
on several GPUs. Preserve headroom for the longest prompt, references and output
size you intend to serve; a successful small request is not a capacity guarantee.
</Note>

For FP32 VAE attention above 1 MP, `quality="lossless"` can use dense-layout,
full-context memory-efficient SDPA; `quality="exact"` retains the original path.
It does not prune attention or reduce precision, but reduction order can change
pixels. In a Base BF16 test with a 2048x2048 reference and 1024x1024 output,
GB300 peak allocated memory fell from 54.3 to 29.5 GiB; latency was effectively
unchanged. GB300/H200 image SSIM was 0.9951-0.9992 against the exact path across
four samples. This is allocated memory, not total device usage: a 32 GB card
still needs offload or quantization headroom for that reference size.

## 5. Cache-DiT

[Cache-DiT](/docs/sglang-diffusion/cache_dit) skips the 28 single-stream blocks
on steps where their input barely changes and reuses the cached residual. Turn
it on for every request with `SGLANG_CACHE_DIT_ENABLED=true` on `sglang serve`,
or per request with `"enable_cache_dit": true`. A request can also tune it, for
example `"cache_dit_params": {"residual_diff_threshold": 0.12, "enable_taylorseer": true}`.

For image editing, send `-F "enable_cache_dit=true"` and a JSON-encoded
`cache_dit_params` form field; generation uses JSON request fields.

Cache-DiT is approximate and may change composition; measure quality and latency
on representative prompts rather than treating its threshold as a fidelity guarantee.
It combines with FP8R and NVFP4, tensor parallelism, CFG parallelism and
layerwise offload. It is skipped, with a warning, under Ulysses sequence
parallelism and breakable CUDA graphs, and for a request whose CFG pair has to
run as two forwards to fit in device memory.

## 6. Request examples

<Tabs>
<Tab title="Text to image">

```bash
curl -sS -X POST http://localhost:30010/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A red fox standing in fresh snow", "size": "1024x1024", "seed": 0, "response_format": "b64_json"}' \
  -o flux3_t2i.json
```

</Tab>
<Tab title="Image to image">

```bash
curl -sS -X POST http://localhost:30010/v1/images/edits \
  -F "image=@/data/flux3/reference.png" \
  -F "prompt=Render the reference as a winter wildlife photograph" \
  -F "seed=0" \
  -F "response_format=b64_json" \
  -o flux3_i2i.json
```

</Tab>
<Tab title="Offline generation">

```bash
sglang generate \
  --model-path black-forest-labs/flux-3-image-klein-base \
  --height 1024 --width 1024 \
  --num-inference-steps 50 \
  --guidance-scale 3.5 \
  --prompt "A red fox standing in fresh snow"
```

Pass `--image-path` (one or more images) to edit references instead.

</Tab>
</Tabs>

## 7. Prompt upsampler

[black-forest-labs/flux-3-image-klein-pup](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)
is a Qwen3.5-9B vision-language fine-tune that expands a short prompt into the detailed
caption FLUX.3 works best with. The BFL reference code runs it in front of the image model
by default. It is a standard `Qwen3_5ForConditionalGeneration` checkpoint, so it is served
as a regular LLM, separately from the image pipeline. The system prompt (`system_prompts.json`
in the repository) selects the mode; `t2i` is the text-to-image upsampler.

Download the package and start the upsampler on its own port:

```bash
PUP_PATH="<local-path>"
hf download black-forest-labs/flux-3-image-klein-pup --local-dir ${PUP_PATH}

CUDA_VISIBLE_DEVICES=1 sglang serve ${PUP_PATH} \
      --model-type llm \
      --served-model-name flux-3-image-klein-pup \
      --host 0.0.0.0 --port 8001 \
      --context-length 16384 \
      --mem-fraction-static 0.90 \
      --reasoning-parser qwen3
```

Upsample a text prompt through the OpenAI-compatible chat endpoint. The `t2i` system
prompt is read from the downloaded package; thinking is disabled and decoding follows the
model card (greedy, presence penalty 1.5):

```bash
jq -n --slurpfile p ${PUP_PATH}/system_prompts.json '{
    model: "flux-3-image-klein-pup",
    messages: [
      {role: "system", content: $p[0].t2i},
      {role: "user",   content: "A man walking with his dog, afternoon sun"}
    ],
    max_tokens: 3584,
    temperature: 0,
    presence_penalty: 1.5,
    chat_template_kwargs: {enable_thinking: false}
  }' | curl -s http://0.0.0.0:8001/v1/chat/completions \
         -H 'Content-Type: application/json' -d @- \
     | jq -r '.choices[0].message.content'
```

You can then pass the returned caption as the `--prompt` of a `t2i` request to the Flux3 pipeline.

## 8. Dynamic batching

Enable a small batching window to collect concurrent requests:

```bash Serve concurrent requests
sglang serve --model-path black-forest-labs/flux-3-image-klein-base \
  --batching-max-size 2 --batching-delay-ms 5
```

The existing scheduler groups requests with matching sampling/runtime settings.
Each keeps its own prompt, seed and reference images. Text encoding groups unique
prompts by token-length bucket; DiT attention remains isolated per packed sample,
including each CFG branch. VAE decoding groups outputs by canvas size before
restoring request order and each reference crop.

Larger batches need more activation memory and may increase individual request
latency. Start with batch 2, compare throughput and tail latency under your actual
QPS, and keep batch 1 for low-latency or low-VRAM workloads. Batching does not imply
bitwise equality with separate forwards: GEMM and attention reduction choices can
change with shape. Check output fidelity with fixed prompts and seeds.

### Pre-release batching measurements

These measurements used pre-release weights, not the release repositories
above. They do not establish release-checkpoint performance.

The following are warmed HTTP measurements on one GB300 (about 277 GiB device
memory), Base BF16, Torch SDPA, eager execution and all components resident.
Each row compares two different prompt/seed requests served serially with the
same two requests served together. Times include local HTTP transfer and JPEG
output; medians cover three rounds after warming both shapes and reference
encoding. Software: PyTorch 2.14.1+cu130, Transformers 5.19.0. Checkpoint revision:
`5de4fe32f17aee3dbc34375a94935f09206dba2c`.

| Workload | Serial pair | Batch-2 pair | Throughput ratio |
| --- | --- | --- | --- |
| T2I, 512x512, 8 steps | 0.649 s | 0.324 s | 2.00x |
| I2I, 512x512, 8 steps | 0.737 s | 0.546 s | 1.35x |
| T2I, 1024x1024, 50 steps | 6.248 s | 6.050 s | 1.03x |
| I2I, 1024x1024, 50 steps | 13.171 s | 12.996 s | 1.01x |

Eight steps exercise batching, not the recommended Base image-quality setting.
The standard 50-step workload shows little batching benefit;
FlashAttention gives a similarly small gain in this
setup. Batch 1 remains the default. Do not extrapolate the small-request speedup
to full-resolution generation or other GPUs.

For these 50-step requests, batch-1 versus batch-2 JPEG outputs had SSIM
0.9920-0.9977. A separate four-image PNG comparison at 512x512/8 steps had SSIM
0.9885-0.9955; JPEG can amplify small rounding differences. These are observed
samples, not a universal quality guarantee. Identical single requests before
and after batching matched exactly. Preserve additional memory headroom for
batching rather than assuming a batch-1 recipe fits on a 32 GB card at batch 2.

## 9. Validation boundaries

The release BF16 and FP8R checkpoints have single-GB300/H200 generation checks;
NVFP4 has GB300 checks. Base BF16 also has an RTX 5090 check. Each uses eager
execution, all components resident, batch 1, 1024x1024 T2I and I2I, two
prompts/seeds and repeated requests. Base uses 50 steps and distilled uses 16.
Repeated requests matched pixel-for-pixel within each tested configuration.
This does not establish official-reference parity or validate every combination
of hardware, parallelism, batching and optional features in the picker.

### Single-GPU measurements

Warmed local HTTP medians, including PNG encoding and response transfer, from
two rounds per prompt. PyTorch 2.14.1+cu130, Transformers 5.19.0, FP32 VAE,
`quality="exact"`; FlashAttention on GB300/H200 and Torch SDPA on RTX 5090.
The Base checkpoint revision is `d8acde427a698bdb98b5c1c941c60fa4c97e9f28`.
SSIM compares each quantized result with BF16 on the same GPU, prompt, seed and
reference, and reports the lowest value across two samples, not a quality bound.

| GPU | Base DiT | T2I | I2I | Minimum SSIM, T2I / I2I |
| --- | --- | --- | --- | --- |
| GB300, 277 GiB | BF16 | 3.21 s | 6.31 s | Reference |
| GB300, 277 GiB | FP8R | 2.18 s | 4.49 s | 0.958 / 0.981 |
| GB300, 277 GiB | NVFP4 | 2.14 s | 4.31 s | 0.919 / 0.958 |
| H200, 140 GiB | BF16 | 7.09 s | 14.52 s | Reference |
| H200, 140 GiB | FP8R | 5.15 s | 11.24 s | 0.955 / 0.984 |
| RTX 5090, 32 GB | BF16 | 20.39 s | 43.4 s | Reference |

For the distilled checkpoint, quantization's T2I benefit was small on GB300;
do not assume Base's speedup transfers to a 16-step workload. Cache-DiT at
threshold 0.12 reduced Base I2I from 6.31 to 3.08 s on GB300, with sample SSIM
0.9786-0.9975. It remains opt-in and approximate. Breakable CUDA graphs did not
provide a clear latency win in the tested GB300 workload; eager stays recommended.

The native DiT's packed CFG forward has focused comparison tests against the
shared FLUX 3 transformer. Such component tests do not replace full-checkpoint
E2E or establish parity with every upstream release. Keep unverified hardware,
topologies and optional optimizations separate from measured deployment recipes.

The opt-in full-checkpoint HTTP regression covers concurrent prompts, image
editing, multiple outputs and a repeated single request:

```bash
SGLANG_TEST_FLUX3_IMAGE=1 pytest -q \
  python/sglang/multimodal_gen/test/server/test_server_flux3_image.py
```

Set `SGLANG_TEST_FLUX3_IMAGE_MODEL` to a local package path when using an
offline checkpoint. This test needs checkpoint access; it does not substitute
for an official-reference golden-image comparison.

## 10. ComfyUI

import { ComfyUISupport } from '/src/snippets/diffusion/comfyui-support.jsx';

<ComfyUISupport model="image" note="FLUX.3 has no integrated KSampler executor. Use server mode; the plugin-specific FLUX.3 workflow has not been validated." />
