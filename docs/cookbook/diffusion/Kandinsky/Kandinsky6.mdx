---
title: Kandinsky6
description: "Deploy Kandinsky 6.0 for joint text- and image-conditioned video and audio generation."
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/Kandinsky/kandinsky6.jsx';

<DiffusionModelTags tags={["video + audio", "text-to-video", "image-conditioning", "5 seconds"]} />

## 1. Quick start

Use Linux with NVIDIA CUDA. This model is not yet in a released package;
use a checkout containing this implementation, not an unmodified release image.
Your Hugging Face account must have access to the checkpoint; authenticate with
`hf auth login` or supply `HF_TOKEN` in the server environment. Install from the repo root:

```bash
uv pip install -e "python[diffusion]" --prerelease=allow
```

Start with Pro-distill and resident weights on B200, GB300, or RTX PRO 6000 (96 GB).
The picker separates
verified lightweight HTTP requests from full-size offline generation and
unverified custom topologies. See the [installation guide](/docs/sglang-diffusion/installation)
for platform setup.

<Deployment config={config} />

The request returns a video job ID, not the video bytes. Poll
`GET /v1/videos/{id}` until `status` is `completed`, then download
`GET /v1/videos/{id}/content` for the MP4 containing both video and audio.
Treat `failed` as an error; delete the job with `DELETE /v1/videos/{id}` when finished.
See [Download a completed video](#5-download-a-completed-video) for a complete polling example.

## 2. Model capabilities

Kandinsky 6.0 generates video and audio jointly from a text prompt, optionally
conditioned on a reference image. One DiT predicts both modalities at each
denoising step. A Reason1 encoder derived from Qwen2.5-VL supplies token features,
and CLIP supplies pooled text features. Both request modes use the same server;
image conditioning does not require switching checkpoints.

Choose Pro-distill for the 10-step path, or Pro for 50-step classifier-free guidance.
The recipes cover 768 x 512, 121-frame clips at 24 fps. They use native SGLang
components, without calling a third-party model pipeline. Validation compares
against the initial SGLang implementation; it does not establish parity with
the unpublished Kandinsky6 Diffusers implementation.

| Checkpoint | Scheduler | Default steps | Default guidance |
| --- | --- | ---: | ---: |
| `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers` | PiFlow | 10 | 1.0 |
| `kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers` | Flow-match Euler | 50 | 5.0 |

Pro-distill only accepts guidance 1.0 and does not encode a negative prompt.
The scheduler is loaded from the checkpoint. For a local directory with an
unrelated name, specify `--model-id Kandinsky-6.0-Pro-distill-5s-Diffusers` to
select its sampling defaults, or explicitly pass 10 steps and guidance 1.0.

## 3. Offline generation

The default request produces a muxed video and audio file:

```bash
sglang generate \
  --model-path kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers \
  --performance-mode speed --attention-backend fa --warmup-mode off \
  --prompt "A woman chops vegetables in a sunlit kitchen, with soft jazz playing." \
  --seed 42 --save-output
```

Add `--image-path /path/to/reference.png` for image conditioning.
On RTX PRO 6000, use `--attention-backend torch_sdpa` in the CLI or
`attention_backend="torch_sdpa"` in Python.
For two B200 GPUs, add `--num-gpus 2 --tp-size 1 --ulysses-degree 2 --ring-degree 1`.
To use Pro, change the model path; its defaults select 50 steps and guidance 5.0.

### Python API

```python
from sglang.multimodal_gen import DiffGenerator

generator = DiffGenerator.from_pretrained(
    model_path="kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers",
    num_gpus=1,
    performance_mode="speed",
    attention_backend="fa",
    warmup_mode="off",
)
try:
    result = generator.generate(sampling_params_kwargs={
        "prompt": "A woman chops vegetables in a sunlit kitchen, with soft jazz playing.",
        "seed": 42,
        "save_output": True,
        "output_path": "kandinsky6_samples",
        "return_file_paths_only": False,
        "return_frames": True,
    })
    print(result.output_file_path)
finally:
    generator.shutdown()
```

`result.frames` contains RGB frames and `result.audio` contains the decoded waveform.
Add `image_path` to the sampling parameters for a reference image. Omit the two
return-data flags when only the output file is needed.

## 4. Parallelism and memory

- **Ulysses**: full-size Pro-distill generation on two B200 or four GB300 GPUs
  matches single-GPU output on the same hardware, including every decoded RGB
  frame and the audio waveform.
  It shards the video stream; audio and text streams remain replicated. Video K/V
  is gathered for audio-to-video attention.
- **CFG parallel**: Pro can distribute its conditional and unconditional branches
  with `--num-gpus 2 --ulysses-degree 1 --enable-cfg-parallel`.
  This is a different topology from Ulysses. It does not accelerate Pro-distill,
  which has no guidance branch.
- **TP and Ring**: implemented, but full-model outputs differ from the single-GPU
  baseline. Do not treat them as verified bitwise-equivalent modes.
  Ring2, Ring4, and TP2 x Ring2 complete full-size Pro-distill generation on
  GB300 and repeat exactly within each topology. Ring2 serving also covers
  image conditioning, Cache-DiT, and dynamic/merged LoRA restoration. Use
  `--num-gpus 2 --ulysses-degree 1 --ring-degree 2` to select Ring2; validate
  both video and audio quality before deployment.
- **TP x SP**: four-GB300 TP2 x Ulysses2 full generation is tested. It reduces
  weight memory but was slower than Ulysses4 and produced different video and
  audio. Prefer Ulysses when memory permits; validate quality before deploying TP.
- **FSDP + Ulysses**: add `--use-fsdp-inference true` to shard DiT weights
  without sharding linear reductions. Four-GB300 Ulysses4 generation matches
  resident Ulysses4 in every decoded RGB frame and the audio waveform. This is
  a lower-memory alternative with additional weight-gather latency, not a
  faster default. Request-level Cache-DiT switching is also tested with FSDP.
- **Encoder parallelism**: `--encoder-parallel fold` shards Reason1 and CLIP
  across the DiT replica, including a Ulysses-only replica. Four-GPU generation
  is tested, but folding changes reductions and the generated video and audio.
  Keep `auto` for the baseline recipes and validate quality before forcing
  folding. See [Encoder parallelism](/docs/sglang-diffusion/encoder_parallel).
- **Offload**: keep components resident when memory allows. Use
  `--cpu-offload-components transformer text_encoder vae` for whole-component
  offload, or `--layerwise-offload-components all` for layerwise offload.
  Layerwise offload includes both DiT text towers and the audio VAE's encoder,
  decoder, and vocoder blocks. Use component-specific residency settings to
  keep selected components resident.
  Both require host RAM and add transfers; lower device memory does not imply
  lower total memory or lower latency.

The picker's **Server** tab exposes attention, weight placement and offload
components. With DiT layerwise offload, you can also tune prefetch depth and
resident layers using `--dit-offload-prefetch-size` and
`--dit-layerwise-resident-layers`. Increasing either uses more VRAM; these
non-default settings are unverified tuning choices, not faster recommendations.
Component offload moves whole components; layerwise offload streams supported
blocks. They are alternatives to FSDP in this picker, not additive presets.

### Deployment limits

Use monolithic serving. Encoder/denoiser/decoder disaggregation is not supported.
Breakable CUDA graphs are also not enabled for Kandinsky6: the server disables
`--enable-breakable-cuda-graph`. Its variable-length text path needs additional
handling to preserve the unpadded computation; same-shape graph experiments do
not establish support for arbitrary requests. Keep eager execution for the
recipes on this page.

### Choosing a topology

Pro-distill, batch 1, 768 x 512, 121 frames, 10 steps, BF16/FA, no offload or
compilation, on GB300 with 277.5 GiB per GPU:

| Topology | Hot E2E (s) | Output-rank peak reserved (GiB) | Matches single-GPU RGB/audio |
| --- | ---: | ---: | --- |
| Single GPU | 33.95 | 80.36 | Reference |
| Ulysses4 | 11.33 | 80.91 | Exact |
| Ulysses4 + FSDP | 12.13 | 42.54 | Exact |
| TP2 x Ulysses2 | 13.99 | 50.97 | No |
| TP4 | 17.72 | 35.91 | No |

Times are medians of three hot requests after one initial request, with startup
warmup disabled and default conditioning caching enabled. They include MP4
saving but exclude model loading. Memory is PyTorch runtime peak reserved on
the output rank, not loading peak or the maximum across all GPUs. These results
do not establish output quality for TP; TP also changes the text encoder's
numerical path.

For this workload, FSDP reduces measured memory by 47% versus resident Ulysses4
at a 7% latency cost. Add it when weight memory is the constraint; keep ordinary
Ulysses when latency is the priority.

### RTX PRO 6000 and SageAttention3

Pro-distill also runs with resident weights on one 96 GB RTX PRO 6000 Blackwell
Server Edition. Use `--attention-backend torch_sdpa` on this SM120 device:
the current platform resolves `fa` to SDPA, so requesting `fa` does not establish
a FlashAttention benchmark.

With [SageAttention3](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention3_blackwell)
installed, `--component-attention-backends transformer=sage_attn_3` selects it
only for the DiT, keeping the encoders on SDPA. The tested upstream revision
requires SM120/121 and rejects B200 SM100. With PyTorch 2.14, its build required
`NVCC_APPEND_FLAGS=-std=c++20`; no kernel source or hardware guard was modified.

Same full-size workload and timing method as above, on one RTX PRO 6000 with
driver 595.84, PyTorch 2.14.1+cu130, Transformers 5.17.0, and Diffusers 0.37.0:

| DiT backend | Hot E2E (s) | Runtime peak reserved (GiB) |
| --- | ---: | ---: |
| SDPA | 133.91 | 80.34 |
| SageAttention3 | 105.33 | 82.05 |

Sage3 reduced latency by 21.3% but changed the outputs substantially: all-frame
PSNR 11.97 dB, mean SSIM 0.375, and raw-audio waveform SNR -2.21 dB versus SDPA.
These are paired differences, not perceptual quality scores. Each backend was
repeatable across four requests, but Sage3 is **not lossless** and is not the
default recipe. Validate both video and audio before choosing it.

### VAE decode

The video VAE defaults to tiled decoding, including multi-GPU execution. Its
decode group is separate from the DiT's TP and CFG collectives. Explicit
`--vae-config.parallel-decode-mode spatial_shard` uses whole-clip decoding with
frame-causal attention computed without a quadratic whole-video mask.

Whole-clip decoding still needs substantial activation memory. A 768 x 512,
121-frame run on two B200 GPUs completed with the transformer, both text encoders,
and audio VAE offloaded, but reached 170.4 GiB output-rank peak reserved. This is
not a lower-memory replacement for tiled decode. Tiled and whole-clip decoding
use different normalization and attention contexts and are not interchangeable
numerical baselines; keep tiled decoding for the validated default output.

The audio VAE contains both the mel decoder and vocoder. Current checkpoints
include a complete vocoder copy inside `audio_vae`, so this pipeline does not
also load the redundant standalone `vocoder` component.

CUDA matrix RoPE uses a fused kernel that preserves the eager FP32 rounding
boundaries. CPU and autograd paths retain the eager implementation. No sampling
parameter or sparse-attention approximation is introduced.
The checkpoint's 128-wide Q/K normalization uses native PyTorch FP32 fusion
during CUDA inference without changing the stored parameter dtype or offload
policy. Other head widths retain the original normalization path.

### Cache-DiT

Cache-DiT is opt-in per request through `enable_cache_dit` and `cache_dit_params`
in the Python sampling parameters or the JSON `/v1/videos` body. Set
`enable_cache_dit: false` to restore uncached execution without restarting.
See [Cache-DiT configuration](/docs/sglang-diffusion/cache_dit)
for the available knobs.

The **Request** tab offers an opt-in Cache-DiT switch for both text and image
conditioning. It uses the runtime cache defaults, not a quality-calibrated
preset. A short request may finish before cache warmup ends and gain no speedup.

The adapter caches both evolving video and audio streams together. Its dynamic
decision uses the video residual, not an independent audio-quality metric.
Skipping blocks is approximate and can change both outputs substantially;
validate video and audio on your own prompts before enabling it. It is not part
of the lossless defaults, and should not be combined with breakable CUDA graphs.

### LoRA adapters

Use adapters trained for the same Kandinsky6 checkpoint architecture. The native
loader accepts PEFT A/B weights with Diffusers or SGLang DiT layer names, including
the `transformer.` prefix. Both video and audio branches belong to `transformer`;
an adapter can change both outputs.

Add `--lora-path /path/to/adapter.safetensors` to load an adapter at startup.
To switch an existing server to dynamic LoRA without restarting:

```bash
curl --fail-with-body http://localhost:30000/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
    "lora_nickname": "kandinsky6",
    "lora_path": "/path/to/adapter.safetensors",
    "target": "transformer",
    "strength": 1.0,
    "merge_mode": "dynamic"
  }'
```

The adapter path must be readable by the server. `dynamic` applies the low-rank
update during inference; `merge` updates the base weights before inference.
These modes may differ numerically. To disable either mode and restore the base
model:

```bash
curl --fail-with-body http://localhost:30000/v1/unmerge_lora_weights \
  -H "Content-Type: application/json" -d '{"target": "transformer"}'
```

Adapter selection is server-wide, not isolated per request. Coordinate changes
with other clients. Merging offloaded weights retains CPU snapshots of the
modified layers for exact restoration; budget for this host memory.
Validation uses synthetic adapters to check loading,
strength changes, repeatability, and exact restoration of video and audio;
it does not establish the quality of a community adapter.

See the [model inventory](/docs/sglang-diffusion/compatibility_matrix) and the
[super-resolution recipe](/cookbook/diffusion/Kandinsky/Kandinsky6-SR) for the
separate video SR pipeline.

## 5. Download a completed video

Use the `id` returned by the picker's Request command. The same endpoints serve
joint generation and SR. Set the base URL to match your server port:

```python Download
import json
import shutil
import time
import urllib.request

base_url = "http://localhost:30000"
video_id = "REPLACE_WITH_RETURNED_ID"
deadline = time.monotonic() + 1800

while time.monotonic() < deadline:
    with urllib.request.urlopen(f"{base_url}/v1/videos/{video_id}", timeout=30) as response:
        job = json.load(response)
    if job["status"] == "failed":
        raise RuntimeError(job)
    if job["status"] == "completed":
        with urllib.request.urlopen(
            f"{base_url}/v1/videos/{video_id}/content", timeout=120
        ) as response, open("output.mp4", "wb") as output:
            shutil.copyfileobj(response, output)
        break
    time.sleep(2)
else:
    raise TimeoutError("Video generation did not finish within 30 minutes")
```

After downloading, remove the server-side job when no longer needed:

```bash Cleanup
curl --fail-with-body -X DELETE http://localhost:30000/v1/videos/REPLACE_WITH_RETURNED_ID
```

## 6. ComfyUI

There is no Kandinsky6 integrated DiT executor in the
[ComfyUI plugin](/docs/sglang-diffusion/comfyui). The generic server video node
has not been validated for this model's joint audio/video output. Use the HTTP
or offline recipes above until that integration is verified.
