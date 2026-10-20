---
title: Kandinsky6 SR
description: "Deploy Kandinsky 6 video super-resolution with x2, x4 or x2.25 upscaling, source audio and explicit decode speed, memory and numerical tradeoffs."
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/Kandinsky/kandinsky6-sr.jsx';

<DiffusionModelTags tags={["video", "video-to-video", "super-resolution", "source audio preserved"]} />

## 1. Quick start

Use Linux with NVIDIA CUDA and a checkout containing this implementation; it is
not yet in a released package or unmodified release image. Authenticate an account
with checkpoint access using `hf auth login`, or supply `HF_TOKEN` to the server.
From the repo root:

```bash Install
uv pip install -e "python[diffusion]" --prerelease=allow
```

The picker uses the distilled two-step checkpoint. Four-GB300 whole-tile decode
is the output-preserving choice measured on multi-tile clips; one GPU suffices
for a functional check when it has enough memory. The picker also includes
explicitly **extrapolated** Hopper and RTX 5090 starting points, not measured
speed or memory guarantees. HTTP verification is scoped separately from offline
measurements. See the [installation guide](/docs/sglang-diffusion/installation)
for platform setup.

<Deployment config={config} />

The request uploads a local video and returns a job `id`. Poll
`GET /v1/videos/{id}`, then download `GET /v1/videos/{id}/content` when completed.
Use the shared [polling and download example](/cookbook/diffusion/Kandinsky/Kandinsky6#5-download-a-completed-video).
For JSON requests, `video_path` refers to a file on the **server**, not the client.

## 2. Model capabilities

Kandinsky 6 VSR upscales a source video without a prompt. It encodes the source
with a causal KVAE, denoises overlapping tiles and blends their decoded outputs.
Unlike the [generation model](/cookbook/diffusion/Kandinsky/Kandinsky6), it retains
the source soundtrack rather than generating audio. Audio is converted to mono
44.1 kHz and trimmed to the processed video duration.

Use `kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers` for x2, x4 or
x2.25 upscaling of up to 121 frames. Its checkpoint-selected PiFlow scheduler
uses two DiT calls per tile; changing `num_inference_steps` does not increase
quality or the number of evaluations. The scheduler and transformer must come
from the same Diffusers checkpoint; sampler settings are read from `scheduler/`.
The native path uses dense attention,
not the checkpoint's requested NABLA sparse attention, so this is not a claim
of numerical parity with the sparse reference.

<Warning>
The tested non-distilled `Kandinsky-6.0-VSR-5s-Diffusers` revision
`d6ddfabc62b3920b6da6487e4f444d37c591d5f4` declares a flow-Euler scheduler but
contains a 640-wide output head instead of the required 64. It is not supported
as-is. Do not change the scheduler to bypass this error; use the distilled model.
</Warning>

## 3. Offline usage

No prompt is required. A single-GPU example:

```bash Generate
sglang generate \
  --model-path kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers \
  --video-path /path/to/input.mp4 \
  --sr-resolution-scale 2 \
  --num-inference-steps 2 \
  --seed 42 --save-output --output-path ./sr_outputs
```

<Accordion title="Python API">
```python Generate
from sglang.multimodal_gen import DiffGenerator

generator = DiffGenerator.from_pretrained(
    model_path="kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers",
    num_gpus=1,
)
try:
    result = generator.generate(sampling_params_kwargs={
        "video_path": "/path/to/input.mp4",
        "sr_resolution_scale": 2,
        "num_inference_steps": 2,
        "seed": 42,
        "output_path": "sr_outputs",
        "save_output": True,
    })
    print(result.output_file_path)
finally:
    generator.shutdown()
```
</Accordion>

### Request parameters

- `sr_resolution_scale`: `2`, `4` or `2.25` (default). The x2.25 path first
  enlarges pixels x1.125, then processes x2 tiles.
- `sr_tiles_batch_size`: default `1`. More tiles per DiT call need more activation
  memory. Each chunk is seeded by its first tile index, so changing this value
  changes the noise and output. It is **not a lossless batching optimization**.
- `sr_tile_min_overlap`: default `0.2`. Changing overlap changes tile geometry
  and blending, not just performance.
- `sr_target_resolution`: optional `hd`, `fullhd`, `2k` or `WxH`; only downscales
  the generated result. `sr_target_resize_mode` is `fit` (default) or `exact`.
- `seed`: default `42`. Compare identical seeds, tile batching, source frames and
  attention policy when evaluating a decode optimization.

The input must fit at least one tile. A tested landscape source is 384 x 256.
The default filename is `<input name>_sr<scale>x_<timestamp>.mp4`; override it
with `--output-file-name`.

## 4. Decode speed, memory and numerics

These are **VAE decode choices**, independent of DiT TP/Ulysses/Ring:

<table>
  <thead><tr><th>Decode</th><th>Hot E2E</th><th>Peak reserved</th><th>Output versus serial</th></tr></thead>
  <tbody>
    <tr><td>Serial</td><td>45.26 s</td><td>47.55 GiB</td><td>Reference</td></tr>
    <tr><td>Whole-tile parallel</td><td>21.73 s</td><td>47.55 GiB</td><td>RGB and audio exact in this test</td></tr>
    <tr><td>Spatial shard</td><td>18.01 s</td><td>24.37 GiB</td><td>RGB differs; audio exact</td></tr>
  </tbody>
</table>

Setup: four GB300 GPUs (277.5 GiB each, NV18 links), host RAM 955.7 GiB with an
860 GiB cgroup limit; BF16/FA, Ulysses4, resident weights, no compile or startup
warmup. PyTorch 2.14.1+cu130, Transformers 5.17.0, Diffusers 0.37.0. Distilled
checkpoint revision `3e3e10bcd23659eeeb23cac740258ee4fc14aa7d`; seed 42, tile
batch 1, two steps per tile. A generated kitchen clip, 768 x 512 x 121 frames at
24 fps, was upscaled x2 to 1536 x 1024. Times are medians of three hot requests
after one initial request, including MP4 saving and excluding model loading.
Memory is PyTorch request peak reserved on the output rank, not loading, NVML or
the maximum across ranks. Host memory is not claimed unchanged.

**Whole-tile parallel** assigns complete existing tiles to ranks and preserves
their temporal caches and blend order. It matched serial RGB and raw audio
exactly on the tested workloads **within the same topology**. Select
`--vae-config.use-parallel-tiling true`. A single tile falls back to serial.
Communication can outweigh savings for small clips or slow interconnects.

**Spatial shard is not bitwise lossless.** It splits each tile's height, using
halo exchanges and the original temporal cache. Changed convolution shapes
change rounding. In the experiment above, all-frame RGB PSNR was **57.51 dB**,
minimum SSIM over the first/middle/last frames **0.99921**, mean absolute
difference **0.1132/255**, and maximum difference **4/255**. Audio was exact.
These measurements are not a guarantee for other videos or perceptual-quality
certification. Select it only after validating the output for your workload:

```bash Spatial decode flags
--vae-config.use-parallel-decode true --vae-config.parallel-decode-mode spatial_shard
```

Both modes are opt-in runtime features. The picker selects whole-tile parallel
for its multi-tile recipe; it does not change runtime defaults. If both flags
are enabled, spatial sharding takes precedence. One GPU or a tile too short to
split retains replicated decoding. Neither mode changes the tile geometry.

### RTX PRO 6000 over PCIe

| GPUs / placement | Decode | Hot E2E | Sampled GPU peak |
|---|---|---|---|
| 1 / resident | Serial | 134.09 s | 48.91 GiB |
| 2 / resident, Ulysses2 | Serial | 122.11 s | 48.81 GiB |
| 2 / resident, Ulysses2 | Whole-tile | 83.04 s | 48.81 GiB |
| 2 / resident, Ulysses2 | Spatial | 73.62 s | 37.45 GiB |
| 1 / all-layerwise offload | Serial | 133.60 s | 37.43 GiB |

Setup: two RTX PRO 6000 Blackwell Server Edition GPUs (96GB nominal), PCIe/SYS
across CPU sockets, **no NVLink**; 499.2 GiB host RAM, no cgroup cap. Driver
595.84, the same PyTorch/Transformers/Diffusers versions and checkpoint as above,
but **SDPA**, not FA. A separate generated 768 x 512 x 121-frame clip at 24 fps
was upscaled x2 with seed 42, tile batch 1 and two steps per tile. Each row uses
one initial request and three hot requests, no compile/startup warmup. E2E
includes MP4 saving and raw-frame return, excludes model loading. GPU memory is
the maximum **per-GPU NVML sample across ranks**, sampled every 0.5 seconds during
requests, not the sum or PyTorch reserved memory; brief peaks may be missed.
These are within-machine comparisons, not GB300-versus-RTX speed ratios.

Ulysses2 alone reduced latency by only 8.9%. Whole-tile decode reduced it by
38.1% versus one GPU, with **exact RGB/audio** in this test. Spatial reached
45.1% lower latency, but RGB differed: PSNR **57.54 dB**, minimum sampled SSIM
**0.99919**, maximum difference **4/255**; audio remained exact. All five
configurations reproduced their own outputs across repeated requests.
Layerwise offload reduced memory with essentially unchanged latency here; this
does not make offload free on other workloads or slower host links.

A separate **memory-budget probe**, still on RTX PRO 6000, completed the full
clip twice with all-layerwise offload and a **30 GiB PyTorch allocator cap**.
Sampled whole-GPU usage peaked at **31.03 GiB**, and RGB/audio matched the
resident baseline exactly. The cap excludes CUDA allocations outside PyTorch;
it is a test setting, not a deployment recommendation. This supports trying the
32GB recipe, but is **not a 5090 speed, memory-fit or quality measurement**.

The complete HTTP regression also passed on one-GPU resident serial, Ulysses2
resident whole-tile and one-GPU layerwise serial with the allocator cap:
six requests each covering JSON, uploads, x2/x4/x2.25, tile batch 2,
audio/silent inputs and repeated-request equality. Arbitrary videos still need
their own validation.

### Small workloads do not necessarily scale

On GB300, a 384 x 256 x 33-frame source upscaled x2 (one tile, serial decode,
otherwise the same BF16/FA/seed/step policy) took **1.77 s on one GPU,
1.72 s on Ulysses2 and 1.74 s on Ulysses4**. These are three-hot-request medians
including saving. The roughly 2% difference is not evidence of a useful speedup;
prefer one GPU for this workload. More tiles, not just more GPUs, create work
that whole-tile decoding can distribute.

## 5. Memory and parallelism

### Hardware and GPU counts

| Hardware | Starting point | Evidence boundary |
|---|---|---|
| GB300 | Resident, FA; Ulysses4 + whole-tile for multi-tile clips | Measured offline; smaller 1/2/4-GPU workloads also tested |
| RTX PRO 6000 96GB | Resident, SDPA; compare 1 GPU with Ulysses2 | PCIe measurements, not a substitute for 5090 memory validation |
| H100 80GB / H200 141GB | Resident, FA; start with 1 GPU, then Ulysses2/4 | Extrapolated from native CUDA support; no Hopper SR measurement |
| B200 | Resident, FA; start with 1 GPU, then Ulysses2/4 | Extrapolated SR recipe; generation-model tests do not validate SR |
| RTX 5090 32GB | SDPA, layerwise offload, tile batch 1; short 384 x 256 input first | Extrapolated from the RTX memory-budget probe; no 5090 execution claim |

Only compare GPU counts on the **same input, scale, tile batching and backend**.
For small single-tile clips, one GPU avoids communication overhead. For larger
multi-tile clips, try Ulysses2/4 with whole-tile decode; extra GPUs do not divide
replicated KVAE/latent-upscaler weights. Spatial decode can reduce activation
memory, but changes rounding. Offload needs sufficient host RAM as well as VRAM.
Do not extrapolate NVLink scaling to a PCIe system, especially across CPU sockets.

Eight-GPU FA configurations are selectable as **Ulysses4 x Ring2**, not Ulysses8:
28 heads are not divisible by 8. This is an unmeasured topology, not a speedup
recommendation. The current RTX SDPA path has no Ring support, so the picker
rejects that combination. Start with fewer GPUs before trying an eight-GPU run.

### Placement and topology

DiT Ulysses, TP, Ring and TP x SP are supported. For example, use
`--num-gpus 4 --tp-size 1 --ulysses-degree 4` for Ulysses4, or
`--num-gpus 4 --tp-size 2 --ulysses-degree 2` for TP2 x Ulysses2.
TP times Ulysses must divide the checkpoint's 28 attention heads.
**TP and Ring can change output numerics independently of decode mode**;
whole-tile decode being exact does not make a TP/Ring topology change lossless.

The decode group includes TP/SP ranks within each DP replica, not different
requests. KVAE encoding, latent upscaling and tile scheduling remain replicated.
For short two-step clips, extra GPUs can add communication without a speedup.

Keep components resident when memory permits. For less GPU weight memory, use
`--layerwise-offload-components all`, or select `vae latent_upscaler` instead.
Temporal caches and non-block weights remain on GPU. Offload is compatible with
both decode modes, but consumes host RAM and adds transfers for tiles and temporal
segments. The assembled output also occupies host memory until saved.

The **Server** tab also offers whole-component offload with
`--cpu-offload-components`, selectable offload components, and DiT prefetch and
resident-layer counts when streaming the DiT. Increasing those counts consumes
additional VRAM. Partial offload recipes, non-default layer counts and
FSDP (`--use-fsdp-inference true`) are **unverified options**, not measured
improvements. FSDP requires multiple GPUs; SDPA cannot be combined with Ring.
The default backend is FA on datacenter Blackwell/Hopper and SDPA on RTX.
Hardware starting points do not automatically adapt to every uploaded video's
memory demand. Request scale and tile batching do not require restarting the server.

## 6. Limits

- **Input:** at most 121 frames. Sources near 24 fps retain their frames; faster
  videos are subsampled, slower videos retain their native rate. This can change
  the source timing; SR is not a bit-preserving video/audio transcode.
- **Attention:** this native path is dense. Do not compare it with a NABLA sparse
  reference and attribute every difference to parallel decoding. The latent
  upscaler uses SDPA motion attention when `natten` is unavailable.
- **Deployment:** monolithic only. Disaggregated roles, `torch.compile`, LoRA,
  direct latent inputs and KVAE-bridge chaining are not supported.
- **Loading:** component config and weight mismatches fail explicitly. For a
  renamed local checkpoint, pass
  `--model-id Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers` to retain SR routing.

See the [compatibility matrix](/docs/sglang-diffusion/compatibility_matrix) for
other model families. The opt-in server regression is
`python/sglang/multimodal_gen/test/server/test_server_kandinsky6_sr.py`; it covers
uploads, JSON, all three scales, source audio and repeated requests, not official
pipeline parity.
