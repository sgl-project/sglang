---
title: MiniMax-H3-Image
description: "Run native six-step MiniMax-H3-Image generation and image editing with SGLang Diffusion."
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';
import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/MiniMaxAI/minimax-h3-image.jsx";

<DiffusionModelTags tags={["text to image", "image editing", "multiple references", "6-step PDD", "BF16"]} />

## 1. Quick start

Use the [SGLang main branch](https://github.com/sgl-project/sglang/tree/main)
with the [diffusion installation guide](/docs/sglang-diffusion/installation),
then install its diffusion dependencies from the repository root:

```bash
python3 -m pip install -e 'python[diffusion]'
```

The public checkpoints are `MiniMaxAI/MiniMax-H3-Image` and its
`MiniMaxAI/MiniMax-H3` text encoder.
Run the commands from the same checkout and Python environment. The picker
saves the returned image as `generated.png`. These recipes use Linux CUDA and
a source installation; DGX Spark uses the [ARM64 environment below](#dgx-spark-128-gb),
and AMD Instinct GPUs use the [ROCm environment below](#amd-instinct-mi300x-mi325x-mi350x-mi355x).

<Deployment config={config} />

For the lowest measured latency, see the [recipe summary](#measured-text-to-image-summary):
two B300s generate a 4K image in 2.654 s at native precision or 2.065 s with
opt-in MXFP8. These are warm pipeline times; quantization can change image quality.

Server warmup runs a valid prefix of the PDD schedule; real requests always
run all six steps. With online AdaLN and enough cache slots, warmup prepares
the full grids for both declared T2I and editing tasks without extra denoising
steps. Smaller custom caches prepare only the warmup task. New shapes can
still initialize kernels. When every plan fits on the GPU, Auto omits the
pinned host cache; an explicit `--minimax-h3-adaln-host-cache-gb` overrides this.
`--warmup-mode off` skips this preparation. For CLI/API options, see the
[CLI reference](/docs/sglang-diffusion/api/cli) and
[OpenAI API guide](/docs/sglang-diffusion/api/openai_api).

## 2. Model capabilities

[MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) uses one
native checkpoint for text-to-image generation and image editing. An edit's
multipart `image` fields preserve reference order; repeat the field to supply
multiple references. The prompt refers to them as `<Picture 1>`, `<Picture 2>`,
and so on. The root checkpoint selects the image pipeline automatically;
`--model-variant` is not needed.

The DiT has about 33.1 billion parameters. Its separate Qwen3-VL encoder
checkpoint has about 33.4 billion; downloading both checkpoints and the VAE
requires about 135 GB. The pipeline loads only the encoder layers needed by
this release. Runtime memory also depends on placement, projection reuse and
the generation canvas.

Prompt text uses no chat template or system prompt and is truncated after
4096 tokens. Use a complete visual description for generation or editing.

The checkpoint has six fused PDD output heads and an explicit sigma
grid. Omit `num_inference_steps`, or set it to 6. It runs one positive branch:
omit guidance parameters and `negative_prompt`. It returns one image per
request. Audio and video inputs and outputs are not supported.

| Task | Endpoint | Input |
| --- | --- | --- |
| Text to image | `/v1/images/generations` | JSON prompt |
| Image editing | `/v1/images/edits` | Multipart prompt and ordered image files |

### Resolution and references

Recommended output sizes are:

| Aspect ratio | Landscape | Portrait |
| --- | --- | --- |
| 1:1 | 4096 × 4096 | — |
| 4:3 | 4736 × 3552 | 3552 × 4736 |
| 3:2 | 5024 × 3360 | 3360 × 5024 |
| 16:9 | 5472 × 3072 | 3072 × 5472 |
| 21:9 | 6272 × 2688 | — |

The default minimum generation area is 2048 × 2048 pixels. Smaller requests
are generated on a larger canvas with the same aspect ratio, aligned to 32
pixels, then resized to the requested output size with LANCZOS. A 1024 × 1024
output therefore has the generation cost of a 2048 × 2048 canvas.

`min_generation_area` is a request-level integer pixel area between 1024 and
16777216. Reducing it changes the generation workload and potentially image
quality; it is not a lossless acceleration of the default recipe. Set
`min_generation_area` to `null` in a pipeline configuration file to disable
canvas enlargement; output dimensions must then be multiples of 32.

Edit references are oriented using EXIF and resized separately for each
encoder. Qwen uses a 1024-pixel short edge; the VAE keeps the short edge within
1024–2048 pixels. Reference order, dimensions, generator device, model revision
and precision all matter when reproducing an edit.

## 3. Deployment options

DiT and text-encoder weights use BF16. The VAE checkpoint uses FP32; its
decoder weights are prepared in BF16 for BF16 autocast. The DiT uses dense
FlashAttention. Keep these precision and attention settings when reproducing
the release. ROCm has no FA3 build, so on AMD GPUs the DiT runs the same dense
packed attention through AITER. It is selected automatically and `fa` is
rejected there. Ring parallelism requires FA; use Ulysses or TP on AMD.

On discrete GPUs, Auto keeps all components resident when each GPU has at least
130 GiB available at startup, as tested on H200, B300, MI325X and MI350X. Otherwise it retains
component offload. GB10 uses the shared-memory policy in the Spark recipe below.
Explicit placement overrides take precedence. Resident mode requires
space for all weights and request activations; sequence parallelism still
replicates weights. TP shards the DiT projections. Ulysses and Ring split the
packed sequence; choose degrees compatible with the 56 attention heads.
The picker marks unmeasured hardware and option combinations as unverified.
For two H200s or B300s, select encoder fold. Ulysses2 has the
lowest measured server pipeline latency; TP2 is also close and uses less memory.

On H200, B300, GB10 and RTX PRO 6000 Blackwell with standard weights, Auto precomputes the native AdaLN timestep
projections and releases their weights. This retains the original precision
and outputs, reducing measured peak reserved memory by about 24.6–24.7 GiB on one
GPU. Text-to-image and editing use separate plans;
keep the checkpoint files available for uncached plans. The bounded GPU and
host caches reuse these exact results. Select **Keep weights** in the picker,
or pass `--minimax-h3-adaln-online false`, to avoid the rebuild cost. Other
GPUs keep projection weights by default; explicit online precomputation is
available through **Precompute**.

`quality=lossless` is the default. It enables shared DiT and VAE QK-norm/RoPE
fusion, uses cuDNN SDPA in the VAE with a backend fallback, and batches existing
overlapping VAE decode tiles, with at most 128 samples per tile batch.
With multiple GPUs, complete tiles run across the decode group before assembly;
tile geometry and overlap stay unchanged. Select **Local tiles** in the picker
or pass `--vae-config.use-parallel-tiling false` to run all tiles on each GPU.
`quality=exact` restores the release's fusion and sequential tile path. Neither
changes the six-step schedule or canvas. Floating-point reordering can change
pixels; lossless does not mean byte-identical output across paths or topologies.
Text refinement and rotary tables are prepared once per request for both paths.
The shared conditioning cache defaults to 512 MiB per worker. Matching prompts
and ordered reference content can reuse text conditioning; matching VAE inputs
can reuse posterior moments, including with multiple GPUs. Sampling still uses
the request's generator. Pass `--disable-conditioning-cache` to disable reuse
across requests.
These recipes retain the release checkout's library numerics defaults, including
cuDNN TF32 and BF16 reduced-precision reduction. The stricter overrides
`--allow-cudnn-tf32 false --allow-bf16-reduced-precision-reduction false` are
available but are not covered by these measurements.

**Layerwise offload** streams the DiT, text encoder and VAE decoder blocks from
host RAM while preserving native precision. Use **Precompute** with it to avoid
reloading the large AdaLN projection weights. This is the capacity option for
smaller GPUs; resident serving remains faster when the model fits.

**Online FP8 / MXFP8** quantizes the DiT, optionally with the text encoder, from
the original checkpoint. Select it separately in the picker; native BF16 remains
the default. These are approximate modes even with `quality=exact` or `lossless`:
quality controls kernel execution within the selected precision. Quantization
can change composition and reference identity, so compare representative edits
before choosing it. AdaLN
precomputation and the VAE retain native precision. MXFP8 uses block scaling on
Blackwell and the FP8 fallback on Hopper; use FP8 on older supported GPUs.

**Breakable CUDA graph** captures the picker's task, size and reference layout
during startup. Its warmup prompt matches the generated request; editing also
needs the reference files on the server. Matching signatures replay the graph,
while other signatures run eager. **Regional compile** compiles repeated DiT
blocks with PyTorch. Both keep all six steps; compile can change floating-point
rounding and increases cold startup time. Graphs also work with layerwise
offload; capture retains device buffers and needs more GPU memory than eager
streaming. Eager remains the default.

### Hardware recommendations

The picker provides measured recipes for H200/B300, AMD Instinct, single-card RTX PRO 6000 and DGX Spark,
and **Derived** recipes for other listed GPUs. A derived recipe estimates placement and
parallel layout for the selected GPU count from checkpoint sizes and measured
H200 memory use. It has no hardware latency measurement or capacity guarantee.
Apply **Use derived recipe** to set its placement, encoder fold and online
AdaLN together. Start with 2048 × 2048.

| GPU class | Default shape | Recommended placement and layout |
| --- | --- | --- |
| B200, GB200, GB300 | 1 GPU | Auto resident; Ulysses as GPU count grows |
| MI355X / MI350X, MI325X / MI300X | 1 GPU; Ulysses as GPU count grows | Measured Auto resident recipes on 1, 2, 4 and 8 GPUs; Ulysses is faster, TP uses less memory |
| H100 / H800, 80 GB | 1 GPU, or 2 for resident serving | Component offload on one; resident Ulysses2 with encoder fold on two |
| RTX PRO 6000 Blackwell, 96 GB | 1 GPU | Measured resident recipe with AdaLN precomputation; component offload saves memory |
| DGX Spark, 128 GB shared | 1 GB10 | Measured Auto resident recipe with native AdaLN precomputation; requires at least 106 GiB available |
| RTX 6000 Ada, RTX 5090 / 4090 and listed 12–48 GB cards without NVLink | 1 GPU | Layerwise offload with native AdaLN precomputation |
| RTX 3090, 24 GB | 1 GPU; optionally a [two-card NVLink pair](https://www.nvidia.com/content/geforce-gtx/GEFORCE_RTX_3090_USER_GUIDE_v02.pdf) | Layerwise offload with native AdaLN precomputation |

Multi-GPU recommendations assume an NVLink-connected machine. Cards without
NVLink use single-card recipes by default. Offload recipes assume a 256 GB-class
host and local NVMe. Layerwise offload preserves the six steps, native precision
and minimum generation canvas. The capacity validation uses a constrained B300
allocator: native 2K/4K generation and editing, including two-reference 4K,
passed with an 8 GiB allocator limit and a 10.01 GiB measured whole-GPU peak.
It is not a latency measurement on the listed consumer GPUs.

### AMD Instinct (MI300X, MI325X, MI350X, MI355X)

Select an AMD GPU in the picker for single-node recipes on 1, 2, 4 and 8 GPUs.
MI355X and MI350X (gfx950) share one recipe; MI325X and MI300X (gfx942) share
another. The recipes were run on MI350X and MI325X. No attention flag is
needed: the DiT selects AITER on ROCm. Auto placement keeps every component
resident, and AdaLN projections keep their weights. Ring is not available; use
Ulysses or TP. At each GPU count, Ulysses has the lowest measured latency and
TP the lowest memory per GPU.

Install this checkout over the SGLang in an SGLang ROCm image for your GPU:

| GPU | Tested image | Python, PyTorch |
| --- | --- | --- |
| MI355X, MI350X | `lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261109` | 3.10, 2.9.1+rocm7.2.0 |
| MI325X, MI300X | `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi30x-20261107` | 3.12, 2.11.0+rocm10.0.0 |

```bash
docker run -it --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -v "$PWD":/workspace/sglang -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e HF_TOKEN \
  lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261109 bash

# Inside the container, from the SGLang checkout:
cd /workspace/sglang
python3 -m pip install -e python --no-deps --no-build-isolation
# On the Python 3.10 MI35x image, add --ignore-requires-python.
```

The image already provides the ROCm builds of PyTorch, AITER and the
diffusion dependencies, so `--no-deps` keeps them in place. Then run the
picker's serving and request commands in this environment.

### DGX Spark (128 GB)

Select **DGX Spark** for a verified single-node, single-GB10 recipe at 2K and the measured 4K sizes. Use an
Arm64 CUDA environment with GB10 support before installing this checkout.
At `539fea720b86` and later, Auto enables native AdaLN precomputation and keeps
components resident when at least 106 GiB is available in the shared pool.
The picker therefore needs no placement or AdaLN override. Resident GB10
rejects **Keep weights**, which would retain the large projection tensors.
Start with a 2048 × 2048 output.

The [128 GB configuration](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
has one [shared CPU/GPU memory pool](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#guidance-for-reporting-memory-resources-with-unified-memory-architecture).
Component offload does not provide another 128 GB of memory. Whole-component transfers can temporarily hold both CPU and GPU copies in
that pool. Use a bounded
container with swap disabled for the process, such as `--memory 104g
--memory-swap 104g --shm-size 8g`, and download weights inside that container
so its file cache can be reclaimed within the limit. The picker marks option
combinations outside its measured scope **Unverified**.
The container limit does not cap all CUDA allocations; keep at least 106 GiB
available in the shared pool before starting the verified Auto recipe.

The tested ARM64 base is `lmsysorg/sglang@sha256:d4a56f6f9356539957974bb4008161b4c08d14a712805ff63eae526f36dbda71`,
with driver 580.173.02, PyTorch 2.14.1+cu130, FlashInfer 0.7.2 and FA4 4.0.0b38.
Mount this checkout and a task-owned cache directory, then install the source
and matching diffusion dependencies inside the container:

```bash
H3_IMAGE_CACHE="$HOME/.cache/h3-image-spark"
umask 077
mkdir -p "$H3_IMAGE_CACHE"
docker run --platform linux/arm64 --gpus all \
  --memory 104g --memory-swap 104g --shm-size 8g --cpus 12 \
  -v "$PWD":/workspace/sglang -v "$H3_IMAGE_CACHE":/root/.cache/huggingface \
  -e HF_TOKEN -it \
  lmsysorg/sglang@sha256:d4a56f6f9356539957974bb4008161b4c08d14a712805ff63eae526f36dbda71 bash

# Inside the container, from the SGLang checkout:
cd /workspace/sglang
export CARGO_TARGET_DIR=/tmp/h3-image-spark-build
python3 -m pip install setuptools-rust setuptools-scm wheel
python3 -m pip install --no-build-isolation -e 'python[diffusion]'
python3 -m pip uninstall -y flashinfer-jit-cache-sm80 flashinfer-jit-cache-sm89 \
  flashinfer-jit-cache-sm90a flashinfer-jit-cache-sm100a flashinfer-jit-cache-sm103a \
  flashinfer-jit-cache-sm110a flashinfer-jit-cache-sm120f flashinfer-jit-cache-sm121a
python3 -m pip install --no-deps flash-attn-4==4.0.0b38
python3 -m pip install --no-deps flashinfer-cubin==0.7.2 \
  --index-url https://flashinfer.ai/whl/
python3 -m pip install --no-deps flashinfer-jit-cache==0.7.2+cu130 \
  --index-url https://flashinfer.ai/whl/cu130
```

This replaces the base image's incompatible optional kernel providers while
retaining FlashInfer's version checks. Download weights and run the picker's
serving/request commands in this environment. The measured container uses 12
CPU cores; the host's 128 GB is a shared pool, not 128 GB of additional VRAM.

## 4. Measured recipes

### Measured text-to-image summary

The native rows keep the checkpoint's release precision. FP8 and MXFP8 quantize
the DiT and text encoder; they can change image details and reference identity.
Here, 2K is 2048 × 2048 and 4K is 4736 × 3552.

| Hardware | Precision | 2K | 4K |
| --- | --- | --- | --- |
| 2 × B300 | Native | 0.604 s | 2.654 s |
| 2 × B300 | MXFP8 | 0.451 s | 2.065 s |
| 2 × H200 | Native | 1.123 s | 5.465 s |
| 1 × H200 | Native | 1.934 s | 10.133 s |
| 1 × RTX PRO 6000 Blackwell | Native | 3.434 s | 18.544 s |
| 1 × RTX PRO 6000 Blackwell | FP8 | 2.338 s | 14.445 s |
| 1 × DGX Spark | Native | 16.21 s | 84.96 s |
| 8 × MI350X | Native | 0.490 s | 2.110 s |
| 4 × MI325X | Native | 0.890 s | 3.680 s |
| 1 × MI325X | Native | 2.190 s | 10.750 s |

In the picker, use **Ulysses2** and encoder **Fold** for two GPUs. Choose **Auto**
placement for native two-GPU, Spark and AMD recipes, or **Resident** for the other rows.
The MI350X row uses TP8 and the four-MI325X row Ulysses4; MI355X and MI300X use
the MI350X and MI325X recipes.
Select **FP8 DiT + encoder** or **MXFP8 DiT + encoder** only for the quantized rows.

All rows run the full six-step schedule with lossless quality, eager execution,
native AdaLN precomputation and warm conditioning caches. Times are medians of
warm server pipeline requests, excluding startup, HTTP transport, PNG encoding
and client saving; they are not first-request or end-to-end latency.
See the sections below for source revisions, checkpoints, sample counts and
environments.

### DGX Spark at 2K and 4K

One GB10 with Auto resident placement and native AdaLN precomputation, at
`539fea720b86`, using the pinned checkpoints and ARM64 environment above:

| Workload | Pipeline median | Observed warm range |
| --- | --- | --- |
| T2I, 2048 × 2048 | 16.21 s | 15.65–16.80 s |
| Editing, 2048 × 2048 | 25.05 s | 24.58–25.25 s |
| T2I, 4736 × 3552 | 84.96 s | 81.51–87.97 s |
| Editing, 4096 × 4096 | 98.83 s | 93.73–105.62 s |
| T2I, 6272 × 2688 | 81.04 s | 79.36–85.95 s |

Each cell uses 10 warm lossless requests across two fresh servers, excluding the
first lossless request per task in each. All six steps run at native precision;
conditioning is warm. T2I uses seed 42; 2K editing uses seed 42 and 4K editing
uses seed 1234, with one reference. These are pipeline times, excluding HTTP,
PNG encoding and client saving. Maximum CUDA reserved memory over the validated
workflows was about 91 GiB; other runtime and host allocations also share the
same pool. The native offline API also passed 2K generation and editing at
exact and lossless quality with matching decoded pixels.

With cached checkpoints, the two server starts took 281–282 seconds to become
healthy in the 12-core container, including startup warmup. Weight downloads and
installation are outside that boundary. Keep the server running to reuse this
preparation; startup time is excluded from the warm-request table.

### B300 at 2K

Unquantized checkpoints at native precision, 2048 × 2048 generation canvas,
six steps, seed 42 and a CPU generator; one node of B300 GPUs. The image revision is
`7b3d20b4659f5d4b3ad7205daa008e47d6b1e868`, with H3 text encoder revision
`42ed227ee7df40d41602854ae760620d6eb651fe`. Source installation was tested at
`7ccff196eb3a` for the single-GPU row and `539fea720b86` for all two-GPU rows,
with PyTorch 2.14.1+cu130, FlashInfer 0.7.2 and Transformers 5.19.0.
The two-GPU comparison uses an NV18-connected pair, parallel VAE tiles and warm
exact conditioning reuse.

| GPUs and layout | T2I pipeline | Editing pipeline |
| --- | --- | --- |
| 1 × B300, Auto resident | 0.994 s | 1.570 s |
| 2 × B300, TP2, encoder fold | 0.621 s | 0.978 s |
| 2 × B300, Ulysses2, encoder fold | 0.604 s | 0.931 s |
| 2 × B300, Ring2, encoder fold | 0.692 s | 1.093 s |

All rows use lossless quality, resident placement and native AdaLN
precomputation. The single-GPU row is the median of six warm requests per
task across two server lifecycles, after two excluded requests per task in each.
Each two-GPU row uses 10 warm requests across two fresh servers, excluding the
first lossless request per task in each. Layout order is balanced across the
six server starts. Cold plan construction is excluded. All use the checkpoint's
example prompts and one edit reference. These are server-reported pipeline
latencies, excluding HTTP transport, PNG encoding and client saving. They are
not throughput measurements or results for other GPUs. H100 and B200 recipes
remain unmeasured.

### B300 with MXFP8

Select **MXFP8 DiT + encoder**, **Resident**, two B300s and Ulysses2 with
encoder fold. Online quantization changes numerics; native precision remains
the default. At `ff5dede83b0f`, a balanced native → MXFP8 → MXFP8 → native
comparison measured:

| Workload | Native pipeline | MXFP8 pipeline | Reduction |
| --- | --- | --- | --- |
| T2I, 2048 × 2048 | 0.604 s | 0.451 s | 25.4% |
| T2I, 4736 × 3552 | 2.653 s | 2.065 s | 22.2% |
| Editing, 4096 × 4096 | 3.146 s | 2.490 s | 20.9% |

Each cell has 10 warm requests across two fresh servers, excluding the first
lossless request per task in each. Both paths use all six steps, warm exact
conditioning reuse, native AdaLN precomputation and parallel VAE tiles on the
same NV18 pair, checkpoints and dependencies. T2I uses seed 42; editing uses
seed 1234 and one reference. Times exclude HTTP, PNG encoding and client saving.
Maximum server-reported CUDA reserved memory across these workloads fell from
69.2 GiB to 40.0 GiB; this is not total host or combined multi-GPU memory.

MXFP8 DiT + encoder also passed 2K/4K generation, editing and ordered multi-image
requests on one DGX Spark. Choose that weight option in the Spark picker;
AdaLN and VAE keep native precision.

### B300 at 4K

Two B300s at `539fea720b86`, with encoder fold, Auto resident placement and
native AdaLN precomputation, using the same pinned checkpoints and dependencies:

| Layout | T2I, 4736 × 3552 | Editing, 4096 × 4096 | T2I, 6272 × 2688 |
| --- | --- | --- | --- |
| TP2 | 2.790 s | 3.314 s | 2.795 s |
| Ulysses2 | 2.654 s | 3.146 s | 2.662 s |
| Ring2 | 3.032 s | 3.568 s | 3.026 s |

Each cell uses 10 warm lossless requests across two fresh servers, excluding the
first lossless request per task in each. The same balanced layout comparison
covers 2K and 4K. All six steps run at native precision; conditioning is warm.
T2I uses seed 42; editing uses seed 1234 and one reference. These are pipeline
times, excluding HTTP, PNG encoding and client saving. Ulysses2 is the fastest
measured layout for these workloads; TP2 lowers maximum reserved memory across
ranks and workflow phases to about 49.6 GiB, versus 69.2 GiB for Ulysses2 and
69.6 GiB for Ring2. These allocator measurements exclude additional CUDA process
memory.

### RTX PRO 6000 Blackwell, 96 GB

On one RTX PRO 6000 Blackwell Server Edition, resident placement is faster than
component offload for the tested workloads. Both use native precision, FA4,
all six steps and AdaLN precomputation; select **Resident** in the picker.

| Workload | Resident pipeline | Component-offload pipeline |
| --- | --- | --- |
| T2I, 2048 × 2048 | 3.434 s | 5.094 s |
| Editing, 2048 × 2048 | 5.603 s | 7.259 s |
| T2I, 4736 × 3552 | 18.544 s | 20.193 s |
| Editing, 4096 × 4096 | 21.971 s | 23.615 s |
| T2I, 6272 × 2688 | 18.634 s | 20.281 s |

Source `a9d721f36e16`, PyTorch 2.14.1+cu130, FlashInfer 0.7.2 and FA4
4.0.0b35; the same pinned checkpoints and example inputs as above. Each cell
uses 10 warm requests across two fresh servers in a resident/offload ABBA
comparison, excluding the first lossless request per task in each. Conditioning
is warm; HTTP transport, PNG encoding and client saving are excluded.
Maximum reserved memory over the 4K workflows was about 92.2 GiB resident or
66.1 GiB with offload, before additional CUDA process memory. Resident has
limited headroom for other workloads; Auto retains conservative component
offload on this 96 GB device. The picker provides the single-card recipe for
this GPU, which has no NVLink.

For a smaller working set, select **FP8 DiT + encoder** with **Resident**.
At `5c7ce652b563`, a separate native → FP8 → FP8 → native comparison on the
same single GPU measured:

| Workload | Native pipeline | FP8 pipeline |
| --- | --- | --- |
| T2I, 2048 × 2048 | 3.432 s | 2.338 s |
| T2I, 4736 × 3552 | 18.557 s | 14.445 s |
| Editing, 4096 × 4096 | 21.988 s | 17.336 s |

Each cell has 10 warm requests across two fresh servers, excluding the first
lossless request per task in each. All six steps, warm conditioning, example
inputs, seeds and the pipeline timing boundary match the B300 quantization
comparison. Maximum CUDA reserved memory across these workloads fell from
91.6 GiB to 50.4 GiB. FP8 is approximate and remains an explicit option.

### H200 at 2K

H200 measurements use the same checkpoints, prompts, precision, six-step
schedule and canvas. Single-GPU T2I uses `851f8cb71469` with explicit resident
placement and online AdaLN; these become Auto defaults at `9979ece836fb`.
Single-GPU editing uses `be38e34be99d`; Ulysses2 uses `862194eeaedb` with
parallel VAE tiles. Both use Auto placement and exact conditioning reuse.
TP2 and Ring2 use `9979ece836fb`.
All two-GPU rows use encoder fold.

| GPUs and layout | T2I pipeline | Editing pipeline |
| --- | --- | --- |
| 1 × H200, resident | 1.934 s | 3.116 s |
| 2 × H200, TP2, encoder fold | 1.169 s | 2.041 s |
| 2 × H200, Ulysses2, encoder fold | 1.123 s | 1.774 s |
| 2 × H200, Ring2, encoder fold | 1.224 s | 2.146 s |

Single-GPU T2I uses 20 warm requests across four fresh servers, excluding the
first request in each. Single-GPU editing uses eight across two servers,
excluding the first two in each. Each two-GPU row uses 10 across two servers,
excluding the first request per task in each. Peak reserved memory was about
90 GiB on one H200 and, per GPU, 48 GiB for TP2 or 67 GiB for Ulysses2.
These are latency measurements, not throughput results.

For the single-H200 editing workload, the picker defaults to exact conditioning
reuse. Disabling it with `--disable-conditioning-cache` increased the warm
pipeline median from 3.116 s to 3.446 s. Both settings use eight measured
requests across two fresh servers in an ABBA comparison, with the first two
requests excluded per server. All 24 output PNGs matched byte for byte.
Changing the prompt or references can miss the cache; these are repeated-input
measurements, not uncached editing latency.

### H200 at 4K

The Ulysses2 row uses `862194eeaedb` with parallel VAE tiles and exact conditioning reuse; the other
rows use `f1af1685f6c3`. All use the same checkpoint revisions and dependencies
above, native precision, lossless quality and all six steps.
Auto placement and AdaLN precomputation are enabled; two-GPU rows use encoder
fold. Each cell is the median of 10 warm requests across two fresh servers,
excluding the first lossless request for each task in each server.

| GPUs and layout | T2I, 4736 × 3552 | Editing, 4096 × 4096 | T2I, 6272 × 2688 |
| --- | --- | --- | --- |
| 1 × H200 | 10.133 s | 11.839 s | 9.948 s |
| 2 × H200, TP2 | 5.850 s | 6.992 s | 5.730 s |
| 2 × H200, Ulysses2 | 5.465 s | 6.379 s | 5.402 s |
| 2 × H200, Ring2 | 6.023 s | 7.145 s | 5.954 s |

Conditioning caches are warm. T2I uses seed 42; editing uses seed 1234 and
one reference image. These are server pipeline times, excluding HTTP transfer,
PNG encoding and client saving. Ulysses2 gives the lowest measured latency;
TP2 uses less memory. Measured peak reserved memory over these workflows was
about 93 GiB on one H200, or a maximum across ranks of 50 GiB for TP2,
70 GiB for Ulysses2 and 70 GiB for Ring2. These allocator measurements do not
include all CUDA process memory. Other 4K sizes, multiple-reference 4K edits
remain unmeasured on H200.

### AMD Instinct at 2K and 4K

MI325X (gfx942) and MI350X (gfx950), one node each, in the images listed
[above](#amd-instinct-mi300x-mi325x-mi350x-mi355x), at `2dcc93b624` with the
ROCm attention selection described in [Deployment options](#3-deployment-options).
The image revision is `7b3d20b4659f5d4b3ad7205daa008e47d6b1e868`, with
text-encoder weights identical to revision `42ed227ee7df40d41602854ae760620d6eb651fe`.
The server ran plain `sglang serve` with startup warmup, Auto placement and no
attention flag; requests used lossless quality.

| GPUs and layout | T2I, 4736 × 3552 | Editing, 4096 × 4096 | T2I, 6272 × 2688 | T2I, 2048 × 2048 | Editing, 2048 × 2048 | Peak GiB/GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 1 × MI325X | 10.750 s | 12.700 s | 10.730 s | 2.190 s | 3.360 s | 117 |
| 2 × MI325X, TP2 | 8.680 s | 10.060 s | 8.690 s | 1.890 s | 2.940 s | 62 |
| 2 × MI325X, Ulysses2 | 7.570 s | 8.460 s | 7.640 s | 1.710 s | 2.460 s | 93 |
| 4 × MI325X, TP4 | 4.960 s | 5.620 s | 4.960 s | 1.080 s | 1.600 s | 33 |
| 4 × MI325X, Ulysses4 | 3.680 s | 4.620 s | 3.610 s | 0.890 s | 1.350 s | 80 |
| 8 × MI350X, TP8 | 2.110 s | 2.290 s | 2.050 s | 0.490 s | 0.740 s | 19 |

Each cell is the median of five warm requests on one server, excluding the
first request per task. T2I uses seed 42; editing uses seed 1234 and one
reference image, with warm conditioning caches. These
are server pipeline times, excluding HTTP transfer, PNG encoding and client
saving. All 180 requests completed without errors. The MI350X GPUs were shared
with an idle process, so that server passed `--component-residency all=resident`
to keep the resident placement Auto selects on a free 288 GB GPU.

The 1-GPU, TP2, Ulysses2, Ulysses8 and TP8 recipes on both GPUs also served the
three 4K workloads without errors at the earlier day-0 revision `e1ab33200` with
the same attention change. Their latency at `2dcc93b624` is not yet measured.
The guide's T2I and editing examples match the release reference images
visually; outputs are not byte-identical to CUDA or across layouts. MI355X and
MI300X use the same gfx950 and gfx942 recipes and were not measured separately.
Multiple-reference, exact-quality and quantized AMD workloads remain unmeasured.

### PNG encoding at 4K

Select **No compression** under Request options, or send `output_compression=0`,
to trade response size for less PNG encoding and decoding work. The default
compression stays unchanged. On two-H200 and two-B300 Ulysses2 recipes at
`862194eeaedb`, all decoded pixels matched for both formats:

| GPUs | Workload | Default PNG, saved-request time | Uncompressed PNG, saved-request time | PNG bytes, default → uncompressed |
| --- | --- | --- | --- | --- |
| 2 × H200 | T2I, 4736 × 3552 | 8.140 s | 7.211 s | 31.7 → 50.5 MB |
| 2 × H200 | Editing, 4096 × 4096 | 8.838 s | 8.258 s | 22.7 → 50.4 MB |
| 2 × H200 | T2I, 6272 × 2688 | 8.009 s | 7.159 s | 31.0 → 50.6 MB |
| 2 × B300 | T2I, 4736 × 3552 | 5.259 s | 4.367 s | 31.7 → 50.5 MB |
| 2 × B300 | Editing, 4096 × 4096 | 5.577 s | 5.012 s | 22.8 → 50.4 MB |
| 2 × B300 | T2I, 6272 × 2688 | 5.225 s | 4.436 s | 31.1 → 50.6 MB |

Each format uses eight measured requests per workload across two fresh servers,
with balanced request order and one excluded warmup per format in each server.
Timing runs from the loopback HTTP request through PNG saving and pixel validation.
Pipeline time stayed approximately unchanged; these are codec/client-path savings.
Larger payloads may take longer over a slower network. Both formats retain native
precision, all six steps and the same generation canvas.

### File URL delivery

**Base64** remains the default and uses temporary output files. Select **File URL**
under Setup to add `--output-path` to the server command and save each response
through a separate image download. Restart the server after switching. Files
remain in the selected output directory until you remove them; the URL records
last only for that server process.

On two NVLink-connected B300s with Ulysses2 at `539fea720b86`, File URL shortened
the measured save path further when using **No compression**:

| Workload | Base64, uncompressed PNG | File URL, uncompressed PNG |
| --- | --- | --- |
| T2I, 4736 × 3552 | 4.380 s | 3.990 s |
| Editing, 4096 × 4096 | 4.982 s | 4.620 s |
| T2I, 6272 × 2688 | 4.397 s | 4.028 s |

Both columns use the same persistent output directory, native precision, six
steps and warm conditioning. Each cell uses eight measured requests across two
fresh servers, excluding the first request per format/server, with balanced
format and compression order. Timing includes the POST, image download, PNG
saving and pixel validation. Default and uncompressed PNG both retained identical
decoded pixels across response formats. Pipeline latency stayed unchanged.
The extra HTTP request and larger PNGs can cost more on a slower connection.
Other hardware and workloads remain unverified for File URL delivery.

### B300 startup warmup

Peak reserved CUDA memory was about 91 GiB on one B300. These results include
native fusion, VAE batching and exact projection reuse, without reducing steps
or generation area.

One-step server warmup at `f3c0a6f09c6b` prepares both task grids when the
online cache holds at least 12 plans and omits the unnecessary host copy.
Compared with `ea0cba698a08`, on the
same cached checkpoints and one resident B300:

| First-request measurement | Before | Current default |
| --- | --- | --- |
| First editing request, pipeline | 4.041 s | 2.065 s |
| Process start to first editing HTTP response | 38.3 s | 35.4 s |
| Process start to first T2I HTTP response | 36.4 s | 34.0 s |
| Worker CPU RSS after first editing request | 14.52 GiB | 6.52 GiB |

Each first-request scenario uses four fresh servers per path. Both-task
precomputation moves work into startup; avoiding the unused host tier saves
about 8 GiB of worker RSS and removes its allocation cost. The warm-request
table above uses a different timing boundary.
