# Diffusion Models

SGLang Diffusion is an inference framework for accelerated image and video generation using diffusion models. It provides an end-to-end unified pipeline with optimized kernels from sgl-kernel and an efficient scheduler loop.

## Key Features

- **Broad Model Support**: Wan series, FastWan series, Hunyuan, Qwen-Image, Qwen-Image-Edit, Flux, Z-Image, GLM-Image, and more
- **Fast Inference**: Optimized kernels from sgl-kernel, efficient scheduler loop, and Cache-DiT acceleration
- **Ease of Use**: OpenAI-compatible API, CLI, and Python SDK
- **Multi-Platform**: NVIDIA GPUs (H100, H200, A100, B200, 4090) and AMD GPUs (MI300X, MI325X)

---

# Install SGLang-diffusion

You can install sglang-diffusion using one of the methods below.

This page primarily applies to common NVIDIA GPU platforms. For AMD Instinct/ROCm environments see the dedicated [ROCm quickstart](#rocm-quickstart-for-sgl-diffusion), which lists the exact steps (including kernel builds) we used to validate sgl-diffusion on MI300X.

## Method 1: With pip or uv

It is recommended to use uv for a faster installation:

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[diffusion]" --prerelease=allow
```

## Method 2: From source

```bash
# Use the latest release branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install the Python packages
pip install --upgrade pip
pip install -e "python[diffusion]"

# With uv
uv pip install -e "python[diffusion]" --prerelease=allow
```

## Method 3: Using Docker

The Docker images are available on Docker Hub at [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang), built from the [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile).
Replace `<secret>` below with your HuggingFace Hub [token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:dev \
    sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion" \
    --save-output
```

---

# ROCm quickstart for sgl-diffusion

```bash
docker run --device=/dev/kfd --device=/dev/dri --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=<secret> \
  lmsysorg/sglang:v0.5.5.post2-rocm700-mi30x \
  sglang generate --model-path black-forest-labs/FLUX.1-dev --prompt "A logo With Bold Large text: SGL Diffusion" --save-output
```

---

# Compatibility Matrix

The table below shows every supported model and the optimizations supported for them.

The symbols used have the following meanings:

- ✅ = Full compatibility
- ❌ = No compatibility
- ⭕ = Does not apply to this model

## Models x Optimization

The `HuggingFace Model ID` can be passed directly to `from_pretrained()` methods, and sglang-diffusion will use the
optimal
default parameters when initializing and generating videos.

### Video Generation Models

| Model Name                   | Hugging Face Model ID                             | Resolutions         | TeaCache | Sliding Tile Attn | Sage Attn | Video Sparse Attention (VSA) |
|:-----------------------------|:--------------------------------------------------|:--------------------|:--------:|:-----------------:|:---------:|:----------------------------:|
| FastWan2.1 T2V 1.3B          | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`         | 480p                |    ⭕     |         ⭕         |     ⭕     |              ✅               |
| FastWan2.2 TI2V 5B Full Attn | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 720p                |    ⭕     |         ⭕         |     ⭕     |              ✅               |
| Wan2.2 TI2V 5B               | `Wan-AI/Wan2.2-TI2V-5B-Diffusers`                 | 720p                |    ⭕     |         ⭕         |     ✅     |              ⭕               |
| Wan2.2 T2V A14B              | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`                | 480p<br>720p        |    ❌     |         ❌         |     ✅     |              ⭕               |
| Wan2.2 I2V A14B              | `Wan-AI/Wan2.2-I2V-A14B-Diffusers`                | 480p<br>720p        |    ❌     |         ❌         |     ✅     |              ⭕               |
| HunyuanVideo                 | `hunyuanvideo-community/HunyuanVideo`             | 720×1280<br>544×960 |    ❌     |         ✅         |     ✅     |              ⭕               |
| FastHunyuan                  | `FastVideo/FastHunyuan-diffusers`                 | 720×1280<br>544×960 |    ❌     |         ✅         |     ✅     |              ⭕               |
| Wan2.1 T2V 1.3B              | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`                | 480p                |    ✅     |         ✅         |     ✅     |              ⭕               |
| Wan2.1 T2V 14B               | `Wan-AI/Wan2.1-T2V-14B-Diffusers`                 | 480p, 720p          |    ✅     |         ✅         |     ✅     |              ⭕               |
| Wan2.1 I2V 480P              | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`            | 480p                |    ✅     |         ✅         |     ✅     |              ⭕               |
| Wan2.1 I2V 720P              | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`            | 720p                |    ✅     |         ✅         |     ✅     |              ⭕               |

**Note**: Wan2.2 TI2V 5B has some quality issues when performing I2V generation. We are working on fixing this issue.

### Image Generation Models

| Model Name       | HuggingFace Model ID                    | Resolutions    |
|:-----------------|:----------------------------------------|:---------------|
| FLUX.1-dev       | `black-forest-labs/FLUX.1-dev`          | Any resolution |
| FLUX.2-dev       | `black-forest-labs/FLUX.2-dev`          | Any resolution |
| FLUX.2-Klein     | `black-forest-labs/FLUX.2-klein-4B`     | Any resolution |
| Z-Image-Turbo    | `Tongyi-MAI/Z-Image-Turbo`              | Any resolution |
| GLM-Image        | `zai-org/GLM-Image`                     | Any resolution |
| Qwen Image       | `Qwen/Qwen-Image`                       | Any resolution |
| Qwen Image 2512  | `Qwen/Qwen-Image-2512`                  | Any resolution |
| Qwen Image Edit  | `Qwen/Qwen-Image-Edit`                  | Any resolution |

## Verified LoRA Examples

This section lists example LoRAs that have been explicitly tested and verified with each base model in the **SGLang Diffusion** pipeline.

> Important: \
> LoRAs that are not listed here are not necessarily incompatible.
> In practice, most standard LoRAs are expected to work, especially those following common Diffusers or SD-style conventions.
> The entries below simply reflect configurations that have been manually validated by the SGLang team.

### Verified LoRAs by Base Model

| Base Model       | Supported LoRAs |
|:-----------------|:----------------|
| Wan2.2           | `lightx2v/Wan2.2-Distill-Loras`<br>`Cseti/wan2.2-14B-Arcane_Jinx-lora-v1` |
| Wan2.1           | `lightx2v/Wan2.1-Distill-Loras` |
| Z-Image-Turbo    | `tarn59/pixel_art_style_lora_z_image_turbo`<br>`wcde/Z-Image-Turbo-DeJPEG-Lora` |
| Qwen-Image       | `lightx2v/Qwen-Image-Lightning`<br>`flymy-ai/qwen-image-realism-lora`<br>`prithivMLmods/Qwen-Image-HeadshotX`<br>`starsfriday/Qwen-Image-EVA-LoRA` |
| Qwen-Image-Edit  | `ostris/qwen_image_edit_inpainting`<br>`lightx2v/Qwen-Image-Edit-2511-Lightning` |
| Flux             | `dvyio/flux-lora-simple-illustration`<br>`XLabs-AI/flux-furry-lora`<br>`XLabs-AI/flux-RealismLora` |

#### Special Requirements

> [!NOTE]
> Sliding Tile Attention: Currently, only Hopper GPUs (H100s) are supported.


---

# SGLang diffusion CLI Inference

The SGLang-diffusion CLI provides a quick way to access the inference pipeline for image and video generation.

## Prerequisites

- A working SGLang diffusion installation and the `sglang` CLI available in `$PATH`.
- Python 3.11+ if you plan to use the OpenAI Python SDK.


## Supported Arguments

### Server Arguments

- `--model-path {MODEL_PATH}`: Path to the model or model ID
- `--vae-path {VAE_PATH}`: Path to a custom VAE model or HuggingFace model ID (e.g., `fal/FLUX.2-Tiny-AutoEncoder`). If not specified, the VAE will be loaded from the main model path.
- `--lora-path {LORA_PATH}`: Path to a LoRA adapter (local path or HuggingFace model ID). If not specified, LoRA will not be applied.
- `--lora-nickname {NAME}`: Nickname for the LoRA adapter. (default: `default`).
- `--num-gpus {NUM_GPUS}`: Number of GPUs to use
- `--tp-size {TP_SIZE}`: Tensor parallelism size (only for the encoder; should not be larger than 1 if text encoder offload is enabled, as layer-wise offload plus prefetch is faster)
- `--sp-degree {SP_SIZE}`: Sequence parallelism size (typically should match the number of GPUs)
- `--ulysses-degree {ULYSSES_DEGREE}`: The degree of DeepSpeed-Ulysses-style SP in USP
- `--ring-degree {RING_DEGREE}`: The degree of ring attention-style SP in USP


### Sampling Parameters

- `--prompt {PROMPT}`: Text description for the video you want to generate
- `--num-inference-steps {STEPS}`: Number of denoising steps
- `--negative-prompt {PROMPT}`: Negative prompt to guide generation away from certain concepts
- `--seed {SEED}`: Random seed for reproducible generation


#### Image/Video Configuration

- `--height {HEIGHT}`: Height of the generated output
- `--width {WIDTH}`: Width of the generated output
- `--num-frames {NUM_FRAMES}`: Number of frames to generate
- `--fps {FPS}`: Frames per second for the saved output, if this is a video-generation task


#### Output Options

- `--output-path {PATH}`: Directory to save the generated video
- `--save-output`: Whether to save the image/video to disk
- `--return-frames`: Whether to return the raw frames

### Using Configuration Files

Instead of specifying all parameters on the command line, you can use a configuration file:

```bash
sglang generate --config {CONFIG_FILE_PATH}
```

The configuration file should be in JSON or YAML format with the same parameter names as the CLI options. Command-line arguments take precedence over settings in the configuration file, allowing you to override specific values while keeping the rest from the configuration file.

Example configuration file (config.json):

```json
{
    "model_path": "FastVideo/FastHunyuan-diffusers",
    "prompt": "A beautiful woman in a red dress walking down a street",
    "output_path": "outputs/",
    "num_gpus": 2,
    "sp_size": 2,
    "tp_size": 1,
    "num_frames": 45,
    "height": 720,
    "width": 1280,
    "num_inference_steps": 6,
    "seed": 1024,
    "fps": 24,
    "precision": "bf16",
    "vae_precision": "fp16",
    "vae_tiling": true,
    "vae_sp": true,
    "vae_config": {
        "load_encoder": false,
        "load_decoder": true,
        "tile_sample_min_height": 256,
        "tile_sample_min_width": 256
    },
    "text_encoder_precisions": [
        "fp16",
        "fp16"
    ],
    "mask_strategy_file_path": null,
    "enable_torch_compile": false
}
```

Or using YAML format (config.yaml):

```yaml
model_path: "FastVideo/FastHunyuan-diffusers"
prompt: "A beautiful woman in a red dress walking down a street"
output_path: "outputs/"
num_gpus: 2
sp_size: 2
tp_size: 1
num_frames: 45
height: 720
width: 1280
num_inference_steps: 6
seed: 1024
fps: 24
precision: "bf16"
vae_precision: "fp16"
vae_tiling: true
vae_sp: true
vae_config:
  load_encoder: false
  load_decoder: true
  tile_sample_min_height: 256
  tile_sample_min_width: 256
text_encoder_precisions:
  - "fp16"
  - "fp16"
mask_strategy_file_path: null
enable_torch_compile: false
```


To see all the options, you can use the `--help` flag:

```bash
sglang generate --help
```

## Serve

Launch the SGLang diffusion HTTP server and interact with it using the OpenAI SDK and curl.

### Start the server

Use the following command to launch the server:

```bash
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  --text-encoder-cpu-offload
  --pin-cpu-memory
  --num-gpus 4
  --ulysses-degree=2
  --ring-degree=2
)

sglang serve "${SERVER_ARGS[@]}"
```

- **--model-path**: Which model to load. The example uses `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
- **--port**: HTTP port to listen on (the default here is `30010`).

For detailed API usage, including Image, Video Generation and LoRA management, please refer to the [OpenAI API Documentation](#sglang-diffusion-openai-api).


## Generate

Run a one-off generation task without launching a persistent server.

To use it, pass both server arguments and sampling parameters in one command, after the `generate` subcommand, for example:

```bash
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers
  --text-encoder-cpu-offload
  --pin-cpu-memory
  --num-gpus 4
  --ulysses-degree=2
  --ring-degree=2
)

SAMPLING_ARGS=(
  --prompt "A curious raccoon"
  --save-output
  --output-path outputs
  --output-file-name "A curious raccoon.mp4"
)

sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"

# Or, users can set `SGLANG_CACHE_DIT_ENABLED` env as `true` to enable cache acceleration
SGLANG_CACHE_DIT_ENABLED=true sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"
```

Once the generation task has finished, the server will shut down automatically.

> [!NOTE]
> The HTTP server-related arguments are ignored in this subcommand.

## Diffusers Backend

SGLang diffusion supports a **diffusers backend** that allows you to run any diffusers-compatible model through SGLang's infrastructure using vanilla diffusers pipelines. This is useful for running models without native SGLang implementations or models with custom pipeline classes.

### Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--backend` | `auto` (default), `sglang`, `diffusers` | `auto`: prefer native SGLang, fallback to diffusers. `sglang`: force native (fails if unavailable). `diffusers`: force vanilla diffusers pipeline. |
| `--diffusers-attention-backend` | `flash`, `_flash_3_hub`, `sage`, `xformers`, `native` | Attention backend for diffusers pipelines. See [diffusers attention backends](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends). |
| `--trust-remote-code` | flag | Required for models with custom pipeline classes (e.g., Ovis). |
| `--vae-tiling` | flag | Enable VAE tiling for large image support (decodes tile-by-tile). |
| `--vae-slicing` | flag | Enable VAE slicing for lower memory usage (decodes slice-by-slice). |
| `--dit-precision` | `fp16`, `bf16`, `fp32` | Precision for the diffusion transformer. |
| `--vae-precision` | `fp16`, `bf16`, `fp32` | Precision for the VAE. |

### Example: Running Ovis-Image-7B

[Ovis-Image-7B](https://huggingface.co/AIDC-AI/Ovis-Image-7B) is a 7B text-to-image model optimized for high-quality text rendering.

```bash
sglang generate \
  --model-path AIDC-AI/Ovis-Image-7B \
  --backend diffusers \
  --trust-remote-code \
  --diffusers-attention-backend flash \
  --prompt "A serene Japanese garden with cherry blossoms" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 30 \
  --save-output \
  --output-path outputs \
  --output-file-name ovis_garden.png
```

### Extra Diffusers Arguments

For pipeline-specific parameters not exposed via CLI, use `diffusers_kwargs` in a config file:

```json
{
    "model_path": "AIDC-AI/Ovis-Image-7B",
    "backend": "diffusers",
    "prompt": "A beautiful landscape",
    "diffusers_kwargs": {
        "cross_attention_kwargs": {"scale": 0.5}
    }
}
```

```bash
sglang generate --config config.json
```

---

# SGLang Diffusion OpenAI API

The SGLang diffusion HTTP server implements an OpenAI-compatible API for image and video generation, as well as LoRA adapter management.

## Serve

Launch the server using the `sglang serve` command.

### Start the server

```bash
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  --text-encoder-cpu-offload
  --pin-cpu-memory
  --num-gpus 4
  --ulysses-degree=2
  --ring-degree=2
  --port 30010
)

sglang serve "${SERVER_ARGS[@]}"
```

- **--model-path**: Path to the model or model ID.
- **--port**: HTTP port to listen on (default: `30000`).

#### Get Model Information

**Endpoint:** `GET /models`

Returns information about the model served by this server, including model path, task type, pipeline configuration, and precision settings.

**Curl Example:**

```bash
curl -sS -X GET "http://localhost:30010/models"
```

**Response Example:**

```json
{
  "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "task_type": "T2V",
  "pipeline_name": "wan_pipeline",
  "pipeline_class": "WanPipeline",
  "num_gpus": 4,
  "dit_precision": "bf16",
  "vae_precision": "fp16"
}
```

---

## Endpoints

### Image Generation

The server implements an OpenAI-compatible Images API under the `/v1/images` namespace.

#### Create an image

**Endpoint:** `POST /v1/images/generations`

**Python Example (b64_json response):**

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1")

img = client.images.generate(
    prompt="A calico cat playing a piano on stage",
    size="1024x1024",
    n=1,
    response_format="b64_json",
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```

**Curl Example:**

```bash
curl -sS -X POST "http://localhost:30010/v1/images/generations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -d '{
        "prompt": "A calico cat playing a piano on stage",
        "size": "1024x1024",
        "n": 1,
        "response_format": "b64_json"
      }'
```

> **Note**
> The `response_format=url` option is not supported for `POST /v1/images/generations` and will return a `400` error.

#### Edit an image

**Endpoint:** `POST /v1/images/edits`

This endpoint accepts a multipart form upload with input images and a text prompt. The server can return either a base64-encoded image or a URL to download the image.

**Curl Example (b64_json response):**

```bash
curl -sS -X POST "http://localhost:30010/v1/images/edits" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -F "image=@local_input_image.png" \
  -F "url=image_url.jpg" \
  -F "prompt=A calico cat playing a piano on stage" \
  -F "size=1024x1024" \
  -F "response_format=b64_json"
```

**Curl Example (URL response):**

```bash
curl -sS -X POST "http://localhost:30010/v1/images/edits" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -F "image=@local_input_image.png" \
  -F "url=image_url.jpg" \
  -F "prompt=A calico cat playing a piano on stage" \
  -F "size=1024x1024" \
  -F "response_format=url"
```

#### Download image content

When `response_format=url` is used with `POST /v1/images/edits`, the API returns a relative URL like `/v1/images/<IMAGE_ID>/content`.

**Endpoint:** `GET /v1/images/{image_id}/content`

**Curl Example:**

```bash
curl -sS -L "http://localhost:30010/v1/images/<IMAGE_ID>/content" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -o output.png
```

### Video Generation

The server implements a subset of the OpenAI Videos API under the `/v1/videos` namespace.

#### Create a video

**Endpoint:** `POST /v1/videos`

**Python Example:**

```python
from openai import OpenAI

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1")

video = client.videos.create(
    prompt="A calico cat playing a piano on stage",
    size="1280x720"
)
print(f"Video ID: {video.id}, Status: {video.status}")
```

**Curl Example:**

```bash
curl -sS -X POST "http://localhost:30010/v1/videos" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -d '{
        "prompt": "A calico cat playing a piano on stage",
        "size": "1280x720"
      }'
```

#### List videos

**Endpoint:** `GET /v1/videos`

**Python Example:**

```python
videos = client.videos.list()
for item in videos.data:
    print(item.id, item.status)
```

**Curl Example:**

```bash
curl -sS -X GET "http://localhost:30010/v1/videos" \
  -H "Authorization: Bearer sk-proj-1234567890"
```

#### Download video content

**Endpoint:** `GET /v1/videos/{video_id}/content`

**Python Example:**

```python
import time

# Poll for completion
while True:
    page = client.videos.list()
    item = next((v for v in page.data if v.id == video_id), None)
    if item and item.status == "completed":
        break
    time.sleep(5)

# Download content
resp = client.videos.download_content(video_id=video_id)
with open("output.mp4", "wb") as f:
    f.write(resp.read())
```

**Curl Example:**

```bash
curl -sS -L "http://localhost:30010/v1/videos/<VIDEO_ID>/content" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -o output.mp4
```

---

### LoRA Management

The server supports dynamic loading, merging, and unmerging of LoRA adapters.

**Important Notes:**
- Mutual Exclusion: Only one LoRA can be *merged* (active) at a time
- Switching: To switch LoRAs, you must first `unmerge` the current one, then `set` the new one
- Caching: The server caches loaded LoRA weights in memory. Switching back to a previously loaded LoRA (same path) has little cost

#### Set LoRA Adapter

Loads one or more LoRA adapters and merges their weights into the model. Supports both single LoRA (backward compatible) and multiple LoRA adapters.

**Endpoint:** `POST /v1/set_lora`

**Parameters:**
- `lora_nickname` (string or list of strings, required): A unique identifier for the LoRA adapter(s). Can be a single string or a list of strings for multiple LoRAs
- `lora_path` (string or list of strings/None, optional): Path to the `.safetensors` file(s) or Hugging Face repo ID(s). Required for the first load; optional if re-activating a cached nickname. If a list, must match the length of `lora_nickname`
- `target` (string or list of strings, optional): Which transformer(s) to apply the LoRA to. If a list, must match the length of `lora_nickname`. Valid values:
  - `"all"` (default): Apply to all transformers
  - `"transformer"`: Apply only to the primary transformer (high noise for Wan2.2)
  - `"transformer_2"`: Apply only to transformer_2 (low noise for Wan2.2)
  - `"critic"`: Apply only to the critic model
- `strength` (float or list of floats, optional): LoRA strength for merge, default 1.0. If a list, must match the length of `lora_nickname`. Values < 1.0 reduce the effect, values > 1.0 amplify the effect

**Single LoRA Example:**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": "lora_name",
        "lora_path": "/path/to/lora.safetensors",
        "target": "all",
        "strength": 0.8
      }'
```

**Multiple LoRA Example:**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": ["lora_1", "lora_2"],
        "lora_path": ["/path/to/lora1.safetensors", "/path/to/lora2.safetensors"],
        "target": ["transformer", "transformer_2"],
        "strength": [0.8, 1.0]
      }'
```

**Multiple LoRA with Same Target:**

```bash
curl -X POST http://localhost:30010/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
        "lora_nickname": ["style_lora", "character_lora"],
        "lora_path": ["/path/to/style.safetensors", "/path/to/character.safetensors"],
        "target": "all",
        "strength": [0.7, 0.9]
      }'
```

> [!NOTE]
> When using multiple LoRAs:
> - All list parameters (`lora_nickname`, `lora_path`, `target`, `strength`) must have the same length
> - If `target` or `strength` is a single value, it will be applied to all LoRAs
> - Multiple LoRAs applied to the same target will be merged in order


#### Merge LoRA Weights

Manually merges the currently set LoRA weights into the base model.

> [!NOTE]
> `set_lora` automatically performs a merge, so this is typically only needed if you have manually unmerged but want to re-apply the same LoRA without calling `set_lora` again.*

**Endpoint:** `POST /v1/merge_lora_weights`

**Parameters:**
- `target` (string, optional): Which transformer(s) to merge. One of "all" (default), "transformer", "transformer_2", "critic"
- `strength` (float, optional): LoRA strength for merge, default 1.0. Values < 1.0 reduce the effect, values > 1.0 amplify the effect

**Curl Example:**

```bash
curl -X POST http://localhost:30010/v1/merge_lora_weights \
  -H "Content-Type: application/json" \
  -d '{"strength": 0.8}'
```


#### Unmerge LoRA Weights

Unmerges the currently active LoRA weights from the base model, restoring it to its original state. This **must** be called before setting a different LoRA.

**Endpoint:** `POST /v1/unmerge_lora_weights`

**Curl Example:**

```bash
curl -X POST http://localhost:30010/v1/unmerge_lora_weights \
  -H "Content-Type: application/json"
```

#### List LoRA Adapters

Returns loaded LoRA adapters and current application status per module.

**Endpoint:** `GET /v1/list_loras`

**Curl Example:**

```bash
curl -sS -X GET "http://localhost:30010/v1/list_loras"
```

**Response Example:**

```json
{
  "loaded_adapters": [
    { "nickname": "lora_a", "path": "/weights/lora_a.safetensors" },
    { "nickname": "lora_b", "path": "/weights/lora_b.safetensors" }
  ],
  "active": {
    "transformer": [
      {
        "nickname": "lora2",
        "path": "tarn59/pixel_art_style_lora_z_image_turbo",
        "merged": true,
        "strength": 1.0
      }
    ]
  }
}
```

Notes:
- If LoRA is not enabled for the current pipeline, the server will return an error.
- `num_lora_layers_with_weights` counts only layers that have LoRA weights applied for the active adapter.

### Example: Switching LoRAs

1.  Set LoRA A:
    ```bash
    curl -X POST http://localhost:30010/v1/set_lora -d '{"lora_nickname": "lora_a", "lora_path": "path/to/A"}'
    ```
2.  Generate with LoRA A...
3.  Unmerge LoRA A:
    ```bash
    curl -X POST http://localhost:30010/v1/unmerge_lora_weights
    ```
4.  Set LoRA B:
    ```bash
    curl -X POST http://localhost:30010/v1/set_lora -d '{"lora_nickname": "lora_b", "lora_path": "path/to/B"}'
    ```
5.  Generate with LoRA B...

---

# Attention Backends

This document describes the attention backends available in sglang diffusion (`sglang.multimodal_gen`) and how to select them.

## Overview

Attention backends are defined by `AttentionBackendEnum` (`sglang.multimodal_gen.runtime.platforms.interface.AttentionBackendEnum`) and selected via the CLI flag `--attention-backend`.

Backend selection is performed by the shared attention layers (e.g. `LocalAttention` / `USPAttention` / `UlyssesAttention` in `sglang.multimodal_gen.runtime.layers.attention.layer`) and therefore applies to any model component using these layers (e.g. diffusion transformer / DiT and encoders).

- **CUDA**: prefers FlashAttention (FA3/FA4) when supported; otherwise falls back to PyTorch SDPA.
- **ROCm**: uses FlashAttention when available; otherwise falls back to PyTorch SDPA.
- **MPS**: always uses PyTorch SDPA.

## Backend options

The CLI accepts the lowercase names of `AttentionBackendEnum`. The table below lists the backends implemented by the built-in platforms. `fa3`/`fa4` are accepted as aliases for `fa`.

| CLI value | Enum value | Notes |
|---|---|---|
| `fa` / `fa3` / `fa4` | `FA` | FlashAttention. `fa3/fa4` are normalized to `fa` during argument parsing (`ServerArgs.__post_init__`). |
| `torch_sdpa` | `TORCH_SDPA` | PyTorch `scaled_dot_product_attention`. |
| `sliding_tile_attn` | `SLIDING_TILE_ATTN` | Sliding Tile Attention (STA). Requires `st_attn` and a mask-strategy config file set via the `SGLANG_DIFFUSION_ATTENTION_CONFIG` environment variable. |
| `sage_attn` | `SAGE_ATTN` | Requires `sageattention`. Upstream SageAttention CUDA extensions target SM80/SM86/SM89/SM90/SM120 (compute capability 8.0/8.6/8.9/9.0/12.0); see upstream `setup.py`: https://github.com/thu-ml/SageAttention/blob/main/setup.py. |
| `sage_attn_3` | `SAGE_ATTN_3` | Requires SageAttention3 installed per upstream instructions. |
| `video_sparse_attn` | `VIDEO_SPARSE_ATTN` | Requires `vsa`. |
| `vmoba_attn` | `VMOBA_ATTN` | Requires `kernel.attn.vmoba_attn.vmoba`. |
| `aiter` | `AITER` | Requires `aiter`. |

## Selection priority

The selection order in `runtime/layers/attention/selector.py` is:

1. `global_force_attn_backend(...)` / `global_force_attn_backend_context_manager(...)`
2. CLI `--attention-backend` (`ServerArgs.attention_backend`)
3. Auto selection (platform capability, dtype, and installed packages)

## Platform support matrix

| Backend | CUDA | ROCm | MPS | Notes |
|---|---:|---:|---:|---|
| `fa` | ✅ | ✅ | ❌ | CUDA requires SM80+ and fp16/bf16. FlashAttention is only used when the required runtime is installed; otherwise it falls back to `torch_sdpa`. |
| `torch_sdpa` | ✅ | ✅ | ✅ | Most compatible option across platforms. |
| `sliding_tile_attn` | ✅ | ❌ | ❌ | CUDA-only. Requires `st_attn` and `SGLANG_DIFFUSION_ATTENTION_CONFIG`. |
| `sage_attn` | ✅ | ❌ | ❌ | CUDA-only (optional dependency). |
| `sage_attn_3` | ✅ | ❌ | ❌ | CUDA-only (optional dependency). |
| `video_sparse_attn` | ✅ | ❌ | ❌ | CUDA-only. Requires `vsa`. |
| `vmoba_attn` | ✅ | ❌ | ❌ | CUDA-only. Requires `kernel.attn.vmoba_attn.vmoba`. |
| `aiter` | ✅ | ❌ | ❌ | Requires `aiter`. |

## Usage

### Select a backend via CLI

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend fa
```

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend torch_sdpa
```

### Using Sliding Tile Attention (STA)

```bash
export SGLANG_DIFFUSION_ATTENTION_CONFIG=/abs/path/to/mask_strategy.json

sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "..." \
  --attention-backend sliding_tile_attn
```

### Notes for ROCm / MPS

- ROCm: use `--attention-backend torch_sdpa` or `fa` depending on what is available in your environment.
- MPS: the platform implementation always uses `torch_sdpa`.

---

# Cache-DiT Acceleration

SGLang integrates [Cache-DiT](https://github.com/vipshop/cache-dit), a caching acceleration engine for Diffusion
Transformers (DiT), to achieve up to **7.4x inference speedup** with minimal quality loss.

## Overview

**Cache-DiT** uses intelligent caching strategies to skip redundant computation in the denoising loop:

- **DBCache (Dual Block Cache)**: Dynamically decides when to cache transformer blocks based on residual differences
- **TaylorSeer**: Uses Taylor expansion for calibration to optimize caching decisions
- **SCM (Step Computation Masking)**: Step-level caching control for additional speedup

## Basic Usage

Enable Cache-DiT by exporting the environment variable and using `sglang generate` or `sglang serve` :

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

## Advanced Configuration

### DBCache Parameters

DBCache controls block-level caching behavior:

| Parameter | Env Variable              | Default | Description                              |
|-----------|---------------------------|---------|------------------------------------------|
| Fn        | `SGLANG_CACHE_DIT_FN`     | 1       | Number of first blocks to always compute |
| Bn        | `SGLANG_CACHE_DIT_BN`     | 0       | Number of last blocks to always compute  |
| W         | `SGLANG_CACHE_DIT_WARMUP` | 4       | Warmup steps before caching starts       |
| R         | `SGLANG_CACHE_DIT_RDT`    | 0.24    | Residual difference threshold            |
| MC        | `SGLANG_CACHE_DIT_MC`     | 3       | Maximum continuous cached steps          |

### TaylorSeer Configuration

TaylorSeer improves caching accuracy using Taylor expansion:

| Parameter | Env Variable                  | Default | Description                     |
|-----------|-------------------------------|---------|---------------------------------|
| Enable    | `SGLANG_CACHE_DIT_TAYLORSEER` | false   | Enable TaylorSeer calibrator    |
| Order     | `SGLANG_CACHE_DIT_TS_ORDER`   | 1       | Taylor expansion order (1 or 2) |

### Combined Configuration Example

DBCache and TaylorSeer are complementary strategies that work together, you can configure both sets of parameters
simultaneously:

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=2 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.4 \
SGLANG_CACHE_DIT_MC=4 \
SGLANG_CACHE_DIT_TAYLORSEER=true \
SGLANG_CACHE_DIT_TS_ORDER=2 \
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A curious raccoon in a forest"
```

### SCM (Step Computation Masking)

SCM provides step-level caching control for additional speedup. It decides which denoising steps to compute fully and
which to use cached results.

#### SCM Presets

SCM is configured with presets:

| Preset   | Compute Ratio | Speed    | Quality    |
|----------|---------------|----------|------------|
| `none`   | 100%          | Baseline | Best       |
| `slow`   | ~75%          | ~1.3x    | High       |
| `medium` | ~50%          | ~2x      | Good       |
| `fast`   | ~35%          | ~3x      | Acceptable |
| `ultra`  | ~25%          | ~4x      | Lower      |

##### Usage

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_PRESET=medium \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A futuristic cityscape at sunset"
```

#### Custom SCM Bins

For fine-grained control over which steps to compute vs cache:

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_COMPUTE_BINS="8,3,3,2,2" \
SGLANG_CACHE_DIT_SCM_CACHE_BINS="1,2,2,2,3" \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A futuristic cityscape at sunset"
```

#### SCM Policy

| Policy    | Env Variable                          | Description                                 |
|-----------|---------------------------------------|---------------------------------------------|
| `dynamic` | `SGLANG_CACHE_DIT_SCM_POLICY=dynamic` | Adaptive caching based on content (default) |
| `static`  | `SGLANG_CACHE_DIT_SCM_POLICY=static`  | Fixed caching pattern                       |

## Environment Variables

All Cache-DiT parameters can be set via the following environment variables:

| Environment Variable                | Default | Description                              |
|-------------------------------------|---------|------------------------------------------|
| `SGLANG_CACHE_DIT_ENABLED`          | false   | Enable Cache-DiT acceleration            |
| `SGLANG_CACHE_DIT_FN`               | 1       | First N blocks to always compute         |
| `SGLANG_CACHE_DIT_BN`               | 0       | Last N blocks to always compute          |
| `SGLANG_CACHE_DIT_WARMUP`           | 4       | Warmup steps before caching              |
| `SGLANG_CACHE_DIT_RDT`              | 0.24    | Residual difference threshold            |
| `SGLANG_CACHE_DIT_MC`               | 3       | Max continuous cached steps              |
| `SGLANG_CACHE_DIT_TAYLORSEER`       | false   | Enable TaylorSeer calibrator             |
| `SGLANG_CACHE_DIT_TS_ORDER`         | 1       | TaylorSeer order (1 or 2)                |
| `SGLANG_CACHE_DIT_SCM_PRESET`       | none    | SCM preset (none/slow/medium/fast/ultra) |
| `SGLANG_CACHE_DIT_SCM_POLICY`       | dynamic | SCM caching policy                       |
| `SGLANG_CACHE_DIT_SCM_COMPUTE_BINS` | not set | Custom SCM compute bins                  |
| `SGLANG_CACHE_DIT_SCM_CACHE_BINS`   | not set | Custom SCM cache bins                    |

## Supported Models

SGLang Diffusion x Cache-DiT supports almost all models originally supported in SGLang Diffusion:

| Model Family | Example Models                            |
|--------------|-------------------------------------------|
| Wan          | Wan2.1, Wan2.2                            |
| Flux         | FLUX.1-dev, FLUX.2-dev, FLUX.2-Klein      |
| Z-Image      | Z-Image-Turbo                             |
| Qwen         | Qwen-Image, Qwen-Image-Edit               |
| GLM          | GLM-Image                                 |
| Hunyuan      | HunyuanVideo                              |

## Performance Tips

1. **Start with defaults**: The default parameters work well for most models
2. **Use TaylorSeer**: It typically improves both speed and quality
3. **Tune R threshold**: Lower values = better quality, higher values = faster
4. **SCM for extra speed**: Use `medium` preset for good speed/quality balance
5. **Warmup matters**: Higher warmup = more stable caching decisions

## Limitations

- **Single GPU only**: Distributed support (TP/SP) is not yet validated; Cache-DiT will be automatically disabled when
  `world_size > 1`
- **SCM minimum steps**: SCM requires >= 8 inference steps to be effective
- **Model support**: Only models registered in Cache-DiT's BlockAdapterRegister are supported

## Troubleshooting

### Distributed environment warning

```
WARNING: cache-dit is disabled in distributed environment (world_size=N)
```

This is expected behavior. Cache-DiT currently only supports single-GPU inference.

### SCM disabled for low step count

For models with < 8 inference steps (e.g., DMD distilled models), SCM will be automatically disabled. DBCache
acceleration still works.

## References

- [Cache-Dit](https://github.com/vipshop/cache-dit)
- [SGLang Diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)

---

# Profiling Multimodal Generation

This guide covers profiling techniques for multimodal generation pipelines in SGLang.

## PyTorch Profiler

PyTorch Profiler provides detailed kernel execution time, call stack, and GPU utilization metrics.

### Denoising Stage Profiling

Profile the denoising stage with sampled timesteps (default: 5 steps after 1 warmup step):

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --seed 0 \
  --profile
```

**Parameters:**
- `--profile`: Enable profiling for the denoising stage
- `--num-profiled-timesteps N`: Number of timesteps to profile after warmup (default: 5)
  - Smaller values reduce trace file size
  - Example: `--num-profiled-timesteps 10` profiles 10 steps after 1 warmup step

### Full Pipeline Profiling

Profile all pipeline stages (text encoding, denoising, VAE decoding, etc.):

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A Logo With Bold Large Text: SGL Diffusion" \
  --seed 0 \
  --profile \
  --profile-all-stages
```

**Parameters:**
- `--profile-all-stages`: Used with `--profile`, profile all pipeline stages instead of just denoising

### Output Location

By default, trace files are saved in the ./logs/ directory.

The exact output file path will be shown in the console output, for example:

```bash
[mm-dd hh:mm:ss] Saved profiler traces to: /sgl-workspace/sglang/logs/mocked_fake_id_for_offline_generate-5_steps-global-rank0.trace.json.gz
```

### View Traces

Load and visualize trace files at:
- https://ui.perfetto.dev/ (recommended)
- chrome://tracing (Chrome only)

For large trace files, reduce `--num-profiled-timesteps` or avoid using `--profile-all-stages`.


### `--perf-dump-path` (Stage/Step Timing Dump)

Besides profiler traces, you can also dump a lightweight JSON report that contains:
- stage-level timing breakdown for the full pipeline
- step-level timing breakdown for the denoising stage (per diffusion step)

This is useful to quickly identify which stage dominates end-to-end latency, and whether denoising steps have uniform runtimes (and if not, which step has an abnormal spike).

The dumped JSON contains a `denoise_steps_ms` field formatted as an array of objects, each with a `step` key (the step index) and a `duration_ms` key.

Example:

```bash
sglang generate \
  --model-path <MODEL_PATH_OR_ID> \
  --prompt "<PROMPT>" \
  --perf-dump-path perf.json
```

## Nsight Systems

Nsight Systems provides low-level CUDA profiling with kernel details, register usage, and memory access patterns.

### Installation

See the [SGLang profiling guide](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/benchmark_and_profiling.md#profile-with-nsight) for installation instructions.

### Basic Profiling

Profile the entire pipeline execution:

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o QwenImage \
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --seed 0
```

### Targeted Stage Profiling

Use `--delay` and `--duration` to capture specific stages and reduce file size:

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --delay 10 \
  --duration 30 \
  -o QwenImage_denoising \
  sglang generate \
    --model-path Qwen/Qwen-Image \
    --prompt "A Logo With Bold Large Text: SGL Diffusion" \
    --seed 0
```

**Parameters:**
- `--delay N`: Wait N seconds before starting capture (skip initialization overhead)
- `--duration N`: Capture for N seconds (focus on specific stages)
- `--force-overwrite`: Overwrite existing output files

## Notes

- **Reduce trace size**: Use `--num-profiled-timesteps` with smaller values or `--delay`/`--duration` with Nsight Systems
- **Stage-specific analysis**: Use `--profile` alone for denoising stage, add `--profile-all-stages` for full pipeline
- **Multiple runs**: Profile with different prompts and resolutions to identify bottlenecks across workloads

## FAQ

- If you are profiling `sglang generate` with Nsight Systems and find that the generated profiler file did not capture any CUDA kernels, you can resolve this issue by increasing the model's inference steps to extend the execution time.

---

# Contributing to SGLang Diffusion

This guide outlines the requirements for contributing to the SGLang Diffusion module (`sglang.multimodal_gen`).

## 1. Commit Message Convention

We follow a structured commit message format to maintain a clean history.

**Format:**
```text
[diffusion] <scope>: <subject>
```

**Examples:**
- `[diffusion] cli: add --perf-dump-path argument`
- `[diffusion] scheduler: fix deadlock in batch processing`
- `[diffusion] model: support Stable Diffusion 3.5`

**Rules:**
- **Prefix**: Always start with `[diffusion]`.
- **Scope** (Optional): `cli`, `scheduler`, `model`, `pipeline`, `docs`, etc.
- **Subject**: Imperative mood, short and clear (e.g., "add feature" not "added feature").

## 2. Performance Reporting

For PRs that impact **latency**, **throughput**, or **memory usage**, you **should** provide a performance comparison report.

### How to Generate a Report

1.  **Baseline**: run the benchmark (for a single generation task)
    ```bash
    $ sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path baseline.json
    ```

2.  **New**: run the same benchmark, without modifying any server_args or sampling_params
    ```bash
    $ sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path new.json
    ```

3.  **Compare**: run the compare script, which will print a Markdown table to the console
    ```bash
    $ python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json new.json [new2.json ...]
    ### Performance Comparison Report
    ...
    ```
4. **Paste**: paste the table into the PR description

## 3. CI-Based Change Protection

Consider adding tests to the `pr-test` or `nightly-test` suites to safeguard your changes, especially for PRs that:

1. support a new model
2. support or fix important features
3. significantly improve performance

See [test](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/test) for examples

---

# How to Support New Diffusion Models

SGLang diffusion uses a modular pipeline architecture built around two key concepts:

- **`ComposedPipeline`**: Orchestrates `PipelineStage`s to define the complete generation process
- **`PipelineStage`**: Modular components (prompt encoding, denoising loop, VAE decoding, etc.)

To add a new model, you'll need to define:
1. **`PipelineConfig`**: Static model configurations (paths, precision settings)
2. **`SamplingParams`**: Runtime generation parameters (prompt, guidance_scale, steps)
3. **`ComposedPipeline`**: Chain together pipeline stages
4. **Modules**: Model components (text_encoder, transformer, vae, scheduler)

For the complete implementation guide with examples, see: **[How to Support New Diffusion Models](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/support_new_models.md)**

---

## References

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Cache-DiT](https://github.com/vipshop/cache-dit)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [xDiT](https://github.com/xdit-project/xDiT)
- [Diffusers](https://github.com/huggingface/diffusers)
