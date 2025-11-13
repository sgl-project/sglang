# SGLang Diffusion CLI Inference

The sgl-diffusion CLI provides a quick way to access the sgl-diffusion inference pipeline for image and video generation.

## Prerequisites

- A working sgl-diffusion installation and the `sgl-diffusion` CLI available in `$PATH`.
- Python 3.10+ if you plan to use the OpenAI Python SDK.


## Supported Arguments

### Server Arguments

- `--model-path {MODEL_PATH}`: Path to the model or model ID
- `--num-gpus {NUM_GPUS}`: Number of GPUs to use
- `--tp-size {TP_SIZE}`: Tensor parallelism size (only for the encoder; should not be larger than 1 if text encoder offload is enabled, as layer-wise offload plus prefetch is faster)
- `--sp-size {SP_SIZE}`: Sequence parallelism size (typically should match the number of GPUs)
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
```
Once the generation task has finished, the server will shut down automatically.

> [!NOTE]
> The HTTP server-related arguments are ignored in this subcommand.


## Serve

Launch the sgl-diffusion HTTP server and interact with it using the OpenAI SDK and curl. The server implements an OpenAI-compatible subset for Videos under the `/v1/videos` namespace.

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

Wait until the port is listening. In CI, the tests probe `127.0.0.1:30010` before sending requests.

---

## Video Generation

Initialize the client with a dummy API key and point `base_url` to your local server:

```python
from openai import OpenAI

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1")
```

- **Create a video**

```python
video = client.videos.create(prompt="A calico cat playing a piano on stage", size="1280x720")
print(video.id, video.status)
```

Response example fields include `id`, `status` (e.g., `queued` → `completed`), `size`, and `seconds`.

- **List videos**

```python
videos = client.videos.list()
for item in videos.data:
    print(item.id, item.status)
```

- **Poll for completion and download content**

```python
import time

video = client.videos.create(prompt="A calico cat playing a piano on stage", size="1280x720")
video_id = video.id

# Simple polling loop
while True:
    page = client.videos.list()
    item = next((v for v in page.data if v.id == video_id), None)
    if item and item.status == "completed":
        break
    time.sleep(5)

# Download binary content (MP4)
resp = client.videos.download_content(video_id=video_id)
content = resp.read()  # bytes
with open("output.mp4", "wb") as f:
    f.write(content)
```

### curl examples

- **Create a video**

```bash
curl -sS -X POST "http://localhost:30010/v1/videos" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -d '{
        "prompt": "A calico cat playing a piano on stage",
        "size": "1280x720"
      }'
```

- **List videos**

```bash
curl -sS -X GET "http://localhost:30010/v1/videos" \
  -H "Authorization: Bearer sk-proj-1234567890"
```

- **Download video content**

```bash
curl -sS -L "http://localhost:30010/v1/videos/<VIDEO_ID>/content" \
  -H "Authorization: Bearer sk-proj-1234567890" \
  -o output.mp4
```

#### API surface implemented here

The server exposes these endpoints (OpenAPI tag `videos`):

- `POST /v1/videos` — Create a generation job and return a queued `video` object.
- `GET /v1/videos` — List jobs.
- `GET /v1/videos/{video_id}/content` — Download binary content when ready (e.g., MP4).

#### Reference

- OpenAI Videos API reference: `https://platform.openai.com/docs/api-reference/videos`

---

## Image Generation

SGLang Diffusion supports image generation and editing with models such as **Qwen-Image**, **Qwen-Image-Edit**, and **FLUX**. You can use these either through the CLI or via an OpenAI-compatible HTTP API.

### CLI Usage

##### `T2I` Text → Image (Qwen-Image)

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A curious raccoon" \
  --width 720 \
  --height 720 \
  --save-output \
  --output-path outputs
```

##### `TI2I` Image → Image (Edit) (Qwen-Image-Edit)

```bash
sglang generate \
  --model-path Qwen/Qwen-Image-Edit \
  --prompt "Convert 2D style to 3D style" \
  --image-path path/to/input.jpg \
  --width 1024 \
  --height 1536 \
  --save-output \
  --output-path outputs
```

These commands create a temporary diffusion job, save the output image under `--output-path`, and exit after generation.

---

### HTTP API Server (OpenAI-Compatible Schema)

You can also serve image models and interact with them using OpenAI-style endpoints.

#### Start a Local Server

```bash
sglang serve --model-path Qwen/Qwen-Image #T2I
# or
sglang serve --model-path Qwen/Qwen-Image-Edit #TI2I
```

This launches a FastAPI server (default: `http://localhost:3000/v1`). You can then send requests via OpenAI Python SDK or CURL:

---

#### Sending Requests

##### `T2I` Text → Image (OpenAI Python SDK)

```python
import base64
from openai import OpenAI

client = OpenAI(
    api_key="sglang-anything",
    base_url="http://localhost:3000/v1",
)

resp = client.images.generate(
    model="Qwen/Qwen-Image",                # or "black-forest-labs/FLUX.1-dev"
    prompt="A cute raccoon hacking a GPU cluster at night",
    size="1024x1024",
    n=1,
    response_format="b64_json",             # required; "url" not supported
)

b64 = resp.data[0].b64_json
img = base64.b64decode(b64)
with open("meme.png", "wb") as f:
    f.write(img)

print("Saved meme.png")
```

---

##### `T2I` Text → Image (curl)

```bash
export OPENAI_API_BASE="http://localhost:3000/v1"
export OPENAI_API_KEY="sglang-anything"

curl -sS -X POST "$OPENAI_API_BASE/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen/Qwen-Image",
        "prompt": "A cute raccoon hacking a GPU cluster at night",
        "size": "1024x1024",
        "n": 1,
        "response_format": "b64_json"
      }' \
  | jq -r '.data[0].b64_json' | base64 --decode > meme.png
```

The response follows the standard OpenAI Images JSON format:

```json
{
  "created": 1234567890,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAA..."
    }
  ]
}
```

You can decode the `b64_json` field to retrieve the generated image.

---

##### `TI2I` Image Edit (Python SDK)

```python
import base64
from openai import OpenAI

client = OpenAI(
    api_key="sglang-anything",
    base_url="http://localhost:3000/v1",
)

with open("input.png", "rb") as f:
    image_bytes = f.read()

resp = client.images.edit(
    model="Qwen/Qwen-Image-Edit",
    image=image_bytes,
    prompt="Change the text to CUDA FOR LIFE",
    size="1024x1024",
    response_format="b64_json",             # required
)

b64 = resp.data[0].b64_json
img = base64.b64decode(b64)
with open("meme.png", "wb") as f:
    f.write(img)

print("Saved meme.png")
```
---

##### `TI2I` Image Edit (curl)

```bash
export OPENAI_API_BASE="http://localhost:3000/v1"
export OPENAI_API_KEY="sglang-anything"

curl -sS -X POST "$OPENAI_API_BASE/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "model=Qwen/Qwen-Image-Edit" \
  -F "image[]=@input.png" \
  -F "prompt=Change the text to CUDA FOR LIFE" \
  -F "response_format=b64_json" \
  | jq -r '.data[0].b64_json' | base64 --decode > meme.png
```

---

> [!NOTE]
> Only `response_format="b64_json"` is currently supported.
> Using `response_format="url"` will return the following error:
> ```
> 400 - {"detail": "response_format=url is not supported"}
> ```
