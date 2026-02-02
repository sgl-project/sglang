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
