---
title: HY-Nomas
description: "Serve the HY-Nomas event-conditioned video world model with SGLang-diffusion, as offline video jobs or a realtime WebSocket session."
metatags:
    description: "Serve the HY-Nomas event-conditioned video world model with SGLang-diffusion, as offline video jobs or a realtime WebSocket session."
tag: EXPERIMENTAL
---

import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';

<DiffusionModelTags tags={["world model", "event-conditioned video", "segment-autoregressive", "realtime", "camera control"]} />

import { Deployment } from '/src/snippets/_deployment.jsx';
import { config } from '/src/snippets/configs/HY-Nomas/hy-nomas.jsx';

## 1. Quick start

Follow the [SGLang Diffusion installation guide](/docs/sglang-diffusion/installation) on Linux with NVIDIA CUDA.

One `sglang serve` process answers both HTTP video jobs and realtime WebSocket sessions. On first load, SGLang downloads the official checkpoints and assembles a local pipeline; see [Weights](#weights).

<Deployment config={config} />

## 2. Model capabilities

[HY-Nomas](https://huggingface.co/tencent/Hy-Nomas) is Tencent Hunyuan's video world model. A video is described as a sequence of events: each event is a structured JSON caption with a frame count and optional camera poses. The model generates one event at a time while attending to the cached state of earlier events, so a scene can be extended or redirected without regenerating what came before.

Choose it for long, steerable scene continuation, either as one offline request with all events planned up front or as a live session that receives events one by one. It generates from noise: there is no image-to-video input. Captions must follow the official event JSON schema, which the official prompt enhancer (Nomas-PE) produces from free text; SGLang does not host that enhancer.

The code and weights are released under Apache-2.0. See the [GitHub repository](https://github.com/Tencent-Hunyuan/Hy-Nomas) for the event schema and training details.

| Component | Implementation |
| --- | --- |
| Text encoder | Qwen2.5-VL-7B-Instruct, 1280 tokens per event |
| DiT | Distilled segment-autoregressive DiT, 8 Euler steps, no CFG |
| Decoder | PixelUMM pixel decoder, windowed over 10 latents with 5-latent overlap |
| Camera | Optional normalized extrinsics, one matrix per latent-aligned frame |

## 3. Deployment

```bash Command
sglang serve --model-path tencent/Hy-Nomas --num-gpus 1 --port 30000
```

For two GPUs, use Ulysses sequence parallelism. Its output matched the single-GPU output pixel for pixel:

```bash Command
sglang serve --model-path tencent/Hy-Nomas --num-gpus 2 --sp-degree 2 --port 30000
```

Tensor parallelism (`--tp-size 2`) also runs, but its trained-weight output differs from one GPU, so prefer SP. `tp_size * sp_degree` must divide the 16 attention heads, and TP and SP can be combined. Ring attention, CFG parallelism and data parallelism are rejected. Both layouts shard only the DiT computation; the text encoder and decoder are not placed on a separate GPU.

To reduce DiT memory, stream its blocks from host memory:

```bash Command
sglang serve --model-path tencent/Hy-Nomas --num-gpus 1 --dit-layerwise-offload true --port 30000
```

### Weights

`tencent/Hy-Nomas` is resolved through a pinned [model overlay](https://huggingface.co/Niehen6174/Hy-Nomas-overlay). It downloads the official distilled, non-quantized DiT, the PixelUMM decoder and the Qwen2.5-VL text encoder. It then caches a Diffusers-style pipeline under `SGLANG_DIFFUSION_CACHE_ROOT`. Weights keep their original values and dtypes. Later starts reuse the cache, and a local directory with the same layout can be passed to `--model-path` instead. The first load needs extra disk space for the decoder export.

The overlay does not include Nomas-PE, LoRA adapters, or FP8/INT8 variants.

## 4. Offline and HTTP generation

A request lists its events in `segments`. Each caption is a JSON string in the official event schema; later events usually describe only what changes. This two-event request makes 240 frames at 352×640 and saves it as `request.json`:

```python Python
import json

first = {
    "Style": "Cinematic fantasy",
    "perspective_view": "Third-person view",
    "subject_list": [{"subject_id": "fox", "display_name": "a small orange fox",
                      "subject_attribute": "Orange fur, white chest, and a fluffy tail."}],
    "segments": [{
        "visual_description_long": "A small orange fox stands on a stone path in a luminous forest.",
        "contained_actions": [{"start_time": "00:00.000", "end_time": "00:05.000",
                               "action_description": "The fox walks slowly along the path."}],
    }],
}
second = {
    "segments": [{
        "description_change": "Glowing blue flowers open beside the path.",
        "contained_actions": [{"start_time": "00:00.000", "end_time": "00:05.000",
                               "action_description": "The fox passes the flowers and looks around."}],
    }],
}
request = {
    "prompt": json.dumps(first),
    "prompt_format": "json",
    "segments": [
        {"frames": 120, "caption": json.dumps(first)},
        {"frames": 120, "caption": json.dumps(second)},
    ],
    "num_frames": 240,
    "height": 352,
    "width": 640,
    "fps": 24,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 42,
}
with open("request.json", "w") as f:
    json.dump(request, f)
```

Send it to the HTTP video API, then poll `GET /v1/videos/{id}` and download from `GET /v1/videos/{id}/content`:

```bash Command
curl -sS http://127.0.0.1:30000/v1/videos \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

The same request runs without a server through the Python API:

```python Python
from sglang.multimodal_gen import DiffGenerator

with DiffGenerator.from_pretrained(model_path="tencent/Hy-Nomas") as generator:
    result = generator.generate({**request, "save_output": True})
```

| Field | Meaning |
| --- | --- |
| `segments` | One entry per event: `frames` and a JSON `caption`, optionally `norm_extrinsics`. Event frames must sum to `num_frames`, and `prompt` repeats the first caption. |
| `prompt_format` | `json` applies the official event schema. `plain` accepts free text for smoke tests only and logs a warning. |
| `norm_extrinsics` | Camera matrices that are already normalized. The server does not normalize raw poses. |
| `num_inference_steps`, `guidance_scale` | The distilled checkpoint uses 8 steps and guidance 1. |
| `height`, `width` | Rounded to multiples of 32 as in the official code; 720×1280 becomes 704×1280. |

Event frame counts do not need to be multiples of four; the decoded video is cropped to each requested count. Sampling noise is always drawn on the CPU, so a fixed seed gives the same video through the Python API and HTTP.

## 5. Realtime session

Open `ws://HOST:PORT/v1/realtime_video/generate` and exchange MessagePack maps.

| Message | Direction | Content |
| --- | --- | --- |
| `{"type": "init", "prompt": ..., "num_frames": ..., "size": "640x352", "num_inference_steps": 8, "condition_inputs": {"prompt_format": "json", "segments": [...]}}` | client | Starts the session. Without `segments`, `prompt` and `num_frames` form the first event. |
| `{"type": "event", "kind": "segment", "event_id": n, "payload": {"caption": ..., "frames": ..., "norm_extrinsics": ...}}` | client | Queues the next event. `norm_extrinsics` covers this event only. |
| `{"type": "event", "kind": "end"}` | client | No more events. The server sends the remaining frames and closes. |
| `frame_batch` / `frame_batch_header` | server | Raw RGB, WebP or JPEG frames for one chunk. A chunk may contain no frames. |

Each event is generated as soon as it arrives, against the session's cached history. The decoder streams in overlapping windows and sends a frame only when no later window can change it, so frames are never revised. Output therefore lags generation, and `end` releases the tail: a 120 + 120 frame session sends 84, 120 and then 36 frames.

The server keeps one session at a time and rejects a second connection. Disconnecting ends the session and frees its state; there is no resume. Because a session draws noise from a persistent bank, compare its frames with another session using the same seed, not with an offline request.

A minimal client sends the first event in `init`, the rest as `segment` events, then `end`, and collects raw RGB frames until the server closes the socket:

```python Python
import msgspec
import numpy as np
import websocket

events = request["segments"]  # from the request above
ws = websocket.create_connection("ws://127.0.0.1:30000/v1/realtime_video/generate")
ws.send_binary(msgspec.msgpack.encode({
    "type": "init",
    "prompt": events[0]["caption"],
    "num_frames": events[0]["frames"],
    "size": "640x352",
    "fps": 24,
    "num_inference_steps": 8,
    "seed": 42,
    "realtime_output_format": "raw",
    "condition_inputs": {"prompt_format": "json", "segments": events[:1]},
}))
for event_id, event in enumerate(events[1:], start=1):
    ws.send_binary(msgspec.msgpack.encode(
        {"type": "event", "kind": "segment", "event_id": event_id, "payload": event}))
ws.send_binary(msgspec.msgpack.encode({"type": "event", "kind": "end"}))

frames = []
while True:
    try:
        raw = ws.recv()
    except websocket.WebSocketConnectionClosedException:
        break
    if not isinstance(raw, bytes):
        break
    message = msgspec.msgpack.decode(raw)
    if message.get("type") == "error":
        raise RuntimeError(message)
    if message.get("type") in ("frame_batch", "frame_batch_header"):
        # Large batches send their payload as the next binary message.
        payload = message["payload"] if message["type"] == "frame_batch" else ws.recv()
        if message["num_frames"]:
            shape = (message["num_frames"], message["height"], message["width"], message["channels"])
            frames.append(np.frombuffer(payload, dtype=np.uint8).reshape(shape))
video = np.concatenate(frames)  # 240 frames
```

To steer from free text, run Nomas-PE as a separate service. It turns the user's text and recent frames into the next event JSON, which the client then sends as the `caption`.

### KV cache budget

The session cache starts at 65,536 tokens and grows by 1.5× up to 32 GiB, or to free GPU memory minus 8 GiB, whichever is smaller. At that limit, the oldest event is evicted first, as in the official code. Setting `max_cache_segments` also limits the number of cached events. Positions keep counting across evictions. If a cache write fails after an eviction, the session must be restarted.

These limits are `NomasPipelineConfig` fields (`interactive_kv_cache_initial_tokens`, `interactive_kv_cache_growth_factor`, `max_cache_gib`, `cache_free_memory_reserve_gib`, `max_cache_segments`). At 1280×704, one 10-latent event uses about 4.5 GB of K/V on one GPU; TP and SP divide this across ranks.

## 6. Limitations

- No image-to-video, classifier-free guidance, or raw camera-pose normalization.
- Nomas-PE is not served by SGLang.
- One realtime session per server; Ring attention, KV CPU swap, INT8 KV and sliding-window history are not implemented.
- TP output is not yet equivalent to single-GPU output with trained weights.
- Validation was on H200 only.
