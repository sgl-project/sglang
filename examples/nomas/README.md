# Hy-Nomas examples

Use Linux, NVIDIA CUDA and SGLang's diffusion dependencies. The native pipeline
runs the distilled, non-quantized DiT, Qwen2.5-VL-7B text encoder and PixelUMM_FT
decoder. PE is a separate service and is not included.

## Load official weights

Pass `tencent/Hy-Nomas` to `--model-path` or `DiffGenerator.from_pretrained`.
The built-in [model overlay](https://huggingface.co/Niehen6174/Hy-Nomas-overlay)
automatically downloads the pinned official checkpoints and text encoder,
links the DiT/text weights and exports PixelUMM's decoder without changing
its tensor values or dtypes. It caches the assembled pipeline under
`SGLANG_DIFFUSION_CACHE_ROOT`; subsequent starts reuse it. There is no manual
preparation command. Keep source files available if the cache uses symlinks.
The first load needs additional disk space for the decoder export.

An already-prepared local pipeline directory remains supported. This overlay
does not load PE, LoRA adapters or FP8/INT8 variants. Use upstream-format JSON
captions for trained weights; plain captions are only a smoke-test shortcut.

## Offline generation

```python
import json
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

# Each event has a positive pixel-frame count and a structured JSON caption.
with open("request.json") as f:
    request = json.load(f)

with DiffGenerator.from_pretrained(
    model_path="tencent/Hy-Nomas", num_gpus=1, warmup_mode="off"
) as generator:
    result = generator.generate(request)
    print(result.output_file_path)
```

The request can specify `segments=[{"frames": 120, "caption": "..."}, ...]`,
`prompt_format="json"`, `max_sequence_length=1280`, `num_inference_steps=8`,
`guidance_scale=1`, `seed=42`, and `save_output=True`. `num_frames` must equal
the sum of event frames; `prompt` contains the first caption. The spatial
alignment maps a requested 720×1280 output to 704×1280.

## Online service

```bash
sglang serve --model-path tencent/Hy-Nomas \
  --host 127.0.0.1 --port 30000 --num-gpus 1 --warmup-mode off
```

The HTTP `/v1/videos` endpoint accepts a JSON request, returns a job ID, and
provides status and video download endpoints. Nomas fixes HTTP and WebSocket
noise generation to CPU to preserve the model's sampling semantics, overriding
the general video API's CUDA-generator default:

```bash
curl http://127.0.0.1:30000/v1/videos \
  -H 'Content-Type: application/json' --data-binary @request.json
curl http://127.0.0.1:30000/v1/videos/VIDEO_ID
curl http://127.0.0.1:30000/v1/videos/VIDEO_ID/content --output video.mp4
```

For WebSocket event streaming, use the existing client:

```bash
python examples/nomas/realtime_client.py --url ws://127.0.0.1:30000 \
  --prompt-format json \
  --segment "120:$(cat first-caption.json)" \
  --segment "120:$(cat second-caption.json)" \
  --size 1280x704 --steps 8 --seed 42 --output ./nomas-session
```

The client sends msgpack messages to `/v1/realtime_video/generate` and saves raw
RGB frames plus a video. A custom interactive client may send each subsequent
`segment` event after receiving the previous chunk, then send `end` to flush
the withheld decoder tail. Event frame counts need not be divisible by four.
The service currently admits one active WebSocket session; a second connection
is rejected. Disconnecting ends the session; there is no reconnect/resume
protocol. Online generation uses a persistent noise bank and session KV, so its
pixels should be compared with another online run, not an offline request.

For two GPUs, replace `--num-gpus 1` with `--num-gpus 2 --tp-size 2` or
`--num-gpus 2 --sp-degree 2`. These shard the DiT computation. They do not place
the text encoder and decoder on a separate GPU. Nomas has not integrated the
framework's EPD service roles.

## Development tests

This directory contains the realtime client and usage examples. Random-weight
fixtures, regression commands and measured validation limits are documented in
[the Nomas test guide](../../python/sglang/multimodal_gen/test/scripts/README.nomas.md).
Single-GPU and SP=2 videos passed the recorded checks; TP=2 completes but its
trained-weight precision relative to single GPU is not yet certified.
