# SANA-WM Live Web UI

Interactive, camera-controlled streaming world model. Press **WASD** to move
(forward / left / back / right) and **IJKL** to look (up / left / down / right);
each keypress autoregressively generates the next video chunk in real time.

Each step runs, in-process and incrementally:
**stage-1 `forward_long`** (chunk-causal DiT, carried per-block KV cache) →
**chunked LTX-2 refiner** (`RefinerChunkRunner`, carried sink/history KV) →
**causal-VAE `decode_chunk`** → frames streamed to the browser over a websocket.

This is the same chunk-wise pipeline the upstream NVlabs streaming inference uses
(stage-1 DMD is coarse by design — the LTX-2 refiner is required for sharp output).

## Setup

```bash
# websockets (or wsproto) is REQUIRED — without it uvicorn rejects /ws with HTTP 403
# ("Unsupported upgrade request") while still serving the page, so the canvas shows
# "disconnected". The server checks for it at startup.
pip install fastapi uvicorn websockets pillow

# Assemble the streaming model dir (test-anas-smoke DiT/VAE/refiner + gemma-2-2b
# stage-1 text encoder). See build_model_dir.sh — in particular the refiner MUST
# point at test-anas-smoke refiner_diffusers + gemma3_12b, not the bidirectional one.
bash build_model_dir.sh /path/to/sana-wm-streaming-model
```

## Run

```bash
# Uses the bundled demo prompt + first frame (assets/) by default:
CUDA_VISIBLE_DEVICES=0 python -m sglang.multimodal_gen.apps.sana_wm_realtime.server \
    --model /path/to/sana-wm-streaming-model --port 8008
# open http://<host>:8008/  and press WASD / IJKL

# Or supply your own:
#   --image first_frame.png --prompt prompt.txt
```

The repo ships a demo prompt + first frame in `assets/` (`demo_prompt.txt`,
`demo_first_frame.png`), so `--image`/`--prompt` are optional.

Notes:
- Loads the 2.6B DiT + LTX-2 refiner + Gemma encoders (~80 GB GPU); startup takes a
  bit. Add `--no-refiner` to decode coarse stage-1 only (faster, lower quality).
- One GPU step per keypress (a few hundred ms–seconds); a held key auto-advances.

## Backend

`SanaWMRealtimeEngine`
(`runtime/pipelines_core/stages/model_specific_stages/sana_wm_realtime_engine.py`)
loads the streaming two-stage pipeline, bootstraps `SanaWMRealtimeSession`, and
exposes `reset(prompt, image)` + `step(keys)`. The websocket server here is a thin
transport over it.
