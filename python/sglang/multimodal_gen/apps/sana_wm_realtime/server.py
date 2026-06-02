"""SANA-WM Live Web UI — FastAPI websocket server.

A single SanaWMRealtimeEngine (the 2.6B camera-controlled streaming world model
loaded in-process) is driven interactively: the browser sends WASD/IJKL keypresses
over a websocket; each one runs engine.step(keys) -> one autoregressive chunk
(stage-1 forward_long with carried KV cache -> chunked LTX-2 refiner -> causal-VAE
decode), streamed back as JPEG frames and played on a canvas at ~16 fps.
WASD = move (fwd/left/back/right), IJKL = look (up/left/down/right).

Run (needs a GPU + the assembled streaming model dir; see build_model_dir.sh):
    python -m sglang.multimodal_gen.apps.sana_wm_realtime.server \
        --model /path/to/sana-wm-streaming-model \
        --image first_frame.png --prompt prompt.txt --port 8008
Then open http://<host>:8008/ and press WASD/IJKL.

Deps: pip install fastapi uvicorn websockets pillow
"""
from __future__ import annotations
import argparse
import asyncio
import io
import json
import os
from pathlib import Path

import numpy as np

_HTML = (Path(__file__).parent / "index.html").read_text()
_engine = None
_lock = asyncio.Lock()  # serialize GPU access (one step at a time)


def _build_app():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from PIL import Image

    app = FastAPI()

    @app.get("/")
    async def index():
        return HTMLResponse(_HTML)

    def _jpegs(frames: np.ndarray) -> list[bytes]:
        out = []
        for f in frames:  # (T,H,W,3) uint8
            buf = io.BytesIO()
            Image.fromarray(f).save(buf, format="JPEG", quality=85)
            out.append(buf.getvalue())
        return out

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        loop = asyncio.get_event_loop()
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                if msg.get("action") == "reset":
                    async with _lock:
                        await loop.run_in_executor(
                            None, _engine.reset, _engine._prompt, _engine._image,
                            _engine._intrinsics)
                    await sock.send_text(json.dumps({"type": "reset_done"}))
                    continue
                keys = msg.get("keys", "w")
                async with _lock:
                    frames = await loop.run_in_executor(None, _engine.step, keys)
                jpegs = _jpegs(frames)
                await sock.send_text(json.dumps(
                    {"type": "chunk_begin", "n": len(jpegs), "keys": keys}))
                for j in jpegs:
                    await sock.send_bytes(j)
                await sock.send_text(json.dumps({"type": "chunk_end"}))
        except WebSocketDisconnect:
            return

    return app


def main():
    global _engine
    ap = argparse.ArgumentParser(description="SANA-WM Live Web UI server")
    _assets = Path(__file__).parent / "assets"
    ap.add_argument("--model", required=True, help="assembled streaming model dir")
    ap.add_argument("--image", default=str(_assets / "demo_first_frame.png"),
                    help="first-frame conditioning image (default: bundled demo)")
    ap.add_argument("--prompt", default=str(_assets / "demo_prompt.txt"),
                    help="prompt text or path to a .txt (default: bundled demo)")
    ap.add_argument("--intrinsics", default=str(_assets / "demo_intrinsics.npy"),
                    help="camera intrinsics .npy (3x3 or 4-vec) for the source image; "
                         "default: bundled demo. Pass '' to use heuristic centered intrinsics.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--no-refiner", action="store_true",
                    help="decode coarse stage-1 only (faster, lower quality)")
    args = ap.parse_args()

    # uvicorn silently serves HTTP but rejects /ws with HTTP 403 ("Unsupported
    # upgrade request") if NO WebSocket library is installed. Fail loudly here,
    # before the slow model load, instead of as a cryptic 403 at connect time.
    import importlib.util
    if (importlib.util.find_spec("websockets") is None
            and importlib.util.find_spec("wsproto") is None):
        raise SystemExit(
            "[sana_wm_realtime] No WebSocket library found — uvicorn would reject "
            "/ws with HTTP 403.\n  Fix:  pip install websockets")

    import uvicorn
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm_realtime_engine import (
        SanaWMRealtimeEngine,
    )

    prompt = open(args.prompt).read().strip() if os.path.isfile(args.prompt) else args.prompt
    print("Loading SANA-WM realtime engine (loads the 2.6B model + LTX-2 refiner)...", flush=True)
    _engine = SanaWMRealtimeEngine(
        model_path=args.model, height=args.height, width=args.width,
        use_refiner=not args.no_refiner)
    _engine.reset(prompt, args.image, intrinsics=(args.intrinsics or None))
    print(f"Engine ready. Open http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(_build_app(), host=args.host, port=args.port, ws_max_size=64 * 1024 * 1024)


if __name__ == "__main__":
    main()
