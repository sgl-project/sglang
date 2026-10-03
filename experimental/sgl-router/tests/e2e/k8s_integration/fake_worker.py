"""Minimal fake SGLang worker for kind E2E integration testing.

Responds to:
  GET  /health                   -> {"status": "ok"}
  GET  /server_info              -> {"served_model_name": MODEL_ID, ...}
  GET  /v1/models                -> list with a single MODEL_ID model entry
  POST /v1/chat/completions      -> echoes the last user message back, plus
                                    the HTTP version the request arrived on

Set `FAKE_WORKER_HTTP2=1` to imitate an engine launched with
`--enable-http2`: `/server_info` advertises the flag and the app is served by
Granian in `HTTPModes.auto`, which is what the real engine runs, so one port
serves cleartext h2c alongside HTTP/1.1. The default is uvicorn, which speaks
HTTP/1.1 only.

`x_http_version` on the chat response is the load-bearing part for
`test_h2c_forwarding.py`: a chat completion returns 200 over either protocol,
so without the worker reporting what it actually received, an h2c test passes
whether or not h2c was used.
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()

MODEL_ID = os.environ.get("MODEL_ID", "tiny")
# Normalised before comparing: a manifest that writes the Python-idiomatic
# "False" must not silently turn this worker into a Granian/h2c one, which
# would fail test_h2c_forwarding with a message about a router bug.
ENABLE_HTTP2 = os.environ.get("FAKE_WORKER_HTTP2", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
    "off",
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/server_info")
async def server_info():
    # The sgl-router worker manager fetches this on every Added event and
    # uses `served_model_name` to populate the registry's model index, and
    # `enable_http2` to resolve the worker's forwarding protocol.
    info = {"served_model_name": MODEL_ID}
    if ENABLE_HTTP2:
        info["enable_http2"] = True
    return info


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "sglang",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages = payload.get("messages", [])
    last_content = messages[-1]["content"] if messages else ""
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": payload.get("model", MODEL_ID),
        # Non-standard, and deliberately so: the router returns the upstream
        # body verbatim (`proxy::forward_*` hands back `resp.bytes()`), so this
        # is how a test on the other side of the router learns which protocol
        # the forward leg actually used. "1.1" or "2".
        "x_http_version": request.scope.get("http_version"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"echo: {last_content}",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


if __name__ == "__main__":
    if ENABLE_HTTP2:
        # Mirrors the engine's own server (`_run_granian_server` in
        # sglang/srt/entrypoints/http_server.py): HTTPModes.auto dispatches per
        # connection on the first bytes, so h2c prior-knowledge and HTTP/1.1
        # share one cleartext port.
        from granian import Granian
        from granian.constants import HTTPModes, Interfaces

        Granian(
            target="fake_worker:app",
            address="0.0.0.0",
            port=30000,
            interface=Interfaces.ASGI,
            http=HTTPModes.auto,
        ).serve()
    else:
        uvicorn.run(app, host="0.0.0.0", port=30000)
