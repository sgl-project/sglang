"""Minimal fake SGLang worker for kind E2E integration testing.

Responds to:
  GET  /health                   -> {"status": "ok"}
  GET  /server_info              -> {"served_model_name": MODEL_ID}
  GET  /v1/models                -> list with a single MODEL_ID model entry
  POST /v1/chat/completions      -> echoes the last user message back
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()

MODEL_ID = os.environ.get("MODEL_ID", "tiny")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/server_info")
async def server_info():
    # The sgl-router worker manager fetches this on every Added event and
    # uses `served_model_name` to populate the registry's model index.
    return {"served_model_name": MODEL_ID}


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
    uvicorn.run(app, host="0.0.0.0", port=30000)
