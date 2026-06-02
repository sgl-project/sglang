"""
fm_mistral.py — FractalMesh Mistral AI Bridge (port 7911)
Direct Mistral API integration (mistralai SDK) as an alternative to OpenRouter routing.
Exposes /chat, /embed, /models, /health endpoints.
Samuel James Hiotis | ABN 56 628 117 363
"""

import logging
import os
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS
from waitress import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fm-mistral")

PORT = int(os.getenv("MISTRAL_BRIDGE_PORT", "7911"))
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
ADMIN_SECRET: str = os.getenv("ADMIN_SECRET", "")

app = Flask(__name__)
CORS(app)


def _mistral_client():
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY not set in vault")
    from mistralai import Mistral  # mistralai>=1.0

    return Mistral(api_key=MISTRAL_API_KEY)


# ── Auth ──────────────────────────────────────────────────────────────────────


def _auth_ok() -> bool:
    import hmac

    secret = request.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET:
        return True
    return hmac.compare_digest(ADMIN_SECRET, secret)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "agent": "fm-mistral",
            "port": PORT,
            "model": MISTRAL_MODEL,
            "key_set": bool(MISTRAL_API_KEY),
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
    )


@app.get("/models")
def list_models():
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        client = _mistral_client()
        result = client.models.list()
        models = [m.id for m in result.data] if hasattr(result, "data") else []
        return jsonify({"models": models, "count": len(models)})
    except Exception as exc:
        logger.exception("models list failed")
        return jsonify({"error": str(exc)}), 502


@app.post("/chat")
def chat():
    """
    POST /chat
    Body: {"messages": [...], "model": "mistral-large-latest", "temperature": 0.7,
           "max_tokens": 1024, "stream": false}
    """
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "messages array required"}), 400

    model = body.get("model", MISTRAL_MODEL)
    temperature = float(body.get("temperature", 0.7))
    max_tokens = int(body.get("max_tokens", 1024))

    try:
        client = _mistral_client()
        t0 = time.monotonic()
        resp = client.chat.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = round(time.monotonic() - t0, 3)
        choice = resp.choices[0]
        return jsonify(
            {
                "content": choice.message.content,
                "model": resp.model,
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
                "finish_reason": choice.finish_reason,
                "elapsed_s": elapsed,
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
    except Exception as exc:
        logger.exception("chat failed")
        return jsonify({"error": str(exc)}), 502


@app.post("/embed")
def embed():
    """
    POST /embed
    Body: {"inputs": ["text1", "text2"], "model": "mistral-embed"}
    """
    if not _auth_ok():
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    inputs = body.get("inputs")
    if not inputs or not isinstance(inputs, list):
        return jsonify({"error": "inputs array required"}), 400

    model = body.get("model", "mistral-embed")
    try:
        client = _mistral_client()
        resp = client.embeddings.create(model=model, inputs=inputs)
        return jsonify(
            {
                "embeddings": [d.embedding for d in resp.data],
                "model": resp.model,
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
                "ts": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
    except Exception as exc:
        logger.exception("embed failed")
        return jsonify({"error": str(exc)}), 502


if __name__ == "__main__":
    logger.info("[fm-mistral] Listening on :%d  model=%s", PORT, MISTRAL_MODEL)
    serve(app, host="0.0.0.0", port=PORT, threads=4)
