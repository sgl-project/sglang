"""
Smoke tests for /v1/models and /v1/chat/completions (streaming + non-streaming).
"""

from __future__ import annotations

import httpx
import pytest

MODEL = "Qwen/Qwen3-0.6B"


def test_models(router: str) -> None:
    """GET /v1/models must list the configured model."""
    resp = httpx.get(f"{router}/v1/models", timeout=30)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    ids = [m["id"] for m in data.get("data", [])]
    assert any(
        MODEL in mid for mid in ids
    ), f"Model {MODEL!r} not found in /v1/models response: {ids}"


def test_chat_non_streaming(router: str) -> None:
    """POST /v1/chat/completions (stream=False) returns an assistant message."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 10,
        "stream": False,
    }
    resp = httpx.post(f"{router}/v1/chat/completions", json=payload, timeout=60)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"], "Expected non-empty assistant content"


def test_chat_streaming(router: str) -> None:
    """POST /v1/chat/completions (stream=True) returns >=2 SSE chunks incl. [DONE]."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 10,
        "stream": True,
    }
    chunks: list[str] = []
    with httpx.stream(
        "POST",
        f"{router}/v1/chat/completions",
        json=payload,
        timeout=60,
    ) as resp:
        assert resp.status_code == 200, resp.read().decode()
        for line in resp.iter_lines():
            line = line.strip()
            if line.startswith("data:"):
                chunks.append(line)

    assert len(chunks) >= 2, f"Expected >=2 SSE chunks, got {len(chunks)}: {chunks}"
    assert any(
        "[DONE]" in c for c in chunks
    ), f"No [DONE] chunk found in SSE stream: {chunks}"
