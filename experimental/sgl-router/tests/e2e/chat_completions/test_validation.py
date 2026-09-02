"""Basic chat-completions correctness — ported from SMG's
``e2e_test/chat_completions/test_validation.py``, narrowed to the
subset that exercises sgl-router (not SMG's per-message validators).

The shape:
  - single-worker regular-mode router
  - non-streaming + streaming chat completion
  - assistant message non-empty, role correct, finish_reason set

These quick checks run first; if they pass, the heavier
multi-worker acceptance tests are worth running.
"""

from __future__ import annotations

import httpx
import pytest
from infra.gateway import Gateway
from infra.model_pool import spawn_worker
from infra.model_specs import get_model_spec


@pytest.mark.real_gpu
def test_chat_non_streaming_returns_assistant_message(
    router_binary,  # noqa: ARG001
    gpu_allocator,
):
    gpu = gpu_allocator.acquire(1)
    try:
        with spawn_worker("qwen3-0.6b", gpu_ids=gpu) as worker:
            spec = get_model_spec("qwen3-0.6b")
            with Gateway() as gw:
                gw.start_regular(
                    model_id=spec["model"],
                    tokenizer_path=spec["model"],
                    worker_urls=[worker.url],
                    timeout=120.0,
                )
                resp = httpx.post(
                    f"{gw.base_url}/v1/chat/completions",
                    json={
                        "model": spec["model"],
                        "messages": [{"role": "user", "content": "Say hi."}],
                        "max_tokens": 16,
                        "stream": False,
                    },
                    timeout=60.0,
                )
                assert resp.status_code == 200, resp.text
                body = resp.json()
                choice = body["choices"][0]
                assert choice["message"]["role"] == "assistant"
                assert choice["message"][
                    "content"
                ], f"empty assistant content: {choice!r}"
                assert choice.get("finish_reason"), choice
    finally:
        gpu_allocator.release(gpu)


@pytest.mark.real_gpu
def test_chat_streaming_emits_sse_chunks_with_done(
    router_binary,  # noqa: ARG001
    gpu_allocator,
):
    gpu = gpu_allocator.acquire(1)
    try:
        with spawn_worker("qwen3-0.6b", gpu_ids=gpu) as worker:
            spec = get_model_spec("qwen3-0.6b")
            with Gateway() as gw:
                gw.start_regular(
                    model_id=spec["model"],
                    tokenizer_path=spec["model"],
                    worker_urls=[worker.url],
                    timeout=120.0,
                )
                chunks: list[str] = []
                with httpx.stream(
                    "POST",
                    f"{gw.base_url}/v1/chat/completions",
                    json={
                        "model": spec["model"],
                        "messages": [{"role": "user", "content": "Say hi."}],
                        "max_tokens": 16,
                        "stream": True,
                    },
                    timeout=60.0,
                ) as resp:
                    assert resp.status_code == 200, resp.read().decode()
                    for line in resp.iter_lines():
                        if line.startswith("data:"):
                            chunks.append(line.strip())
                assert len(chunks) >= 2, f"expected >=2 SSE chunks, got: {chunks}"
                assert any(
                    "[DONE]" in c for c in chunks
                ), f"no [DONE] terminator in stream: {chunks}"
    finally:
        gpu_allocator.release(gpu)
