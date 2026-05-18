"""M4 acceptance: PD-mode response carries ``x-sgl-decode-url``.

Setup: PD-disagg router with one prefill worker + two decode workers.
Fire a chat-completion. Read the response headers and assert the
``x-sgl-decode-url`` header is set to one of the two decode URLs.

This validates Patch 3 (response-side header copy) end-to-end against a
real router process and real workers, complementing the in-process
``cargo test`` for the same behavior.
"""

from __future__ import annotations

import httpx
import pytest
from infra.gateway import Gateway
from infra.model_pool import spawn_worker
from infra.model_specs import get_model_spec


@pytest.mark.real_gpu
@pytest.mark.pd_mode
@pytest.mark.slow
def test_pd_mode_response_has_decode_affinity_header(
    router_binary,  # noqa: ARG001
    gpu_allocator,
):
    """Prefill→decode response carries x-sgl-decode-url matching a decode URL."""
    gpu_prefill = gpu_allocator.acquire(1)
    gpu_decode_a = gpu_allocator.acquire(1)
    gpu_decode_b = gpu_allocator.acquire(1)
    try:
        with spawn_worker(
            "qwen3-0.6b",
            gpu_ids=gpu_prefill,
            disagg_mode="prefill",
            bootstrap_port=8997,
        ) as prefill, spawn_worker(
            "qwen3-0.6b", gpu_ids=gpu_decode_a, disagg_mode="decode"
        ) as decode_a, spawn_worker(
            "qwen3-0.6b", gpu_ids=gpu_decode_b, disagg_mode="decode"
        ) as decode_b:
            spec = get_model_spec("qwen3-0.6b")
            with Gateway() as gw:
                gw.start_pd(
                    model_id=spec["model"],
                    tokenizer_path=spec["model"],
                    prefill_urls=[prefill.url],
                    decode_urls=[decode_a.url, decode_b.url],
                    timeout=120.0,
                )

                resp = httpx.post(
                    f"{gw.base_url}/v1/chat/completions",
                    json={
                        "model": spec["model"],
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 4,
                    },
                    timeout=60.0,
                )
                assert resp.status_code == 200, resp.text
                hdr = resp.headers.get("x-sgl-decode-url")
                assert hdr is not None, (
                    f"PD-mode response missing x-sgl-decode-url; "
                    f"headers: {dict(resp.headers)}"
                )
                assert hdr in (decode_a.url, decode_b.url), (
                    f"decode hint {hdr!r} not one of the registered decode URLs: "
                    f"{[decode_a.url, decode_b.url]}"
                )
    finally:
        gpu_allocator.release(gpu_prefill)
        gpu_allocator.release(gpu_decode_a)
        gpu_allocator.release(gpu_decode_b)
