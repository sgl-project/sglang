"""M4 acceptance: no prefill workers available → 503 with specific error code.

Setup: PD-disagg router with a single prefill worker and a single decode
worker. We SIGSTOP the prefill worker process to force its breaker open
(after a few failed requests). The next request should return 503 with
``error.code == "no_prefill_workers_available"``.

Skipped by default unless the test environment supports SIGSTOP — on
some CI runners under cgroup constraints SIGSTOP doesn't quite freeze
the process the way the test expects.
"""

from __future__ import annotations

import os
import signal
import time

import httpx
import pytest
from infra.gateway import Gateway
from infra.model_pool import spawn_worker
from infra.model_specs import get_model_spec


@pytest.mark.real_gpu
@pytest.mark.pd_mode
@pytest.mark.slow
def test_no_prefill_workers_available_returns_503(
    router_binary,  # noqa: ARG001
    gpu_allocator,
):
    """Frozen prefill worker → breaker opens → 503 no_prefill_workers_available."""
    gpu_prefill = gpu_allocator.acquire(1)
    gpu_decode = gpu_allocator.acquire(1)
    try:
        with spawn_worker(
            "qwen3-0.6b",
            gpu_ids=gpu_prefill,
            disagg_mode="prefill",
            bootstrap_port=8998,
        ) as prefill, spawn_worker(
            "qwen3-0.6b", gpu_ids=gpu_decode, disagg_mode="decode"
        ) as decode:
            spec = get_model_spec("qwen3-0.6b")
            # Short proxy timeout (2 s) so each spawned-prefill failure
            # surfaces fast enough for the breaker to accumulate three
            # failures within the test's 20-second warmup loop. Default
            # 60 s would mean the breaker never opens in time.
            with Gateway(proxy_request_timeout_secs=2) as gw:
                gw.start_pd(
                    model_id=spec["model"],
                    tokenizer_path=spec["model"],
                    prefill_urls=[prefill.url],
                    prefill_bootstrap_ports=[8998],
                    decode_urls=[decode.url],
                    timeout=120.0,
                )

                # Freeze the prefill worker so all requests time out and
                # trip the circuit breaker.
                os.kill(prefill.process.pid, signal.SIGSTOP)
                try:
                    # Fire enough requests to trip the breaker.
                    for _ in range(10):
                        try:
                            httpx.post(
                                f"{gw.base_url}/v1/chat/completions",
                                json={
                                    "model": spec["model"],
                                    "messages": [{"role": "user", "content": "hi"}],
                                    "max_tokens": 2,
                                },
                                timeout=2.0,
                            )
                        except httpx.RequestError:
                            pass

                    # Now request should fail with no_prefill_workers_available.
                    resp = httpx.post(
                        f"{gw.base_url}/v1/chat/completions",
                        json={
                            "model": spec["model"],
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 2,
                        },
                        timeout=10.0,
                    )
                    assert resp.status_code == 503, (
                        f"expected 503 after freezing prefill, got "
                        f"{resp.status_code}: {resp.text}"
                    )
                    body = resp.json()
                    assert (
                        body.get("error", {}).get("code")
                        == "no_prefill_workers_available"
                    ), body
                finally:
                    os.kill(prefill.process.pid, signal.SIGCONT)
                    # Give the worker a moment to drain the queued SIGCONT.
                    time.sleep(1.0)
    finally:
        gpu_allocator.release(gpu_prefill)
        gpu_allocator.release(gpu_decode)
