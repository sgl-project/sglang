"""M4 acceptance: cache-aware-zmq convergence.

Setup: two prefill workers serving the same model, both publishing
kv-events via ZMQ. The router runs the ``cache_aware_zmq`` policy.

Test: fire 10 chat-completions requests that share a long common prefix
(200 tokens). After the first request "primes" the radix tree on
whichever worker handled it, subsequent requests should converge to
that same worker (the policy reads the tree and picks the worker with
the highest overlap). We accept >=8/10 as the convergence threshold —
two stragglers cover the inherent race between request 1's response
and request 2's selection (the tree update only fires AFTER request 1
produces its first block of KV-cache, so requests 2..K can still
trickle to the other worker until the BlockStored event propagates).

Acceptance: read `/metrics`, parse `sgl_router_requests_total{...,
outcome="success"}` by `worker_url`, and assert the max is >=8.
"""

from __future__ import annotations

import re
import time

import httpx
import pytest
from infra.gateway import Gateway
from infra.model_pool import spawn_worker
from infra.model_specs import get_model_spec


@pytest.mark.real_gpu
@pytest.mark.slow
def test_cache_aware_zmq_converges_to_one_worker(
    router_binary,  # noqa: ARG001  (forces binary check)
    gpu_allocator,
):
    """At least 8 of 10 same-prefix requests land on the same prefill worker."""
    gpu_a = gpu_allocator.acquire(1)
    gpu_b = gpu_allocator.acquire(1)
    try:
        with spawn_worker(
            "qwen3-0.6b", gpu_ids=gpu_a, enable_kv_events=True
        ) as worker_a, spawn_worker(
            "qwen3-0.6b", gpu_ids=gpu_b, enable_kv_events=True
        ) as worker_b:
            spec = get_model_spec("qwen3-0.6b")
            with Gateway() as gw:
                gw.start_regular(
                    model_id=spec["model"],
                    tokenizer_path=spec["model"],
                    worker_urls=[worker_a.url, worker_b.url],
                    policy="cache_aware_zmq",
                    timeout=120.0,
                )

                # Shared 200-token prefix — large enough to dominate the
                # cache-aware overlap score over the per-request suffix.
                prefix = (
                    "You are an assistant. The following is a long-form context. " * 30
                )
                for i in range(10):
                    resp = httpx.post(
                        f"{gw.base_url}/v1/chat/completions",
                        json={
                            "model": spec["model"],
                            "messages": [
                                {"role": "user", "content": f"{prefix} Q{i}?"},
                            ],
                            "max_tokens": 8,
                            "stream": False,
                        },
                        timeout=60.0,
                    )
                    assert resp.status_code == 200, resp.text

                # Give the metrics layer a beat to record the final request.
                time.sleep(0.5)
                metrics = gw.metrics_text()
                assert metrics is not None, "router did not serve /metrics"
                # Persist the metrics snapshot for the M4 E2E artifact log
                # whenever the test reaches this point — captures the
                # convergence test's per-worker requests_total distribution.
                import os as _os

                dump_path = _os.environ.get(
                    "SGL_E2E_METRICS_DUMP_PATH", "/tmp/m4-metrics.txt"
                )
                try:
                    with open(dump_path, "w", encoding="utf-8") as _f:
                        _f.write(metrics)
                except OSError:
                    pass
                counts = _parse_requests_total_by_worker(metrics)
                assert counts, (
                    f"no successful request counters seen in metrics; "
                    f"raw metrics:\n{metrics}"
                )
                top = max(counts.values())
                assert top >= 8, (
                    f"cache-aware-zmq did not converge: per-worker counts={counts}. "
                    f"expected >=8 on the top worker. metrics:\n{metrics}"
                )
    finally:
        gpu_allocator.release(gpu_a)
        gpu_allocator.release(gpu_b)


_REQUESTS_TOTAL_LINE = re.compile(
    r'sgl_router_requests_total\{worker_url="([^"]+)",model_id="[^"]*",mode="[^"]*",outcome="success"\}\s+([0-9]+)'
)


def _parse_requests_total_by_worker(metrics_text: str) -> dict[str, int]:
    """Project `sgl_router_requests_total{...,outcome="success"}` lines into
    a `{worker_url: count}` dict. Returns an empty dict if no successful
    requests have been recorded yet.
    """
    out: dict[str, int] = {}
    for line in metrics_text.splitlines():
        m = _REQUESTS_TOTAL_LINE.search(line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out
