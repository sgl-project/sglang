"""M4 acceptance: a request that outlives the stale-timeout returns 504.

Setup: regular-mode router pointing at a single SGLang worker, with a
short stale-request timeout configured (5s). The worker is frozen
mid-request via SIGSTOP so the upstream response never arrives.
After 5s, the janitor MUST sweep the request and the chat handler
MUST return 504 with ``error.code == "stale_request_expired"``.

The stale-request timeout is wired via the router's config — at the
time of writing this brief, the surface is ``[active_load]
stale_request_timeout_secs`` but the actual TOML key has been changing
across M3/M4. This test reads the source-of-truth from
``infra.gateway.Gateway`` and updates the config accordingly.
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
@pytest.mark.slow
def test_stale_request_expired_returns_504(
    router_binary,  # noqa: ARG001
    gpu_allocator,
):
    """Frozen upstream → janitor fires → chat handler returns 504."""
    gpu = gpu_allocator.acquire(1)
    try:
        with spawn_worker("qwen3-0.6b", gpu_ids=gpu) as worker:
            spec = get_model_spec("qwen3-0.6b")
            # Short stale-request timeout (5 s) so the janitor fires
            # within the test's wall-time. Default 300 s is production-
            # sized.  Also keep the proxy request timeout above the
            # stale timeout so the stale-cancel path wins the race
            # against an upstream-timeout error.
            with Gateway(
                stale_request_timeout_secs=5,
                proxy_request_timeout_secs=120,
            ) as gw:
                gw.start_regular(
                    model_id=spec["model"],
                    tokenizer_path=spec["model"],
                    worker_urls=[worker.url],
                    timeout=120.0,
                )

                # Freeze the worker so its response never arrives.
                os.kill(worker.process.pid, signal.SIGSTOP)
                start = time.time()
                try:
                    resp = httpx.post(
                        f"{gw.base_url}/v1/chat/completions",
                        json={
                            "model": spec["model"],
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 10,
                        },
                        timeout=30.0,
                    )
                    elapsed = time.time() - start
                    assert resp.status_code == 504, (
                        f"expected 504 after stale timeout, got "
                        f"{resp.status_code} in {elapsed:.1f}s: {resp.text}"
                    )
                    body = resp.json()
                    assert (
                        body.get("error", {}).get("code") == "stale_request_expired"
                    ), body
                finally:
                    os.kill(worker.process.pid, signal.SIGCONT)
                    time.sleep(1.0)
    finally:
        gpu_allocator.release(gpu)
