import concurrent.futures
import time

import pytest
import requests

from .conftest import _spawn_mock_worker


@pytest.mark.integration
def test_worker_crash_reroute_with_retries(router_manager):
    # Start one healthy and one that will crash on first request
    ok_proc, ok_url, _ = _spawn_mock_worker([])
    crash_proc, crash_url, _ = _spawn_mock_worker(["--crash-on-request"])
    try:
        rh = router_manager.start_router(
            worker_urls=[crash_url, ok_url],
            policy="round_robin",
            extra={
                "retry_max_retries": 3,
                "retry_initial_backoff_ms": 10,
                "retry_max_backoff_ms": 50,
            },
        )

        # A single request should succeed via retry to the healthy worker
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "crash",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
        )
        assert r.status_code == 200
    finally:
        import subprocess

        for p in (ok_proc, crash_proc):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
