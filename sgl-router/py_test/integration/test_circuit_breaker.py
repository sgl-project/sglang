import time

import pytest
import requests

from .conftest import _spawn_mock_worker


@pytest.mark.integration
def test_circuit_breaker_opens_and_recovers(router_manager):
    # A single worker that fails first 3 requests, then succeeds
    proc, wurl, wid = _spawn_mock_worker(["--fail-first-n", "3"])
    try:
        rh = router_manager.start_router(
            worker_urls=[wurl],
            policy="round_robin",
            extra={
                "cb_failure_threshold": 3,
                "cb_success_threshold": 2,
                "cb_timeout_duration_secs": 3,
                "cb_window_duration_secs": 10,
                "disable_retries": True,  # simpler failure accounting
            },
        )

        def post_once():
            return requests.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "trigger",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=3,
            )

        # Induce failures until CB opens
        saw_503 = False
        for _ in range(8):
            r = post_once()
            if r.status_code == 503:
                saw_503 = True
                break
        assert saw_503, "circuit breaker did not open to return 503"

        # Wait for CB timeout and half-open
        time.sleep(4)

        # Two successful responses should close the breaker
        r1 = post_once()
        r2 = post_once()
        assert r1.status_code == 200 and r2.status_code == 200
    finally:
        import subprocess

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
