import concurrent.futures
import time

import pytest
import requests

from .conftest import _spawn_mock_worker


@pytest.mark.integration
def test_retry_reroutes_to_healthy_worker(router_manager):
    # Worker A always 500; Worker B healthy
    proc_a, url_a, id_a = _spawn_mock_worker(["--status-code", "500"])  # always fail
    proc_b, url_b, id_b = _spawn_mock_worker([])
    try:
        rh = router_manager.start_router(
            worker_urls=[url_a, url_b],
            policy="random",
            extra={
                "retry_max_retries": 3,
                "retry_initial_backoff_ms": 10,
                "retry_max_backoff_ms": 50,
            },
        )

        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "x",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
        )
        assert r.status_code == 200
        wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
        assert wid == id_b  # should have retried onto healthy worker
    finally:
        import subprocess

        for p in (proc_a, proc_b):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()


@pytest.mark.integration
def test_disable_retries_surfaces_failure(router_manager):
    # Single failing worker, retries disabled -> should return 500
    proc, url, wid = _spawn_mock_worker(["--status-code", "500"])  # always fail
    try:
        rh = router_manager.start_router(
            worker_urls=[url],
            policy="round_robin",
            extra={
                "disable_retries": True,
            },
        )

        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "x",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
        )
        assert r.status_code == 500
    finally:
        import subprocess

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
