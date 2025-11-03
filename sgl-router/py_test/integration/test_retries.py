import pytest
import requests


@pytest.mark.integration
def test_retry_reroutes_to_healthy_worker(router_manager, mock_workers):
    # Worker A always 500; Worker B healthy
    # Worker A always 500; Worker B/C healthy
    _, [url_a], [id_a] = mock_workers(n=1, args=["--status-code", "500"])  # fail
    _, [url_b], [id_b] = mock_workers(n=1)
    _, [url_c], [id_c] = mock_workers(n=1)
    rh = router_manager.start_router(
        worker_urls=[url_a, url_b, url_c],
        policy="round_robin",
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
    assert wid in [id_b, id_c]  # should have retried onto a healthy worker (B or C)
    # mock_workers fixture handles cleanup


@pytest.mark.integration
def test_disable_retries_surfaces_failure(router_manager, mock_workers):
    # Single failing worker, retries disabled -> should return 500
    _, [url], [wid] = mock_workers(n=1, args=["--status-code", "500"])  # always fail
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
    # mock_workers fixture handles cleanup
