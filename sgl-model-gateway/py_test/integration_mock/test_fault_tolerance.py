import pytest
import requests


@pytest.mark.integration
def test_worker_crash_reroute_with_retries(router_manager, mock_workers):
    # Start one healthy and one that will crash on first request
    _, [ok_url], _ = mock_workers(n=1)
    _, [crash_url], _ = mock_workers(n=1, args=["--crash-on-request"])
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
    # mock_workers fixture handles cleanup
