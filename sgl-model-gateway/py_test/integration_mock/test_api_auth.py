import pytest
import requests


@pytest.mark.integration
def test_router_api_key_enforcement(router_manager, mock_workers):
    # Start backend requiring API key; router should forward Authorization header transparently
    _, urls, _ = mock_workers(
        n=1, args=["--require-api-key", "--api-key", "correct_api_key"]
    )
    rh = router_manager.start_router(
        worker_urls=urls,
        policy="round_robin",
        extra={},
    )

    # No auth -> 401
    r = requests.post(
        f"{rh.url}/v1/completions",
        json={"model": "test-model", "prompt": "x", "max_tokens": 1, "stream": False},
    )
    assert r.status_code == 401

    # Invalid auth -> 401
    r = requests.post(
        f"{rh.url}/v1/completions",
        json={"model": "test-model", "prompt": "x", "max_tokens": 1, "stream": False},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401

    # Correct auth -> 200
    r = requests.post(
        f"{rh.url}/v1/completions",
        json={"model": "test-model", "prompt": "x", "max_tokens": 1, "stream": False},
        headers={"Authorization": "Bearer correct_api_key"},
    )
    assert r.status_code == 200
