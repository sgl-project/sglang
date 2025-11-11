import pytest
import requests


@pytest.mark.integration
def test_payload_size_limit(router_manager, mock_workers):
    # Start one backend and a router with a 1MB payload limit
    _, urls, _ = mock_workers(n=1)
    rh = router_manager.start_router(
        worker_urls=urls,
        policy="round_robin",
        extra={"max_payload_size": 1 * 1024 * 1024},  # 1MB
    )

    # Payload just under 1MB should succeed
    payload_small = {
        "model": "test-model",
        "prompt": "x" * int(0.5 * 1024 * 1024),  # ~0.5MB
        "max_tokens": 1,
        "stream": False,
    }
    r = requests.post(f"{rh.url}/v1/completions", json=payload_small)
    assert r.status_code == 200

    # Payload over 1MB should fail with 413
    payload_large = {
        "model": "test-model",
        "prompt": "x" * int(1.2 * 1024 * 1024),  # ~1.2MB
        "max_tokens": 1,
        "stream": False,
    }
    r = requests.post(f"{rh.url}/v1/completions", json=payload_large)
    assert r.status_code == 413
