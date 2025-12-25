import collections

import pytest
import requests


ROUTING_KEY_HEADER = "X-SMG-Routing-Key"


@pytest.mark.integration
def test_manual_routing_with_header(mock_workers, router_manager):
    """With X-SMG-Routing-Key header: sticky routing + distribution across workers."""
    _, urls, _ = mock_workers(n=2)
    rh = router_manager.start_router(worker_urls=urls, policy="manual")

    with requests.Session() as s:
        # First pass: record which worker each key routes to
        key_to_worker = {f"user-{i}": send_completion(s, rh.url, f"user-{i}") for i in range(10)}

        # Verify distribution
        assert len(set(key_to_worker.values())) > 1, "Should distribute across workers"

        # Second pass: verify sticky routing
        for key, expected_worker in key_to_worker.items():
            actual_worker = send_completion(s, rh.url, key)
            assert actual_worker == expected_worker, f"Key {key} should stick to same worker"


@pytest.mark.integration
def test_manual_routing_without_header(mock_workers, router_manager):
    """Without X-SMG-Routing-Key header: random fallback distribution."""
    _, urls, _ = mock_workers(n=2)
    rh = router_manager.start_router(worker_urls=urls, policy="manual")

    with requests.Session() as s:
        counts = collections.Counter(send_completion(s, rh.url) for _ in range(20))

    assert len(counts) > 1, f"Random fallback should distribute: {counts}"


def send_completion(session, base_url, routing_key=None):
    headers = {ROUTING_KEY_HEADER: routing_key} if routing_key else {}
    r = session.post(
        f"{base_url}/v1/completions",
        json={"model": "test", "prompt": "hi", "max_tokens": 1, "stream": False},
        headers=headers,
    )
    assert r.status_code == 200
    return r.headers.get("X-Worker-Id") or r.json().get("worker_id")
