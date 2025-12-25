import collections

import pytest
import requests


ROUTING_KEY_HEADER = "X-SMG-Routing-Key"


@pytest.mark.integration
def test_manual_routing_with_header(mock_workers, router_manager):
    """With X-SMG-Routing-Key header: sticky routing + distribution across workers."""
    procs, urls, ids = mock_workers(n=2)
    rh = router_manager.start_router(worker_urls=urls, policy="manual")

    key_to_worker = {}
    with requests.Session() as s:
        # First pass: send requests with different routing keys
        for i in range(10):
            routing_key = f"user-{i}"
            r = s.post(
                f"{rh.url}/v1/completions",
                json={"model": "test", "prompt": "hi", "max_tokens": 1, "stream": False},
                headers={ROUTING_KEY_HEADER: routing_key},
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            key_to_worker[routing_key] = wid

        # Verify distribution: should use multiple workers
        assert len(set(key_to_worker.values())) > 1, "Should distribute across workers"

        # Second pass: verify sticky routing
        for i in range(10):
            routing_key = f"user-{i}"
            r = s.post(
                f"{rh.url}/v1/completions",
                json={"model": "test", "prompt": "hi", "max_tokens": 1, "stream": False},
                headers={ROUTING_KEY_HEADER: routing_key},
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            assert wid == key_to_worker[routing_key], f"Key {routing_key} should stick to same worker"


@pytest.mark.integration
def test_manual_routing_without_header(mock_workers, router_manager):
    """Without X-SMG-Routing-Key header: random fallback distribution."""
    procs, urls, ids = mock_workers(n=2)
    rh = router_manager.start_router(worker_urls=urls, policy="manual")

    counts = collections.Counter()
    with requests.Session() as s:
        for i in range(20):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={"model": "test", "prompt": "hi", "max_tokens": 1, "stream": False},
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            counts[wid] += 1

    assert len(counts) > 1, f"Random fallback should distribute: {counts}"
