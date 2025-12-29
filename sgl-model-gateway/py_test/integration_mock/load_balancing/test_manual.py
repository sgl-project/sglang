import collections

import pytest
import requests

ROUTING_KEY_HEADER = "X-SMG-Routing-Key"


@pytest.mark.integration
def test_manual_routing_with_header(mock_workers, router_manager):
    """With X-SMG-Routing-Key header: sticky routing + distribution across workers."""
    _, urls, _ = mock_workers(n=2)
    rh = router_manager.start_router(worker_urls=urls, policy="manual")

    # Send requests: 5 keys × 4 requests each
    results = collections.defaultdict(set)
    with requests.Session() as s:
        for key_id in range(5):
            for _ in range(4):
                worker = send_completion(s, rh.url, f"user-{key_id}")
                results[f"user-{key_id}"].add(worker)

    # Verify sticky: each key should route to exactly one worker
    for key, workers in results.items():
        assert len(workers) == 1, f"Key {key} routed to multiple workers: {workers}"

    # Verify distribution: different keys should use multiple workers
    all_workers = {list(w)[0] for w in results.values()}
    assert len(all_workers) > 1, f"Should distribute across workers: {results}"


@pytest.mark.integration
def test_manual_routing_without_header(mock_workers, router_manager):
    """Without X-SMG-Routing-Key header: random fallback distribution."""
    _, urls, _ = mock_workers(n=2)
    rh = router_manager.start_router(worker_urls=urls, policy="manual")

    with requests.Session() as s:
        counts = collections.Counter(send_completion(s, rh.url) for _ in range(20))

    assert len(counts) > 1, f"Random fallback should distribute: {counts}"


def send_completion(session, base_url, routing_key=None):
    headers = {ROUTING_KEY_HEADER: routing_key} if routing_key is not None else {}
    r = session.post(
        f"{base_url}/v1/completions",
        json={"model": "test", "prompt": "hi", "max_tokens": 1, "stream": False},
        headers=headers,
    )
    assert r.status_code == 200
    return r.headers.get("X-Worker-Id") or r.json().get("worker_id")
