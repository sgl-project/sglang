import collections

import pytest
import requests


ROUTING_KEY_HEADER = "X-SMG-Routing-Key"


@pytest.mark.integration
def test_manual_routing_same_key_routes_to_same_worker(mock_workers, router_manager):
    """Requests with the same routing key should route to the same worker."""
    procs, urls, ids = mock_workers(n=2)

    rh = router_manager.start_router(worker_urls=urls, policy="manual")

    # Send 10 requests with the same routing key
    target_worker = None
    with requests.Session() as s:
        for i in range(10):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"hello {i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                headers={ROUTING_KEY_HEADER: "user-123"},
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            assert wid in ids

            if target_worker is None:
                target_worker = wid
            else:
                assert wid == target_worker, (
                    f"Same routing key should route to same worker. "
                    f"Expected {target_worker}, got {wid}"
                )


@pytest.mark.integration
def test_manual_routing_different_keys_distribute(mock_workers, router_manager):
    """Different routing keys should distribute across workers."""
    procs, urls, ids = mock_workers(n=2)

    rh = router_manager.start_router(worker_urls=urls, policy="manual")

    counts = collections.Counter()
    with requests.Session() as s:
        for i in range(30):
            routing_key = f"user-{i}"
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"hello {i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                headers={ROUTING_KEY_HEADER: routing_key},
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            assert wid in ids
            counts[wid] += 1

    # Should distribute across multiple workers
    assert len(counts) > 1, f"Should distribute across workers, but only used: {counts}"


@pytest.mark.integration
def test_manual_routing_no_header_uses_random_fallback(mock_workers, router_manager):
    """Requests without routing key header should use random fallback."""
    procs, urls, ids = mock_workers(n=2)

    rh = router_manager.start_router(worker_urls=urls, policy="manual")

    counts = collections.Counter()
    with requests.Session() as s:
        for i in range(30):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"hello {i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                # No X-SMG-Routing-Key header
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            assert wid in ids
            counts[wid] += 1

    # Random fallback should distribute across workers
    assert len(counts) > 1, f"Random fallback should use multiple workers: {counts}"


@pytest.mark.integration
def test_manual_routing_sticky_sessions(mock_workers, router_manager):
    """Test sticky session behavior - same key always routes to same worker."""
    procs, urls, ids = mock_workers(n=2)

    rh = router_manager.start_router(worker_urls=urls, policy="manual")

    # Map routing keys to their target workers
    key_to_worker = {}

    with requests.Session() as s:
        # First pass: establish routing for each key
        for user_id in range(10):
            routing_key = f"session-{user_id}"
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "hello",
                    "max_tokens": 1,
                    "stream": False,
                },
                headers={ROUTING_KEY_HEADER: routing_key},
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            key_to_worker[routing_key] = wid

        # Second pass: verify same keys route to same workers
        for user_id in range(10):
            routing_key = f"session-{user_id}"
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "hello again",
                    "max_tokens": 1,
                    "stream": False,
                },
                headers={ROUTING_KEY_HEADER: routing_key},
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            expected_worker = key_to_worker[routing_key]
            assert wid == expected_worker, (
                f"Routing key {routing_key} should stick to worker {expected_worker}, "
                f"but got {wid}"
            )

