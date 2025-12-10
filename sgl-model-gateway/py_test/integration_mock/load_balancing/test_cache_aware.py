import collections
import concurrent.futures
import uuid

import pytest
import requests


@pytest.mark.integration
def test_cache_aware_affinity(mock_workers, router_manager):
    # Two workers; same prompt should stick to one due to cache tree
    _, urls, ids = mock_workers(n=2)
    rh = router_manager.start_router(worker_urls=urls, policy="cache_aware")

    counts = collections.Counter()
    with requests.Session() as s:
        for i in range(12):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "repeated prompt for cache",
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            counts[wid] += 1

    # Expect strong skew toward one worker (tree match); majority > 80%
    top = max(counts.values())
    assert top >= 10, counts


@pytest.mark.integration
def test_cache_aware_diverse_prompts_balances(mock_workers, router_manager):
    # Add latency so concurrent requests overlap and influence load-based selection
    _, urls, ids = mock_workers(n=3, args=["--latency-ms", "30"])
    rh = router_manager.start_router(
        worker_urls=urls,
        policy="cache_aware",
        extra={
            "cache_threshold": 0.99,
            "balance_abs_threshold": 0,
            "balance_rel_threshold": 1.0,
        },
    )

    counts = collections.Counter()

    def call(i):
        # Use diverse, unrelated prompts to avoid prefix matches entirely
        prompt = str(uuid.uuid4())
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": prompt,
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
        )
        assert r.status_code == 200
        return r.headers.get("X-Worker-Id") or r.json().get("worker_id")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for wid in ex.map(call, range(40)):
            counts[wid] += 1

    # Expect participation of at least two workers
    assert sum(1 for v in counts.values() if v > 0) >= 2, counts
