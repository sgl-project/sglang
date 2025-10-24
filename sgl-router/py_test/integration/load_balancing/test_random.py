import collections

import pytest
import requests


@pytest.mark.integration
def test_random_distribution(mock_workers, router_manager):
    procs, urls, ids = mock_workers(n=4)
    rh = router_manager.start_router(worker_urls=urls, policy="random")

    counts = collections.Counter()
    N = 200
    with requests.Session() as s:
        for i in range(N):
            r = s.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"p{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            counts[wid] += 1

    # simple statistical tolerance: each worker should be within ±50% of mean
    mean = N / len(ids)
    for wid in ids:
        assert 0.5 * mean <= counts[wid] <= 1.5 * mean, counts
