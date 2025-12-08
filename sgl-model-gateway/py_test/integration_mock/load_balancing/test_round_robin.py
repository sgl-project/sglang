import collections

import pytest
import requests


@pytest.mark.integration
def test_round_robin_distribution(mock_workers, router_manager):
    procs, urls, ids = mock_workers(n=3)

    rh = router_manager.start_router(worker_urls=urls, policy="round_robin")

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
            )
            assert r.status_code == 200
            wid = r.headers.get("X-Worker-Id") or r.json().get("worker_id")
            assert wid in ids
            counts[wid] += 1

    # Expect near-even distribution across 3 workers
    # 30 requests -> ideally 10 each; allow small tolerance ±3
    for wid in ids:
        assert 7 <= counts[wid] <= 13, counts
