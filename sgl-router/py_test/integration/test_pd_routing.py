import collections

import pytest
import requests

from .conftest import _spawn_mock_worker


@pytest.mark.integration
def test_pd_power_of_two_decode_attribution(router_manager):
    # Start two prefill and three decode mock workers
    prefill = [
        _spawn_mock_worker([]),
        _spawn_mock_worker([]),
    ]
    decode = [
        _spawn_mock_worker([]),
        _spawn_mock_worker([]),
        _spawn_mock_worker([]),
    ]
    try:
        prefill_urls = [(u, None) for _, u, _ in prefill]
        decode_urls = [u for _, u, _ in decode]
        decode_ids = {wid for _, _, wid in decode}

        rh = router_manager.start_router(
            policy="power_of_two",
            pd_disaggregation=True,
            prefill_urls=prefill_urls,
            decode_urls=decode_urls,
        )

        counts = collections.Counter()
        with requests.Session() as s:
            for i in range(30):
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
                # Response should originate from decode server
                assert wid in decode_ids
                counts[wid] += 1

        # Ensure multiple decode servers are exercised
        assert sum(1 for v in counts.values() if v > 0) >= 2
    finally:
        import subprocess

        for trio in (*prefill, *decode):
            p = trio[0]
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
