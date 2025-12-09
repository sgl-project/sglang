import collections
import concurrent.futures
import time

import pytest
import requests


@pytest.mark.integration
def test_power_of_two_prefers_less_loaded(mock_workers, router_manager):
    # Start two workers: one slow (higher inflight), one fast
    # Router monitors /get_load and Power-of-Two uses cached loads to choose
    # Start one slow and one fast worker using the fixture factory
    procs_slow, urls_slow, ids_slow = mock_workers(n=1, args=["--latency-ms", "200"])
    procs_fast, urls_fast, ids_fast = mock_workers(n=1, args=["--latency-ms", "0"])
    procs = procs_slow + procs_fast
    urls = urls_slow + urls_fast
    ids = ids_slow + ids_fast
    slow_id = ids_slow[0]
    slow_url = urls_slow[0]

    rh = router_manager.start_router(
        worker_urls=urls,
        policy="power_of_two",
        extra={"worker_startup_check_interval": 1},
    )

    # Prime: fire a burst to create measurable load on slow worker, then wait for monitor tick

    def _prime_call(i):
        try:
            requests.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"warm-{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=5,
            )
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(_prime_call, range(128)))
    time.sleep(2)

    # Apply direct background load on the slow worker to amplify load diff
    def _direct_load(i):
        try:
            requests.post(
                f"{slow_url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"bg-{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=5,
            )
        except Exception:
            pass

    # Start background load in a non-blocking way to keep slow worker busy
    background_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    background_futures = []
    for i in range(32):
        future = background_executor.submit(_direct_load, i)
        background_futures.append(future)

    # Wait longer for the load monitor to update (at least 2 monitor intervals)
    time.sleep(3)

    def call(i):
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": f"p{i}",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=5,
        )
        assert r.status_code == 200
        return r.headers.get("X-Worker-Id") or r.json().get("worker_id")

    counts = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        for wid in ex.map(call, range(200)):
            counts[wid] += 1

    # Clean up background executor
    background_executor.shutdown(wait=False)

    # Expect the slow worker (higher latency/inflight) to receive fewer requests
    fast_worker_id = [i for i in ids if i != slow_id][0]
    assert counts[slow_id] < counts[fast_worker_id], counts
