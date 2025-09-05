import collections
import concurrent.futures

import pytest
import requests


@pytest.mark.integration
def test_power_of_two_prefers_less_loaded(mock_workers, router_manager):
    # Start two workers: one slow (higher inflight), one fast
    # Router monitors /get_load and Power-of-Two uses cached loads to choose
    procs, urls, ids = mock_workers(
        n=2, args=["--latency-ms", "0"]
    )  # start fast by default

    # Replace one with a slow worker
    from ..conftest import _spawn_mock_worker

    slow_proc, slow_url, slow_id = _spawn_mock_worker(["--latency-ms", "200"])  # slower
    procs.append(slow_proc)
    urls[0] = slow_url
    ids[0] = slow_id

    rh = router_manager.start_router(
        worker_urls=urls,
        policy="power_of_two",
        extra={"worker_startup_check_interval": 1},
    )

    # Prime: fire a burst to create measurable load on slow worker, then wait for monitor tick
    import time as _t

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
    _t.sleep(2)

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

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(_direct_load, range(128)))
    _t.sleep(1)

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

    # Expect the slow worker (higher latency/inflight) to receive fewer requests
    assert counts[slow_id] < counts[[i for i in ids if i != slow_id][0]], counts
