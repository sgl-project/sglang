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
            extra={"worker_startup_check_interval": 1},
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


@pytest.mark.integration
def test_pd_power_of_two_skews_to_faster_decode(router_manager):
    # Start two prefill workers (fast)
    prefill = [
        _spawn_mock_worker([]),
        _spawn_mock_worker([]),
    ]

    # Start two decode workers: one slow, one fast
    decode_slow = _spawn_mock_worker(["--latency-ms", "300"])  # slower decode
    decode_fast = _spawn_mock_worker([])
    decode = [decode_slow, decode_fast]

    try:
        prefill_urls = [(u, None) for _, u, _ in prefill]
        decode_urls = [u for _, u, _ in decode]
        slow_id = decode_slow[2]
        fast_id = decode_fast[2]

        rh = router_manager.start_router(
            policy="power_of_two",
            pd_disaggregation=True,
            prefill_urls=prefill_urls,
            decode_urls=decode_urls,
            extra={"worker_startup_check_interval": 1},
        )

        # Prime phase to build decode load and allow monitor to update
        import concurrent.futures
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
                    timeout=8,
                )
            except Exception:
                pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            list(ex.map(_prime_call, range(128)))
        _t.sleep(2)

        # Apply direct background load on the slow decode to amplify difference
        def _direct_decode_load(i):
            try:
                requests.post(
                    f"{decode_slow[1]}/v1/completions",
                    json={
                        "model": "test-model",
                        "prompt": f"bg-{i}",
                        "max_tokens": 1,
                        "stream": False,
                    },
                    timeout=8,
                )
            except Exception:
                pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            list(ex.map(_direct_decode_load, range(128)))
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
                timeout=8,
            )
            assert r.status_code == 200
            return r.headers.get("X-Worker-Id") or r.json().get("worker_id")

        import concurrent.futures

        counts = collections.Counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
            for wid in ex.map(call, range(200)):
                counts[wid] += 1

        # Expect the slow decode worker to receive fewer requests than the fast one
        assert counts[slow_id] < counts[fast_id], counts
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
