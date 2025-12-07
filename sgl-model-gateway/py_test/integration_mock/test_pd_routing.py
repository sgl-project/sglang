import collections
import concurrent.futures
import time

import pytest
import requests


@pytest.mark.integration
def test_pd_power_of_two_decode_attribution(router_manager, mock_workers):
    # Start two prefill and three decode mock workers via fixture
    _, prefill_urls_raw, prefill_ids = mock_workers(n=2)
    _, decode_urls_raw, decode_ids_list = mock_workers(n=3)
    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)
    decode_ids = set(decode_ids_list)

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
            assert wid in decode_ids
            counts[wid] += 1

    assert sum(1 for v in counts.values() if v > 0) >= 2


@pytest.mark.integration
def test_pd_power_of_two_skews_to_faster_decode(router_manager, mock_workers):
    # Start two prefill workers (fast)
    _, prefill_urls_raw, _ = mock_workers(n=2)

    # Start two decode workers: one slow, one fast
    _, [decode_slow_url], [slow_id] = mock_workers(
        n=1, args=["--latency-ms", "300"]
    )  # slower decode
    _, [decode_fast_url], [fast_id] = mock_workers(n=1)
    decode_urls_raw = [decode_slow_url, decode_fast_url]

    prefill_urls = [(u, None) for u in prefill_urls_raw]
    decode_urls = list(decode_urls_raw)

    rh = router_manager.start_router(
        policy="power_of_two",
        pd_disaggregation=True,
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        extra={"worker_startup_check_interval": 1},
    )

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
    time.sleep(2)

    def _direct_decode_load(i):
        try:
            requests.post(
                f"{decode_slow_url}/v1/completions",
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
    time.sleep(1)

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

    counts = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        for wid in ex.map(call, range(200)):
            counts[wid] += 1

    assert counts[slow_id] < counts[fast_id], counts
