import concurrent.futures
import time

import pytest
import requests


@pytest.mark.integration
def test_rate_limit_and_queue(router_manager, mock_workers):
    # One fast backend
    _, urls, _ = mock_workers(n=1)
    rh = router_manager.start_router(
        worker_urls=urls,
        policy="round_robin",
        extra={
            "max_concurrent_requests": 2,
            "queue_size": 0,  # no queue -> immediate 429 when limit exceeded
        },
    )

    def call_once(i):
        try:
            r = requests.post(
                f"{rh.url}/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": f"p{i}",
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=3,
            )
            return r.status_code
        except Exception:
            return 599

    # Fire a burst of concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(call_once, range(16)))

    # Expect some to succeed and some to be rate limited (429)
    assert any(code == 200 for code in results)
    assert any(code == 429 for code in results)
