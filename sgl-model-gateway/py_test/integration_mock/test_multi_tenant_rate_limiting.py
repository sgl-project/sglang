import concurrent.futures
import time

import pytest
import requests


@pytest.mark.integration
def test_multi_tenant_rate_limiting(router_manager, mock_workers):
    """
    Test that rate limits are applied correctly based on tenant and model.
    """
    # Start 1 mock worker with slight latency to allow concurrency to build
    _, urls, _ = mock_workers(n=1, args=["--latency-ms", "500"])

    # customer-a: restricted to 1 concurrent request
    # model-b: restricted to 2 concurrent requests globally
    # default: 10 concurrent requests
    rh = router_manager.start_router(
        worker_urls=urls,
        extra={
            "rate_limit_rule": ["customer-a:*:1:1", "*:model-b:2:2"],
            "max_concurrent_requests": 10,
            "queue_size": 0,  # Immediate rejection for testing
        },
    )

    def send_request(tenant, model):
        headers = {}
        if tenant:
            headers["X-Tenant-ID"] = tenant

        try:
            r = requests.post(
                f"{rh.url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "test"}],
                },
                timeout=5,
            )
            return r.status_code
        except Exception as e:
            print(f"Request failed: {e}")
            return 500

    # 1. Verify restricted tenant (customer-a)
    # Send 3 concurrent requests, only 1 should succeed
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(send_request, "customer-a", "any-model") for _ in range(3)
        ]
        results = [f.result() for f in futures]

    assert results.count(200) == 1
    assert results.count(429) == 2

    # Wait for bucket to refill
    time.sleep(1.1)

    # 2. Verify restricted model (model-b)
    # Send 4 concurrent requests from different tenants, only 2 should succeed
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(send_request, f"tenant-{i}", "model-b") for i in range(4)
        ]
        results = [f.result() for f in futures]

    assert results.count(200) == 2
    assert results.count(429) == 2

    # 3. Verify global default
    # Send 5 concurrent requests for unrestricted combinations, all should succeed
    # (Global limit is 10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(send_request, "other-tenant", "other-model")
            for _ in range(5)
        ]
        results = [f.result() for f in futures]

    assert all(code == 200 for code in results)
