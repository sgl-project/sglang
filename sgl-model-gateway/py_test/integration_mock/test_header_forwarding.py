import pytest
import requests


@pytest.mark.integration
def test_header_forwarding_whitelist(mock_workers, router_manager):
    _, urls, _ = mock_workers(n=1)
    rh = router_manager.start_router(worker_urls=urls)

    with requests.Session() as s:
        r = s.post(
            f"{rh.url}/v1/completions",
            json={"model": "test", "prompt": "hi", "max_tokens": 1, "stream": False},
            headers={
                "Authorization": "Bearer test-token",
                "X-SMG-Routing-Key": "routing-123",
                "X-Request-Id": "req-456",
                "X-Correlation-Id": "corr-789",
                "traceparent": "00-trace-span-01",
                "tracestate": "vendor=value",
                "X-Custom-Header": "should-not-forward",
                "Cookie": "session=abc",
            },
        )
        assert r.status_code == 200
        h = r.json().get("received_headers", {})

        assert h.get("authorization") == "Bearer test-token"
        assert h.get("x-request-id") == "req-456"
        assert h.get("x-correlation-id") == "corr-789"
        assert h.get("traceparent") == "00-trace-span-01"
        assert h.get("tracestate") == "vendor=value"

        assert "x-smg-routing-key" not in h
        assert "x-custom-header" not in h
        assert "cookie" not in h
