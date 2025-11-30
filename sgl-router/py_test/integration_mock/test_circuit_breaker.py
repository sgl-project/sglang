import time

import pytest
import requests


@pytest.mark.integration
def test_circuit_breaker_opens_and_recovers(router_manager, mock_workers):
    # A single worker that fails first 3 requests, then succeeds
    _, [wurl], _ = mock_workers(n=1, args=["--fail-first-n", "3"])  # fails first 3
    rh = router_manager.start_router(
        worker_urls=[wurl],
        policy="round_robin",
        extra={
            "cb_failure_threshold": 3,
            "cb_success_threshold": 2,
            "cb_timeout_duration_secs": 3,
            "cb_window_duration_secs": 10,
            "disable_retries": True,
        },
    )

    def post_once():
        return requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "trigger",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=3,
        )

    # should see 500 when worker actually starts, before that should see 503
    saw_500 = False
    for _ in range(8):
        r = post_once()
        if r.status_code == 500:
            # Worker starts, continue to circuit breaker test
            saw_500 = True
            break
        assert (
            r.status_code == 503
        ), "Should only see 503 when waiting for worker to start"
    assert saw_500, "Worker didn't start after 8 requests"

    saw_503 = False
    for _ in range(4):
        r = post_once()
        if r.status_code == 503:
            saw_503 = True
            break
    assert saw_503, "circuit breaker did not open to return 503"

    time.sleep(4)
    r1 = post_once()
    r2 = post_once()
    assert r1.status_code == 200 and r2.status_code == 200


@pytest.mark.integration
def test_circuit_breaker_half_open_failure_reopens(router_manager, mock_workers):
    _, [wurl], _ = mock_workers(n=1, args=["--status-code", "500"])  # always fail
    rh = router_manager.start_router(
        worker_urls=[wurl],
        policy="round_robin",
        extra={
            "cb_failure_threshold": 2,
            "cb_success_threshold": 2,
            "cb_timeout_duration_secs": 2,
            "cb_window_duration_secs": 5,
            "disable_retries": True,
        },
    )

    def post_once():
        return requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "x",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=3,
        )

    # should see 500 when worker actually starts, before that should see 503
    saw_500 = False
    for _ in range(8):
        r = post_once()
        if r.status_code == 500:
            # Worker starts, continue to circuit breaker test
            saw_500 = True
            break
        assert (
            r.status_code == 503
        ), "Should only see 503 when waiting for worker to start"
    assert saw_500, "Worker didn't start after 8 requests"

    opened = False
    for _ in range(8):
        r = post_once()
        if r.status_code == 503:
            opened = True
            break
    assert opened, "circuit breaker did not open"

    time.sleep(3)
    r = post_once()
    assert r.status_code == 500
    r2 = post_once()
    assert r2.status_code == 503


@pytest.mark.integration
def test_circuit_breaker_disable_flag(router_manager, mock_workers):
    _, [wurl], _ = mock_workers(n=1, args=["--status-code", "500"])  # always fail
    rh = router_manager.start_router(
        worker_urls=[wurl],
        policy="round_robin",
        extra={
            "disable_circuit_breaker": True,
            "disable_retries": True,
        },
    )

    saw_500 = False
    for _ in range(8):
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "x",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=3,
        )
        if r.status_code == 500:
            # Worker starts, continue to check
            saw_500 = True
            break
        assert (
            r.status_code == 503
        ), "Should only see 503 when waiting for worker to start"

    assert saw_500


@pytest.mark.integration
def test_circuit_breaker_per_worker_isolation(router_manager, mock_workers):
    _, [fail_url], _ = mock_workers(n=1, args=["--status-code", "500"])  # always fail
    _, [ok_url], _ = mock_workers(n=1)
    rh = router_manager.start_router(
        worker_urls=[fail_url, ok_url],
        policy="round_robin",
        extra={
            "cb_failure_threshold": 2,
            "cb_success_threshold": 1,
            "cb_timeout_duration_secs": 2,
            "cb_window_duration_secs": 10,
            "disable_retries": True,
        },
    )

    def post_once():
        return requests.post(
            f"{rh.url}/v1/completions",
            json={
                "model": "test-model",
                "prompt": "y",
                "max_tokens": 1,
                "stream": False,
            },
            timeout=3,
        )

    failures = 0
    successes_after_open = 0
    opened = False
    for _ in range(30):
        r = post_once()
        if not opened:
            if r.status_code == 500:
                failures += 1
            if failures >= 2:
                _ = post_once()
                _ = post_once()
                opened = True
        else:
            if r.status_code == 200:
                successes_after_open += 1
            else:
                assert False, f"Unexpected non-200 after CB open: {r.status_code}"
    assert opened and successes_after_open >= 5


@pytest.mark.integration
def test_circuit_breaker_with_retries(router_manager, mock_workers):
    _, [fail_url], _ = mock_workers(n=1, args=["--status-code", "500"])  # always fail
    _, [ok_url], _ = mock_workers(n=1)
    rh = router_manager.start_router(
        worker_urls=[fail_url, ok_url],
        policy="round_robin",
        extra={
            "retry_max_retries": 3,
            "retry_initial_backoff_ms": 10,
            "retry_max_backoff_ms": 50,
            "cb_failure_threshold": 2,
            "cb_success_threshold": 1,
            "cb_timeout_duration_secs": 2,
            "cb_window_duration_secs": 10,
        },
    )

    r = requests.post(
        f"{rh.url}/v1/completions",
        json={
            "model": "test-model",
            "prompt": "z",
            "max_tokens": 1,
            "stream": False,
        },
        timeout=5,
    )
    assert r.status_code == 200
