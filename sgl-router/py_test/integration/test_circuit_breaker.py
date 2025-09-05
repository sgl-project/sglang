import time

import pytest
import requests

from .conftest import _spawn_mock_worker


@pytest.mark.integration
def test_circuit_breaker_opens_and_recovers(router_manager):
    # A single worker that fails first 3 requests, then succeeds
    proc, wurl, wid = _spawn_mock_worker(["--fail-first-n", "3"])
    try:
        rh = router_manager.start_router(
            worker_urls=[wurl],
            policy="round_robin",
            extra={
                "cb_failure_threshold": 3,
                "cb_success_threshold": 2,
                "cb_timeout_duration_secs": 3,
                "cb_window_duration_secs": 10,
                "disable_retries": True,  # simpler failure accounting
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

        # Induce failures until CB opens
        saw_503 = False
        for _ in range(8):
            r = post_once()
            if r.status_code == 503:
                saw_503 = True
                break
        assert saw_503, "circuit breaker did not open to return 503"

        # Wait for CB timeout and half-open
        time.sleep(4)

        # Two successful responses should close the breaker
        r1 = post_once()
        r2 = post_once()
        assert r1.status_code == 200 and r2.status_code == 200
    finally:
        import subprocess

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.mark.integration
def test_circuit_breaker_half_open_failure_reopens(router_manager):
    # Always failing worker triggers CB open, then after timeout the trial fails and CB remains open
    proc, wurl, wid = _spawn_mock_worker(["--status-code", "500"])  # always fail
    try:
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
                json={"model": "test-model", "prompt": "x", "max_tokens": 1, "stream": False},
                timeout=3,
            )

        # Open breaker (see 503 when selecting a worker is impossible)
        # With a single failing worker, after threshold the router should return 503
        opened = False
        for _ in range(8):
            r = post_once()
            if r.status_code == 503:
                opened = True
                break
        assert opened, "circuit breaker did not open"

        # Wait past timeout, then half-open trial will reach backend and fail
        # The trial response is the upstream 500, not 503; subsequent calls return 503 again
        time.sleep(3)
        r = post_once()
        assert r.status_code == 500  # half-open trial propagated
        # Next call should see CB re-opened -> 503
        r2 = post_once()
        assert r2.status_code == 503
    finally:
        import subprocess

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.mark.integration
def test_circuit_breaker_disable_flag(router_manager):
    # With CB disabled, we should not see 503; failures surface as 500
    proc, wurl, wid = _spawn_mock_worker(["--status-code", "500"])  # always fail
    try:
        rh = router_manager.start_router(
            worker_urls=[wurl],
            policy="round_robin",
            extra={
                "disable_circuit_breaker": True,
                "disable_retries": True,
            },
        )
        r = requests.post(
            f"{rh.url}/v1/completions",
            json={"model": "test-model", "prompt": "x", "max_tokens": 1, "stream": False},
            timeout=3,
        )
        assert r.status_code == 500
    finally:
        import subprocess

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.mark.integration
def test_circuit_breaker_per_worker_isolation(router_manager):
    # One failing worker + one healthy; once CB opens for the failing one, router should keep serving via healthy
    fail_proc, fail_url, _ = _spawn_mock_worker(["--status-code", "500"])  # always fail
    ok_proc, ok_url, _ = _spawn_mock_worker([])
    try:
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
                json={"model": "test-model", "prompt": "y", "max_tokens": 1, "stream": False},
                timeout=3,
            )

        # Drive requests; after some initial 500s, CB for failing worker should open
        # Then all subsequent responses should be 200 (router avoids the failing worker)
        failures = 0
        successes_after_open = 0
        opened = False
        for i in range(30):
            r = post_once()
            if not opened:
                if r.status_code == 500:
                    failures += 1
                else:
                    # Success could be from healthy worker; keep going until failures hit threshold
                    pass
                if failures >= 2:  # failure_threshold
                    # give a couple more requests to allow state to flip
                    _ = post_once()
                    _ = post_once()
                    opened = True
            else:
                if r.status_code == 200:
                    successes_after_open += 1
                else:
                    # Once opened we should not see further 500s
                    assert False, f"Unexpected non-200 after CB open: {r.status_code}"
        assert opened and successes_after_open >= 5
    finally:
        import subprocess

        for p in (fail_proc, ok_proc):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()


@pytest.mark.integration
def test_circuit_breaker_with_retries(router_manager):
    # Failing + healthy with retries: requests should succeed via reroute
    fail_proc, fail_url, _ = _spawn_mock_worker(["--status-code", "500"])  # always fail
    ok_proc, ok_url, _ = _spawn_mock_worker([])
    try:
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
            json={"model": "test-model", "prompt": "z", "max_tokens": 1, "stream": False},
            timeout=5,
        )
        assert r.status_code == 200
    finally:
        import subprocess

        for p in (fail_proc, ok_proc):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
