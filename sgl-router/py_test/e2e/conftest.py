import socket
import subprocess
import time
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
import requests

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
)


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _parse_url(base_url: str) -> tuple[str, str]:
    """Parse a base URL and return (host, port) as strings.

    This is more robust than simple string splitting and supports different schemes
    and URL shapes like trailing paths.
    """
    parsed = urlparse(base_url)
    return parsed.hostname or "127.0.0.1", (
        str(parsed.port) if parsed.port is not None else ""
    )


def _wait_router_health(base_url: str, timeout: float) -> None:
    start = time.perf_counter()
    with requests.Session() as session:
        while time.perf_counter() - start < timeout:
            try:
                r = session.get(f"{base_url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
    raise TimeoutError("Router failed to become healthy in time")


def _popen_launch_router(
    model: str,
    base_url: str,
    dp_size: int,
    timeout: float,
    policy: str = "cache_aware",
) -> subprocess.Popen:
    host, port = _parse_url(base_url)

    prom_port = _find_available_port()

    cmd = [
        "python3",
        "-m",
        "sglang_router.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        "--dp",
        str(dp_size),
        "--router-policy",
        policy,
        "--allow-auto-truncate",
        "--router-prometheus-port",
        str(prom_port),
        "--router-prometheus-host",
        "127.0.0.1",
    ]

    proc = subprocess.Popen(cmd)
    _wait_router_health(base_url, timeout)
    return proc


def _popen_launch_worker(
    model: str,
    base_url: str,
    *,
    dp_size: int | None = None,
    api_key: str | None = None,
) -> subprocess.Popen:
    host, port = _parse_url(base_url)

    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        port,
        "--base-gpu-id",
        "0",
    ]
    if dp_size is not None:
        cmd += ["--dp-size", str(dp_size)]
    if api_key is not None:
        cmd += ["--api-key", api_key]
    return subprocess.Popen(cmd)


def _popen_launch_router_only(
    base_url: str,
    policy: str = "round_robin",
    timeout: float = 120.0,
    *,
    dp_aware: bool = False,
    api_key: str | None = None,
) -> subprocess.Popen:
    host, port = _parse_url(base_url)

    prom_port = _find_available_port()
    cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        host,
        "--port",
        port,
        "--policy",
        policy,
    ]
    if dp_aware:
        cmd += ["--dp-aware"]
    if api_key is not None:
        cmd += ["--api-key", api_key]
    cmd += [
        "--prometheus-port",
        str(prom_port),
        "--prometheus-host",
        "127.0.0.1",
    ]
    proc = subprocess.Popen(cmd)
    _wait_router_health(base_url, timeout)
    return proc


def _terminate(proc: subprocess.Popen, timeout: float = 120) -> None:
    if proc is None:
        return
    proc.terminate()
    start = time.perf_counter()
    while proc.poll() is None:
        if time.perf_counter() - start > timeout:
            proc.kill()
            break
        time.sleep(1)


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: mark as end-to-end test")


@pytest.fixture(scope="session")
def e2e_model() -> str:
    # Always use the default test model
    return DEFAULT_MODEL_NAME_FOR_TEST


@pytest.fixture
def e2e_router(e2e_model: str):
    # Keep this available but tests below use router-only to avoid GPU contention
    base_url = DEFAULT_URL_FOR_TEST
    proc = _popen_launch_router(
        e2e_model, base_url, dp_size=2, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    )
    try:
        yield SimpleNamespace(proc=proc, url=base_url)
    finally:
        _terminate(proc)


@pytest.fixture
def e2e_router_only_rr():
    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _popen_launch_router_only(base_url, policy="round_robin")
    try:
        yield SimpleNamespace(proc=proc, url=base_url)
    finally:
        _terminate(proc)


@pytest.fixture(scope="session")
def e2e_primary_worker(e2e_model: str):
    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _popen_launch_worker(e2e_model, base_url)
    # Router health gate will handle worker readiness
    try:
        yield SimpleNamespace(proc=proc, url=base_url)
    finally:
        _terminate(proc)


@pytest.fixture
def e2e_router_only_rr_dp_aware_api():
    """Router-only with dp-aware enabled and an API key."""
    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    api_key = "secret"
    proc = _popen_launch_router_only(
        base_url, policy="round_robin", timeout=180.0, dp_aware=True, api_key=api_key
    )
    try:
        yield SimpleNamespace(proc=proc, url=base_url, api_key=api_key)
    finally:
        _terminate(proc)


@pytest.fixture
def e2e_worker_dp2_api(e2e_model: str, e2e_router_only_rr_dp_aware_api):
    """Worker with dp-size=2 and the same API key as the dp-aware router."""
    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    api_key = e2e_router_only_rr_dp_aware_api.api_key
    proc = _popen_launch_worker(e2e_model, base_url, dp_size=2, api_key=api_key)
    try:
        yield SimpleNamespace(proc=proc, url=base_url)
    finally:
        _terminate(proc)
