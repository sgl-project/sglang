import socket
import subprocess
import time
from types import SimpleNamespace

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
    _, host, port = base_url.split(":")
    host = host[2:]

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


def _popen_launch_worker(model: str, base_url: str) -> subprocess.Popen:
    _, host, port = base_url.split(":")
    host = host[2:]

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
    return subprocess.Popen(cmd)


def _popen_launch_router_only(
    base_url: str,
    policy: str = "round_robin",
    timeout: float = 120.0,
) -> subprocess.Popen:
    _, host, port = base_url.split(":")
    host = host[2:]

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
