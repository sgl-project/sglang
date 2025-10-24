import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pytest
import requests

from ..fixtures.ports import find_free_port
from ..fixtures.router_manager import RouterManager


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark as router integration test")


@pytest.fixture
def router_manager() -> Iterable[RouterManager]:
    mgr = RouterManager()
    try:
        yield mgr
    finally:
        mgr.stop_all()


def _spawn_mock_worker(args: List[str]) -> Tuple[subprocess.Popen, str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "py_test" / "fixtures" / "mock_worker.py"
    port = find_free_port()
    worker_id = f"worker-{port}"
    base_cmd = [
        "python3",
        str(script),
        "--port",
        str(port),
        "--worker-id",
        worker_id,
    ]
    cmd = base_cmd + args
    proc = subprocess.Popen(cmd)
    url = f"http://127.0.0.1:{port}"
    _wait_health(url)
    return proc, url, worker_id


def _wait_health(url: str, timeout: float = 10.0):
    start = time.time()
    with requests.Session() as s:
        while time.time() - start < timeout:
            try:
                r = s.get(f"{url}/health", timeout=1)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.1)
    raise TimeoutError(f"Mock worker at {url} did not become healthy")


@pytest.fixture
def mock_worker():
    """Start a single healthy mock worker; yields (process, url, worker_id)."""
    proc, url, worker_id = _spawn_mock_worker([])
    try:
        yield proc, url, worker_id
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture
def mock_workers():
    """Factory to start N workers with custom args.

    Usage:
        procs, urls, ids = mock_workers(n=3, args=["--latency-ms", "5"])  # same args for all
        ...
    """

    procs: List[subprocess.Popen] = []

    def _start(n: int, args: Optional[List[str]] = None):
        args = args or []
        new_procs: List[subprocess.Popen] = []
        urls: List[str] = []
        ids: List[str] = []
        for _ in range(n):
            p, url, wid = _spawn_mock_worker(args)
            procs.append(p)
            new_procs.append(p)
            urls.append(url)
            ids.append(wid)
        return new_procs, urls, ids

    try:
        yield _start
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
