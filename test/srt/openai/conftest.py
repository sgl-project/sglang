# sglang/test/srt/openai/conftest.py
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from typing import Generator

import pytest
import requests

from sglang.srt.utils import kill_process_tree  # reuse SGLang helper
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

SERVER_MODULE = "sglang.srt.entrypoints.openai.api_server"
DEFAULT_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
STARTUP_TIMEOUT = float(os.getenv("SGLANG_OPENAI_STARTUP_TIMEOUT", 120))


def _pick_free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_healthy(proc: subprocess.Popen, base: str, timeout: float) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if proc.poll() is not None:  # crashed
            raise RuntimeError("api_server terminated prematurely")
        try:
            if requests.get(f"{base}/health", timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.4)
    raise RuntimeError("api_server readiness probe timed out")


def launch_openai_server(model: str = DEFAULT_MODEL, **kw):
    """Spawn the draft OpenAI-compatible server and wait until it's ready."""
    port = _pick_free_port()
    cmd = [
        sys.executable,
        "-m",
        SERVER_MODULE,
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *map(str, kw.get("args", [])),
    ]
    env = {**os.environ, **kw.get("env", {})}

    # Write logs to a temp file so the child never blocks on a full pipe.
    log_file = tempfile.NamedTemporaryFile("w+", delete=False)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    base = f"http://127.0.0.1:{port}"
    try:
        _wait_until_healthy(proc, base, STARTUP_TIMEOUT)
    except Exception as e:
        proc.terminate()
        proc.wait(5)
        log_file.seek(0)
        print("\n--- api_server log ---\n", log_file.read(), file=sys.stderr)
        raise e
    return proc, base, log_file


@pytest.fixture(scope="session")
def openai_server() -> Generator[str, None, None]:
    """PyTest fixture that provides the server's base URL and cleans up."""
    proc, base, log_file = launch_openai_server()
    yield base
    kill_process_tree(proc.pid)
    log_file.close()
