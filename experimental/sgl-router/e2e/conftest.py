"""
Session-scoped fixtures that spin up a real SGLang server and sgl-router,
then tear them down after the test session completes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import pytest

MODEL = "Qwen/Qwen3-0.6B"
SGLANG_PORT = 30000
ROUTER_PORT = 8090

# Path to the release binary relative to the repo root (experimental/sgl-router/ working dir).
_REPO_ROOT = Path(__file__).parent.parent.parent  # sglang_workspace root
_BINARY = (
    Path(
        os.environ.get("CARGO_TARGET_DIR", str(Path(__file__).parent.parent / "target"))
    )
    / "release"
    / "sgl-router"
)


def _wait_http(url: str, timeout: int = 120) -> None:
    """Poll *url* until it returns 2xx or raises RuntimeError on timeout."""
    deadline = time.time() + timeout
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=5)
            if resp.status_code < 300:
                return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        time.sleep(5)
    raise RuntimeError(
        f"Timed out waiting for {url} after {timeout}s (last error: {last_exc})"
    )


@pytest.fixture(scope="session")
def sglang_server():
    """Launch a real SGLang server on port 30000 and wait until healthy."""
    proc = subprocess.Popen(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL,
            "--port",
            str(SGLANG_PORT),
            "--tp",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        _wait_http(f"http://localhost:{SGLANG_PORT}/health", timeout=300)
    except Exception:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        raise

    yield f"http://localhost:{SGLANG_PORT}"

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _find_tokenizer_path(model: str) -> str:
    """Locate the tokenizer.json for *model* from the local HF Hub cache.

    Falls back to the model string itself (a valid HF Hub repo identifier
    that dynamo-tokenizers can resolve at runtime) when the cache is absent.
    """
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import]

        path = try_to_load_from_cache(model, "tokenizer.json")
        if path and Path(path).is_file():
            return str(path)
    except Exception:  # noqa: BLE001
        pass
    # Let dynamo-tokenizers resolve the repo identifier directly.
    return model


@pytest.fixture(scope="session")
def router(sglang_server):  # noqa: ARG001  (sglang_server must start first)
    """Launch sgl-router on port 8090 pointed at the SGLang worker."""
    tok_path = _find_tokenizer_path(MODEL)
    config_content = f"""\
[server]
host = "0.0.0.0"
port = {ROUTER_PORT}

[[models]]
id = "{MODEL}"
tokenizer_path = "{tok_path}"

[[workers]]
url = "http://localhost:{SGLANG_PORT}"
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False
    ) as cfg_file:
        cfg_file.write(config_content)
        cfg_path = cfg_file.name

    try:
        proc = subprocess.Popen(
            [str(_BINARY), "--config", cfg_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        try:
            _wait_http(f"http://localhost:{ROUTER_PORT}/readyz", timeout=60)
        except Exception:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)
            raise

        yield f"http://localhost:{ROUTER_PORT}"

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        Path(cfg_path).unlink(missing_ok=True)
