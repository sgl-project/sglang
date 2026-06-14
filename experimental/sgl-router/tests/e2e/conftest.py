"""Pytest fixtures for ``experimental/sgl-router/tests/e2e/``.

Two flavors of fixtures coexist here:

  1. **Session-scoped smoke fixtures** (``sglang_server`` + ``router``) —
     launch ONE SGLang worker + ONE router on fixed ports for the whole
     test session. Used by the lightweight ``test_chat_smoke.py`` /
     ``test_tokenize_smoke.py`` files. These are the cheap "did the
     binary start at all" sanity tests.

  2. **Per-test multi-worker fixtures** (``router_binary`` +
     ``gpu_allocator``) — just enough infra for the acceptance tests in
     ``chat_completions/`` to bring up their own multi-worker
     topologies. Backed by the ``infra.gateway.Gateway`` and
     ``infra.model_pool.spawn_worker`` helpers.

Both sets share the same release binary; ``SGL_ROUTER_BINARY`` env var
overrides the path for both.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

logger = logging.getLogger(__name__)

# Make `from infra import gateway, model_pool, model_specs` resolve from
# tests under tests/e2e/ without requiring a sibling `__init__.py` chain.
# Mirrors SMG's e2e_test/conftest.py sys.path setup.
_E2E_DIR = Path(__file__).resolve().parent
if str(_E2E_DIR) not in sys.path:
    sys.path.insert(0, str(_E2E_DIR))

MODEL = "Qwen/Qwen3-0.6B"
SGLANG_PORT = 30000
ROUTER_PORT = 8090

# Path to the release binary. This file lives at
# `experimental/sgl-router/tests/e2e/conftest.py`, so:
#   parent             = tests/e2e/
#   parent.parent      = tests/
#   parent.parent.parent = experimental/sgl-router/   ← cargo workspace root
# A previous version used `parent.parent / "target"`, which pointed at
# `experimental/sgl-router/tests/target/` and silently broke every
# fixture that tries to launch the router binary (CI's
# `cargo build --release` lands the artifact at
# `experimental/sgl-router/target/release/sgl-router`, not under
# `tests/`).
_SGL_ROUTER_ROOT = Path(__file__).parent.parent.parent
_BINARY = (
    Path(os.environ.get("CARGO_TARGET_DIR", str(_SGL_ROUTER_ROOT / "target")))
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
    # Stream the server's stdout/stderr to a file rather than capturing
    # to subprocess.PIPE. The launch_server startup log is verbose (model
    # download, JIT warmup, NCCL init); once a PIPE'd output fills its
    # ~64 KB OS buffer with nothing reading it, the SGLang process
    # blocks on stdout write and never reaches "Server started" — the
    # health probe then times out at 300 s and we have no visibility
    # into *why*. A real log file fixes both (no buffer pressure, and
    # the file is dumped on failure for triage).
    log_path = Path(tempfile.gettempdir()) / f"sglang-server-{SGLANG_PORT}.log"
    log_handle = open(log_path, "w", buffering=1)  # line-buffered
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
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    try:
        _wait_http(f"http://localhost:{SGLANG_PORT}/health", timeout=300)
    except Exception:
        # Dump the server log so the operator can see why startup failed
        # (model download error, port conflict, OOM, JIT crash, etc.).
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_handle.flush()
        log_handle.close()
        try:
            tail = log_path.read_text(errors="replace").splitlines()[-200:]
        except OSError:
            tail = ["(server log unreadable)"]
        logger.error(
            "sglang_server fixture failed; last 200 log lines from %s:\n%s",
            log_path,
            "\n".join(tail),
        )
        raise

    yield f"http://localhost:{SGLANG_PORT}"

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_handle.flush()
    log_handle.close()


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


def build_smoke_router_args(
    *,
    host: str,
    port: int,
    model: str,
    tokenizer_path: str,
    sglang_url: str,
) -> list[str]:
    """Build the sgl-router CLI flags the smoke ``router`` fixture launches.

    Static single-worker discovery (``--worker-urls``) pointed at the one
    SGLang worker, serving exactly one model.
    """
    return [
        "--host",
        host,
        "--port",
        str(port),
        "--model-id",
        model,
        "--tokenizer-path",
        tokenizer_path,
        "--worker-urls",
        sglang_url,
    ]


@pytest.fixture(scope="session")
def router(sglang_server):  # noqa: ARG001  (sglang_server must start first)
    """Launch sgl-router on port 8090 pointed at the SGLang worker."""
    tok_path = _find_tokenizer_path(MODEL)
    args = build_smoke_router_args(
        host="0.0.0.0",
        port=ROUTER_PORT,
        model=MODEL,
        tokenizer_path=tok_path,
        sglang_url=f"http://localhost:{SGLANG_PORT}",
    )

    proc = subprocess.Popen(
        [str(_BINARY), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # try/finally so the router is always reaped — on a readiness-probe
    # failure, a test-body error, or a session-teardown exception alike.
    try:
        _wait_http(f"http://localhost:{ROUTER_PORT}/readyz", timeout=60)
        yield f"http://localhost:{ROUTER_PORT}"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


# ---------------------------------------------------------------------------
# Per-test multi-worker acceptance fixtures
# ---------------------------------------------------------------------------


def _detect_gpu_count() -> int:
    """Count visible GPUs via ``nvidia-smi``. Returns 0 when no NVIDIA GPU
    is available (CI on CPU-only runners, dev laptops, etc.).
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0
    return len([ln for ln in out.decode().splitlines() if ln.strip()])


class GPUAllocator:
    """Single-process GPU index allocator. Test-scoped; not safe for
    cross-process use (pytest-xdist) — each worker would race over the
    full GPU set. Acceptance tests run serially, so this is fine.
    """

    def __init__(self, total: int):
        self.total = total
        self._free: list[int] = list(range(total))
        self._lock = threading.Lock()

    def acquire(self, n: int = 1) -> list[int]:
        with self._lock:
            if n > len(self._free):
                raise pytest.skip.Exception(
                    f"requested {n} GPUs, only {len(self._free)}/{self.total} free"
                )
            picked = self._free[:n]
            self._free = self._free[n:]
            return picked

    def release(self, ids: list[int]) -> None:
        with self._lock:
            self._free.extend(ids)
            self._free.sort()


@pytest.fixture(scope="session")
def router_binary() -> Path:
    """Locate the release ``sgl-router`` binary or skip the session.

    Used by the multi-worker acceptance tests (which spawn their own
    Gateway per test instead of using the session-scoped ``router``
    fixture).
    """
    env_path = os.environ.get("SGL_ROUTER_BINARY")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_BINARY)
    for c in candidates:
        if c.exists():
            return c
    pytest.skip(
        "sgl-router release binary not found at any of: "
        + ", ".join(str(c) for c in candidates)
        + ". Build with `cargo build --release` in experimental/sgl-router/."
    )


@pytest.fixture(scope="session")
def gpu_allocator() -> Iterator[GPUAllocator]:
    """Session-scoped GPU index allocator. Skips the entire session when
    no GPUs are visible — acceptance tests under chat_completions/ are
    real-GPU.
    """
    n = _detect_gpu_count()
    if n == 0:
        pytest.skip(
            "no NVIDIA GPUs visible to nvidia-smi; acceptance tests are GPU-only"
        )
    yield GPUAllocator(n)
