"""Pytest fixtures for sgl-router e2e_test/.

Adapted from SMG's e2e_test/conftest.py. The two key fixtures:

  - ``router_binary``: ensures the release binary exists (or skips with
    a clear message). Override path via ``SGL_ROUTER_BINARY``.
  - ``gpu_allocator``: a session-scoped GPU index manager. M4 tests
    request slices via ``gpu_allocator.acquire(n=1)`` and release on
    teardown.

The fixtures stay deliberately thin — anything fancier (per-test pool,
shared workers, etc.) belongs in the test fixtures themselves.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


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
    full GPU set. M4 tests run serially, so this is fine.
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
    """Locate the release `sgl-router` binary or skip the session."""
    env_path = os.environ.get("SGL_ROUTER_BINARY")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(
        Path(__file__).resolve().parent.parent / "target" / "release" / "sgl-router"
    )
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
    no GPUs are visible — M4 e2e tests are real-GPU.
    """
    n = _detect_gpu_count()
    if n == 0:
        pytest.skip(
            "no NVIDIA GPUs visible to nvidia-smi; sgl-router e2e tests are GPU-only"
        )
    yield GPUAllocator(n)
