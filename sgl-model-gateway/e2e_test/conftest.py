"""Pytest configuration for E2E tests.

Parallel Execution
------------------
Tests can run in parallel using pytest-parallel with shared worker processes.
Use --workers 1 --tests-per-worker N for N concurrent test threads:

    pytest --workers 1 --tests-per-worker 4 e2e_test/router/

This leverages the thread-safe ModelPool and GPUAllocator classes to enable
true shared-worker parallelism where all threads share the same session-scoped
model_pool fixture. Tests marked with @pytest.mark.thread_unsafe will be
automatically skipped in parallel mode.

Markers
-------
This module defines several pytest markers for configuring E2E tests:

@pytest.mark.model(name)
    Specify which model to use for the test.

    Args:
        name: Model ID from MODEL_SPECS (e.g., "llama-8b", "qwen-7b")

    GPU Resource Management:
        When GPUs are limited (e.g., 4 GPUs, 6 models), the model pool uses
        MRU (Most Recently Used) eviction:
        1. Models are pre-launched until GPUs are full
        2. When a test needs a model that isn't running, MRU model is evicted
           (models just used are likely done, models not yet used are waiting)
        3. The needed model is then launched on-demand

    Examples:
        @pytest.mark.model("llama-8b")
        @pytest.mark.model("qwen-72b")

@pytest.mark.workers(count=1, prefill=None, decode=None)
    Configure worker topology for the test.

    Args:
        count: Number of regular workers (default: 1)
        prefill: Number of prefill workers for PD disaggregation
        decode: Number of decode workers for PD disaggregation

    Examples:
        @pytest.mark.workers(count=3)  # 3 regular workers
        @pytest.mark.workers(prefill=2, decode=2)  # PD mode

@pytest.mark.gateway(policy="round_robin", timeout=None, extra_args=None)
    Configure the gateway/router.

    Args:
        policy: Routing policy ("round_robin", "random", etc.)
        timeout: Startup timeout in seconds
        extra_args: Additional CLI arguments for the router

    Examples:
        @pytest.mark.gateway(policy="random")
        @pytest.mark.gateway(extra_args=["--cache-routing"])

@pytest.mark.e2e
    Mark test as an end-to-end test requiring GPU workers.

@pytest.mark.slow
    Mark test as slow-running.

@pytest.mark.thread_unsafe(reason=None)
    Mark test as incompatible with parallel thread execution.
    Tests with this marker are automatically skipped when running
    with --tests-per-worker > 1.

    Args:
        reason: Optional explanation of why the test is thread-unsafe.

    Examples:
        @pytest.mark.thread_unsafe
        @pytest.mark.thread_unsafe(reason="Modifies global state")

Fixtures
--------
model_pool: Session-scoped fixture managing SGLang worker processes.
setup_backend: Class-scoped fixture that launches gateway + provides client.

Usage Examples
--------------
Basic test with default model:

    @pytest.mark.e2e
    @pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
    class TestBasic:
        def test_chat(self, setup_backend):
            backend, model, client, gateway = setup_backend
            response = client.chat.completions.create(...)

Test with specific model and multiple backends:

    @pytest.mark.e2e
    @pytest.mark.model("qwen-7b")
    @pytest.mark.parametrize("setup_backend", ["grpc", "http"], indirect=True)
    class TestQwen:
        def test_generate(self, setup_backend):
            ...

PD disaggregation mode:

    @pytest.mark.e2e
    @pytest.mark.workers(prefill=1, decode=1)
    @pytest.mark.parametrize("setup_backend", ["pd"], indirect=True)
    class TestPD:
        def test_pd_inference(self, setup_backend):
            ...
"""

from __future__ import annotations

import logging
import sys
from importlib.util import find_spec
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup (must happen before other imports)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]  # sgl-model-gateway/
_E2E_TEST = Path(__file__).resolve().parent  # e2e_test/
_SRC = _ROOT / "bindings" / "python"

# Add e2e_test to path so "from infra import ..." works
if str(_E2E_TEST) not in sys.path:
    sys.path.insert(0, str(_E2E_TEST))

# Add bindings/python to path if the wheel is not installed (for local development)
_wheel_installed = find_spec("sglang_router.sglang_router_rs") is not None

if not _wheel_installed and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Logging setup (clean output without pytest's "---- live log ----" dividers)
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Configure clean logging to stdout with timestamps and thread info.

    In parallel mode (--tests-per-worker > 1), logs from different threads
    would be interleaved. Including thread name helps identify which test
    produced each log line.
    """
    # Include thread name for parallel execution readability
    # MainThread for sequential, Thread-N for parallel workers
    fmt = "%(asctime)s.%(msecs)03d [%(threadName)s] [%(name)s] %(message)s"
    datefmt = "%H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt))

    for logger_name in ("e2e_test", "infra", "fixtures"):
        log = logging.getLogger(logger_name)
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        log.propagate = False

    for logger_name in ("openai", "httpx", "httpcore", "numexpr"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


_setup_logging()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test visibility hooks
# ---------------------------------------------------------------------------


def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:
    """Print clear test header at start of each test."""
    import threading

    from infra import LOG_SEPARATOR_WIDTH

    test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid
    thread_name = threading.current_thread().name
    print(f"\n{'=' * LOG_SEPARATOR_WIDTH}")
    print(f"[{thread_name}] TEST: {test_name}")
    print(f"{'=' * LOG_SEPARATOR_WIDTH}")


# ---------------------------------------------------------------------------
# Import pytest hooks and fixtures from fixtures/ package
# ---------------------------------------------------------------------------

# Import fixtures - pytest discovers these by name
# Import hooks - pytest discovers these by name
from fixtures import (
    backend_router,
    model_base_url,
    model_client,
    model_pool,
    pytest_collection_finish,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_runtest_setup,
    setup_backend,
)

# Re-export for pytest discovery
__all__ = [
    # Hooks
    "pytest_runtest_logstart",
    "pytest_collection_modifyitems",
    "pytest_collection_finish",
    "pytest_configure",
    "pytest_runtest_setup",
    # Fixtures
    "model_pool",
    "model_client",
    "model_base_url",
    "setup_backend",
    "backend_router",
]
