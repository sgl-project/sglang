"""Model pool fixtures for E2E tests.

This module provides session-scoped fixtures for managing SGLang worker processes.
Workers are expensive to start (~30-60s each), so they're kept running across tests.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from infra import ModelPool

from .hooks import get_pool_requirements

logger = logging.getLogger(__name__)

# Global model pool instance with thread-safe initialization
_model_pool: "ModelPool | None" = None
_model_pool_lock = threading.Lock()
_shutdown_registered = False


def _shutdown_model_pool() -> None:
    """Shutdown the global model pool at process exit.

    This is registered with atexit to ensure cleanup happens after all tests
    complete, which is important for pytest-parallel where multiple threads
    share the session-scoped fixture.
    """
    global _model_pool
    if _model_pool is not None:
        logger.info("Shutting down model pool at process exit")
        _model_pool.shutdown()
        _model_pool = None


@pytest.fixture(scope="session")
def model_pool(request: pytest.FixtureRequest) -> "ModelPool":
    """Session-scoped fixture that manages SGLang worker processes.

    Workers (sglang.launch_server) are expensive to start (~30-60s each due to
    model loading). This fixture starts them ONCE per session and keeps them
    running across all tests. The setup_backend fixture then launches cheap
    routers (~1-2s) pointing to these workers.

    Startup behavior:
    - Scans test markers to determine required workers (model, mode, type, count)
    - Launches workers in test collection order
    - Waits for all workers to become healthy before returning

    Test requirements are auto-detected from:
    - @pytest.mark.parametrize("setup_backend", ["grpc", "http", "pd"])
    - @pytest.mark.model("model-name")
    - @pytest.mark.workers(count=N) for regular workers
    - @pytest.mark.workers(prefill=N, decode=N) for PD workers

    Environment variable overrides:
    - E2E_MODELS: Comma-separated model IDs (e.g., "llama-8b,qwen-7b")
    - E2E_BACKENDS: Comma-separated backends (e.g., "grpc,http")
    - SKIP_MODEL_POOL: Set to "1" to skip worker startup
    """
    global _model_pool

    from infra import (
        DEFAULT_MODEL,
        ENV_BACKENDS,
        ENV_MODELS,
        ENV_SKIP_MODEL_POOL,
        ENV_STARTUP_TIMEOUT,
        LOCAL_MODES,
        MODEL_SPECS,
        ConnectionMode,
        GPUAllocator,
        ModelPool,
        WorkerIdentity,
        WorkerType,
    )

    # Thread-safe initialization: use lock to ensure only one thread creates the pool
    # This is critical for pytest-parallel which runs tests as concurrent threads
    with _model_pool_lock:
        if _model_pool is not None:
            return _model_pool

        # Check if we should skip model startup
        if os.environ.get(ENV_SKIP_MODEL_POOL, "").lower() in ("1", "true", "yes"):
            logger.info("%s is set, skipping model pool startup", ENV_SKIP_MODEL_POOL)
            _model_pool = ModelPool(GPUAllocator(gpus=[]))
            return _model_pool

        # Determine requirements from scanned tests or env vars
        models_env = os.environ.get(ENV_MODELS, "")
        backends_env = os.environ.get(ENV_BACKENDS, "")

        if models_env or backends_env:
            # Use env var overrides
            models = (
                {m.strip() for m in models_env.split(",") if m.strip()}
                if models_env
                else {DEFAULT_MODEL}
            )

            # Parse backend strings to ConnectionMode enums
            backend_modes: set[ConnectionMode] = set()
            if backends_env:
                for b in backends_env.split(","):
                    b = b.strip()
                    if b:
                        try:
                            mode = ConnectionMode(b)
                            if mode in LOCAL_MODES:
                                backend_modes.add(mode)
                        except ValueError:
                            logger.warning("Unknown backend '%s', skipping", b)

            # Default to HTTP if no valid backends
            if not backend_modes:
                backend_modes = {ConnectionMode.HTTP}

            # Create WorkerIdentity objects (regular workers only from env vars)
            requirements = [
                WorkerIdentity(m, b, WorkerType.REGULAR, 0)
                for m in models
                for b in backend_modes
            ]
            logger.info(
                "Using env var requirements: %s", [str(r) for r in requirements]
            )
        else:
            # Use scanned requirements from test markers
            requirements = get_pool_requirements()
            logger.info(
                "Using scanned requirements: %s", [str(r) for r in requirements]
            )

        # Filter to valid models
        requirements = [r for r in requirements if r.model_id in MODEL_SPECS]

        if not requirements:
            logger.warning("No valid requirements, model pool will be empty")
            _model_pool = ModelPool(GPUAllocator(gpus=[]))
            return _model_pool

        # Create and start the pool
        allocator = GPUAllocator()
        _model_pool = ModelPool(allocator)

        startup_timeout = int(os.environ.get(ENV_STARTUP_TIMEOUT, "300"))
        _model_pool.startup(
            requirements=requirements,
            startup_timeout=startup_timeout,
        )

        # Log final GPU allocation summary
        logger.info(_model_pool.allocator.summary())

        # Register cleanup with atexit instead of request.addfinalizer
        # This is critical for pytest-parallel where multiple threads share
        # the session-scoped fixture - addfinalizer can fire too early
        global _shutdown_registered
        if not _shutdown_registered:
            atexit.register(_shutdown_model_pool)
            _shutdown_registered = True

        return _model_pool


@pytest.fixture
def model_client(request: pytest.FixtureRequest, model_pool: "ModelPool"):
    """Get OpenAI client for the model specified by @pytest.mark.model().

    Usage:
        @pytest.mark.model("llama-8b")
        def test_chat(model_client):
            response = model_client.chat.completions.create(...)
    """
    import openai
    from infra import PARAM_MODEL

    marker = request.node.get_closest_marker(PARAM_MODEL)
    if marker is None:
        pytest.fail(
            f"Test must be marked with @pytest.mark.{PARAM_MODEL}('model-id') "
            "to use model_client fixture"
        )

    model_id = marker.args[0]

    try:
        # get() auto-acquires the returned instance
        instance = model_pool.get(model_id)
    except KeyError:
        pytest.skip(f"Model {model_id} not available in model pool")

    client = openai.OpenAI(
        base_url=f"{instance.base_url}/v1",
        api_key="not-used",
    )

    yield client

    # Release reference to allow eviction
    instance.release()


@pytest.fixture
def model_base_url(request: pytest.FixtureRequest, model_pool: "ModelPool") -> str:
    """Get the base URL for the model specified by @pytest.mark.model().

    Usage:
        @pytest.mark.model("llama-8b")
        def test_direct_http(model_base_url):
            response = httpx.get(f"{model_base_url}/health")
    """
    from infra import PARAM_MODEL

    marker = request.node.get_closest_marker(PARAM_MODEL)
    if marker is None:
        pytest.fail(
            f"Test must be marked with @pytest.mark.{PARAM_MODEL}('model-id') "
            "to use model_base_url fixture"
        )

    model_id = marker.args[0]

    try:
        # get() auto-acquires the returned instance
        instance = model_pool.get(model_id)
    except KeyError:
        pytest.skip(f"Model {model_id} not available in model pool")

    yield instance.base_url

    # Release reference to allow eviction
    instance.release()
