"""Pytest configuration for E2E tests."""

from __future__ import annotations

import logging
import os
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from infra import ModelPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Only add bindings/python to path if the wheel is not installed (for local development)
# This ensures CI tests use the installed wheel which contains the Rust extension
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "bindings" / "python"

# Check if sglang_router is already installed with the Rust extension
_wheel_installed = find_spec("sglang_router.sglang_router_rs") is not None

# Only add bindings/python if wheel is not installed (development mode)
if not _wheel_installed and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Custom pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "model(name): mark test to use a specific model from the model pool",
    )
    config.addinivalue_line(
        "markers",
        "backend(name): mark test to use a specific backend (grpc, openai, etc.)",
    )
    config.addinivalue_line(
        "markers",
        "e2e: mark test as an end-to-end test requiring GPU workers",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running",
    )


# ---------------------------------------------------------------------------
# Model pool fixtures (session-scoped)
# ---------------------------------------------------------------------------

# Global model pool instance
_model_pool: "ModelPool | None" = None


def _get_requested_models(config: pytest.Config) -> list[str]:
    """Determine which models are needed based on collected tests.

    This scans all test items for @pytest.mark.model() markers and returns
    the unique set of models requested.
    """
    models = set()

    # This is called during collection, so we need to iterate items
    for item in config.pluginmanager.get_plugin("main").session.items:
        marker = item.get_closest_marker("model")
        if marker and marker.args:
            models.add(marker.args[0])

    return list(models)


@pytest.fixture(scope="session")
def model_pool(request: pytest.FixtureRequest) -> "ModelPool":
    """Session-scoped fixture providing the model pool.

    The model pool pre-loads all models needed by tests in this session,
    running them in parallel across available GPUs.

    Usage:
        @pytest.mark.model("llama-8b")
        def test_chat(model_pool):
            client = model_pool.get_client("llama-8b")
            ...
    """
    global _model_pool

    # Import here to avoid import errors when infra is not set up
    from infra import MODEL_SPECS, GPUAllocator, ModelPool

    if _model_pool is not None:
        return _model_pool

    # Check if we should skip model startup (e.g., for unit tests)
    if os.environ.get("SKIP_MODEL_POOL", "").lower() in ("1", "true", "yes"):
        logger.info("SKIP_MODEL_POOL is set, skipping model pool startup")
        _model_pool = ModelPool(GPUAllocator(gpus=[]))
        return _model_pool

    # Determine which models to start
    # For now, start models based on environment or a default set
    models_env = os.environ.get("E2E_MODELS", "")
    if models_env:
        model_ids = [m.strip() for m in models_env.split(",") if m.strip()]
    else:
        # Default: start commonly needed models
        model_ids = ["llama-8b", "qwen-7b"]

    # Filter to available specs
    model_ids = [m for m in model_ids if m in MODEL_SPECS]

    if not model_ids:
        logger.warning("No models specified, model pool will be empty")
        _model_pool = ModelPool(GPUAllocator(gpus=[]))
        return _model_pool

    logger.info("Starting model pool with models: %s", model_ids)

    # Create and start the pool
    allocator = GPUAllocator()
    _model_pool = ModelPool(allocator)

    grpc_mode = os.environ.get("E2E_GRPC_MODE", "").lower() in ("1", "true", "yes")
    startup_timeout = int(os.environ.get("E2E_STARTUP_TIMEOUT", "300"))

    _model_pool.startup(
        model_ids=model_ids,
        grpc_mode=grpc_mode,
        startup_timeout=startup_timeout,
    )

    # Register cleanup
    request.addfinalizer(_model_pool.shutdown)

    return _model_pool


@pytest.fixture
def model_client(request: pytest.FixtureRequest, model_pool: "ModelPool"):
    """Get OpenAI client for the model specified by @pytest.mark.model().

    Usage:
        @pytest.mark.model("llama-8b")
        def test_chat(model_client):
            response = model_client.chat.completions.create(...)
    """
    marker = request.node.get_closest_marker("model")
    if marker is None:
        pytest.fail(
            "Test must be marked with @pytest.mark.model('model-id') to use model_client fixture"
        )

    model_id = marker.args[0]

    try:
        return model_pool.get_client(model_id)
    except KeyError:
        pytest.skip(f"Model {model_id} not available in model pool")


@pytest.fixture
def model_base_url(request: pytest.FixtureRequest, model_pool: "ModelPool") -> str:
    """Get the base URL for the model specified by @pytest.mark.model().

    Usage:
        @pytest.mark.model("llama-8b")
        def test_direct_http(model_base_url):
            response = httpx.get(f"{model_base_url}/health")
    """
    marker = request.node.get_closest_marker("model")
    if marker is None:
        pytest.fail(
            "Test must be marked with @pytest.mark.model('model-id') to use model_base_url fixture"
        )

    model_id = marker.args[0]

    try:
        return model_pool.get_base_url(model_id)
    except KeyError:
        pytest.skip(f"Model {model_id} not available in model pool")


# ---------------------------------------------------------------------------
# Backend fixtures (class-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def setup_backend(request: pytest.FixtureRequest):
    """Class-scoped fixture for launching backend clusters.

    This fixture is used with pytest.mark.parametrize to run tests
    against multiple backends.

    Usage:
        @pytest.mark.parametrize("setup_backend", ["grpc", "openai"], indirect=True)
        class TestChatCompletions:
            def test_basic(self, setup_backend):
                backend, model, client = setup_backend
                response = client.chat.completions.create(...)

    Environment variables:
        - SKIP_BACKEND_SETUP: Skip backend startup (for dry runs)
        - SHOW_ROUTER_LOGS: Show subprocess output
    """
    import openai
    from backends import BACKENDS, ClusterInfo, launch_backend

    backend_name = request.param

    # Skip if requested
    if os.environ.get("SKIP_BACKEND_SETUP", "").lower() in ("1", "true", "yes"):
        pytest.skip("SKIP_BACKEND_SETUP is set")

    # Check if backend requires API key
    cfg = BACKENDS.get(backend_name)
    if cfg is None:
        pytest.fail(f"Unknown backend: {backend_name}")

    api_key_env = cfg.get("api_key_env")
    if api_key_env and not os.environ.get(api_key_env):
        pytest.skip(f"{api_key_env} not set, skipping {backend_name} tests")

    logger.info("Setting up backend: %s", backend_name)

    # Launch the backend
    cluster: ClusterInfo = launch_backend(backend_name)

    # Create OpenAI client
    api_key = os.environ.get(api_key_env) if api_key_env else "not-used"
    client = openai.OpenAI(
        base_url=f"{cluster.base_url}/v1",
        api_key=api_key,
    )

    # Yield to test
    try:
        yield backend_name, cfg["model"], client
    finally:
        logger.info("Tearing down backend: %s", backend_name)
        cluster.shutdown()


@pytest.fixture
def backend_cluster(request: pytest.FixtureRequest):
    """Function-scoped fixture for launching a fresh backend per test.

    Unlike setup_backend (class-scoped), this creates a new cluster
    for each test function. Use for tests that modify cluster state.

    Usage:
        @pytest.mark.parametrize("backend_cluster", ["grpc"], indirect=True)
        def test_add_worker(backend_cluster):
            cluster = backend_cluster
            # cluster.base_url, cluster.router_process, etc.
    """
    from backends import BACKENDS, launch_backend

    backend_name = request.param

    cfg = BACKENDS.get(backend_name)
    if cfg is None:
        pytest.fail(f"Unknown backend: {backend_name}")

    api_key_env = cfg.get("api_key_env")
    if api_key_env and not os.environ.get(api_key_env):
        pytest.skip(f"{api_key_env} not set")

    cluster = launch_backend(backend_name)

    try:
        yield cluster
    finally:
        cluster.shutdown()
