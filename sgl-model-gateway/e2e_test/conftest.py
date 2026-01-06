"""Pytest configuration for E2E tests."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from infra import ModelPool


# ---------------------------------------------------------------------------
# Logging setup (clean output without pytest's "---- live log ----" dividers)
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    """Configure clean logging to stdout with timestamps."""
    # Custom format: timestamp [logger] message
    fmt = "%(asctime)s.%(msecs)03d [%(name)s] %(message)s"
    datefmt = "%H:%M:%S"

    # Create handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt))

    # Configure our e2e_test and infra modules for INFO level
    for logger_name in ("e2e_test", "infra"):
        log = logging.getLogger(logger_name)
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        log.propagate = False  # Don't double-log

    # Suppress noisy third-party loggers
    for logger_name in ("openai", "httpx", "httpcore", "numexpr"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


_setup_logging()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test visibility hooks
# ---------------------------------------------------------------------------
def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:
    """Print clear test header at start of each test."""
    # Extract test name from nodeid (e.g., "test_mmlu.py::TestMMLU::test_mmlu_basic[grpc]")
    test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")


# Path setup for imports
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

# Import constants after path setup
from infra import (
    DEFAULT_MODEL,
    DEFAULT_ROUTER_TIMEOUT,
    ENV_BACKENDS,
    ENV_MODEL,
    ENV_MODELS,
    ENV_SHOW_ROUTER_LOGS,
    ENV_SKIP_BACKEND_SETUP,
    ENV_SKIP_MODEL_POOL,
    ENV_STARTUP_TIMEOUT,
    LOCAL_MODES,
    PARAM_MODEL,
    PARAM_SETUP_BACKEND,
    ConnectionMode,
)

# ---------------------------------------------------------------------------
# Test collection: scan for required backends
# ---------------------------------------------------------------------------

# Global storage for scanned requirements
_scanned_backends: set[str] = set()  # {"grpc", "http", "openai", ...}
_scanned_models: set[str] = set()  # {"llama-8b", "qwen-7b", ...}


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Scan collected tests to determine required backends and models.

    This runs after test collection but before tests execute.
    It extracts backend requirements from @pytest.mark.parametrize markers.
    """
    global _scanned_backends, _scanned_models

    for item in items:
        # Scan parametrize markers for setup_backend
        for marker in item.iter_markers("parametrize"):
            if marker.args and len(marker.args) >= 2:
                param_name = marker.args[0]
                param_values = marker.args[1]

                if param_name == PARAM_SETUP_BACKEND:
                    # Extract backend names from parametrize values
                    if isinstance(param_values, (list, tuple)):
                        _scanned_backends.update(param_values)

                elif param_name == PARAM_MODEL or PARAM_MODEL in param_name:
                    # Extract model names
                    if isinstance(param_values, (list, tuple)):
                        _scanned_models.update(param_values)

        # Also check for @pytest.mark.model("name") markers
        model_marker = item.get_closest_marker(PARAM_MODEL)
        if model_marker and model_marker.args:
            _scanned_models.add(model_marker.args[0])

    logger.info(
        "Scanned test requirements - backends: %s, models: %s",
        _scanned_backends or {"(none)"},
        _scanned_models or {"(none)"},
    )


def get_pool_requirements() -> list[tuple[str, ConnectionMode]]:
    """Build pool requirements from scanned test markers.

    Returns:
        List of (model_id, ConnectionMode) tuples needed by tests.
    """
    # Default model if none specified
    models = _scanned_models or {DEFAULT_MODEL}

    # Convert scanned string backends to ConnectionMode enums
    # Filter to local backends only (grpc, http) - cloud backends don't need workers
    local_modes: set[ConnectionMode] = set()
    for backend in _scanned_backends:
        try:
            mode = ConnectionMode(backend)
            if mode in LOCAL_MODES:
                local_modes.add(mode)
        except ValueError:
            # Not a ConnectionMode (e.g., "openai", "xai") - skip
            pass

    # Default to HTTP if no local backends specified
    if not local_modes:
        local_modes = {ConnectionMode.HTTP}

    # Build requirements: each model needs each mode
    requirements: list[tuple[str, ConnectionMode]] = []
    for model in models:
        for mode in local_modes:
            requirements.append((model, mode))

    return requirements


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
        "backend(name): mark test to use a specific backend (grpc, http, openai, etc.)",
    )
    config.addinivalue_line(
        "markers",
        "workers(n): number of workers to launch behind the router (default: 1)",
    )
    config.addinivalue_line(
        "markers",
        "pd(num_prefill=1, num_decode=1): PD disaggregation worker configuration",
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


@pytest.fixture(scope="session")
def model_pool(request: pytest.FixtureRequest) -> "ModelPool":
    """Session-scoped fixture that manages SGLang worker processes.

    Workers (sglang.launch_server) are expensive to start (~30-60s each due to
    model loading). This fixture starts them ONCE per session and keeps them
    running across all tests. The setup_backend fixture then launches cheap
    routers (~1-2s) pointing to these workers.

    Startup behavior:
    - Scans test markers to determine required (model, mode) combinations
    - Launches workers sequentially, but they boot up concurrently
    - Waits for all workers to become healthy before returning

    Test requirements are auto-detected from:
    - @pytest.mark.parametrize("setup_backend", ["grpc", "http"])
    - @pytest.mark.model("model-name")

    Environment variable overrides:
    - E2E_MODELS: Comma-separated model IDs (e.g., "llama-8b,qwen-7b")
    - E2E_BACKENDS: Comma-separated backends (e.g., "grpc,http")
    - SKIP_MODEL_POOL: Set to "1" to skip worker startup
    """
    global _model_pool

    from infra import MODEL_SPECS, GPUAllocator, ModelPool

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

        requirements = [(m, b) for m in models for b in backend_modes]
        logger.info("Using env var requirements: %s", requirements)
    else:
        # Use scanned requirements from test markers
        requirements = get_pool_requirements()
        logger.info("Using scanned requirements: %s", requirements)

    # Filter to valid models
    requirements = [(m, b) for m, b in requirements if m in MODEL_SPECS]

    if not requirements:
        logger.warning("No valid requirements, model pool will be empty")
        _model_pool = ModelPool(GPUAllocator(gpus=[]))
        return _model_pool

    # Create and start the pool
    allocator = GPUAllocator()
    _model_pool = ModelPool(allocator)

    startup_timeout = int(os.environ.get(ENV_STARTUP_TIMEOUT, "300"))
    _model_pool.startup(requirements=requirements, startup_timeout=startup_timeout)

    # Pre-launch PD workers if 'pd' backend is detected
    if "pd" in _scanned_backends:
        logger.info("PD backend detected, pre-launching PD workers")
        # Use default model for PD workers
        pd_model = next(iter(_scanned_models), DEFAULT_MODEL)
        if pd_model in MODEL_SPECS:
            try:
                _model_pool.launch_pd_workers(
                    model_id=pd_model,
                    num_prefill=1,
                    num_decode=1,
                    startup_timeout=startup_timeout,
                )
            except Exception as e:
                logger.warning("Failed to pre-launch PD workers: %s", e)

    # Log final GPU allocation summary
    logger.info(_model_pool.allocator.summary())

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
    marker = request.node.get_closest_marker(PARAM_MODEL)
    if marker is None:
        pytest.fail(
            f"Test must be marked with @pytest.mark.{PARAM_MODEL}('model-id') to use model_client fixture"
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
    marker = request.node.get_closest_marker(PARAM_MODEL)
    if marker is None:
        pytest.fail(
            f"Test must be marked with @pytest.mark.{PARAM_MODEL}('model-id') to use model_base_url fixture"
        )

    model_id = marker.args[0]

    try:
        return model_pool.get_base_url(model_id)
    except KeyError:
        pytest.skip(f"Model {model_id} not available in model pool")


# ---------------------------------------------------------------------------
# Router launching helpers
# ---------------------------------------------------------------------------


def launch_local_router(
    worker_urls: list[str],
    model_path: str,
    *,
    policy: str = "round_robin",
    router_args: list[str] | None = None,
    timeout: float = DEFAULT_ROUTER_TIMEOUT,
    show_output: bool | None = None,
) -> tuple[str, subprocess.Popen]:
    """Launch a router pointing to pre-started workers.

    Args:
        worker_urls: List of worker URLs (e.g., ["http://127.0.0.1:30000"])
        model_path: Model path for the router
        policy: Routing policy
        router_args: Additional router arguments
        timeout: Startup timeout in seconds
        show_output: Show subprocess output

    Returns:
        Tuple of (base_url, router_process)
    """
    from infra import get_open_port, wait_for_workers_ready

    if show_output is None:
        show_output = os.environ.get(ENV_SHOW_ROUTER_LOGS, "0") == "1"

    router_port = get_open_port()
    prometheus_port = get_open_port()
    base_url = f"http://127.0.0.1:{router_port}"

    cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(router_port),
        "--prometheus-port",
        str(prometheus_port),
        "--policy",
        policy,
        "--model-path",
        model_path,
        "--log-level",
        "warn",
        "--worker-urls",
        *worker_urls,
    ]

    if router_args:
        cmd.extend(router_args)

    logger.info("Starting router on port %d with workers: %s", router_port, worker_urls)

    router_proc = subprocess.Popen(
        cmd,
        stdout=None if show_output else subprocess.PIPE,
        stderr=None if show_output else subprocess.PIPE,
        start_new_session=True,
    )

    try:
        wait_for_workers_ready(base_url, len(worker_urls), timeout=timeout)
    except TimeoutError:
        from infra import kill_process_tree

        kill_process_tree(router_proc.pid)
        raise

    logger.info("Router ready at %s", base_url)
    return base_url, router_proc


def launch_pd_router(
    prefills: list,
    decodes: list,
    *,
    policy: str = "round_robin",
    router_args: list[str] | None = None,
    timeout: float = DEFAULT_ROUTER_TIMEOUT,
    show_output: bool | None = None,
) -> tuple[str, subprocess.Popen]:
    """Launch a PD disaggregation router.

    Args:
        prefills: List of prefill ModelInstance objects.
        decodes: List of decode ModelInstance objects.
        policy: Routing policy.
        router_args: Additional router arguments.
        timeout: Startup timeout in seconds.
        show_output: Show subprocess output.

    Returns:
        Tuple of (base_url, router_process)
    """
    from infra import get_open_port, wait_for_health

    if show_output is None:
        show_output = os.environ.get(ENV_SHOW_ROUTER_LOGS, "0") == "1"

    router_port = get_open_port()
    prometheus_port = get_open_port()
    base_url = f"http://127.0.0.1:{router_port}"

    cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        "127.0.0.1",
        "--port",
        str(router_port),
        "--prometheus-port",
        str(prometheus_port),
        "--prometheus-host",
        "127.0.0.1",
        "--policy",
        policy,
        "--pd-disaggregation",
        "--log-level",
        "warn",
    ]

    # Add prefill workers with bootstrap ports
    for pf in prefills:
        cmd += ["--prefill", pf.base_url, str(pf.bootstrap_port)]

    # Add decode workers
    for dc in decodes:
        cmd += ["--decode", dc.base_url]

    if router_args:
        cmd.extend(router_args)

    logger.info(
        "Starting PD router on port %d with %d prefill, %d decode workers",
        router_port,
        len(prefills),
        len(decodes),
    )

    router_proc = subprocess.Popen(
        cmd,
        stdout=None if show_output else subprocess.PIPE,
        stderr=None if show_output else subprocess.PIPE,
        start_new_session=True,
    )

    try:
        wait_for_health(base_url, timeout=timeout)
    except TimeoutError:
        from infra import kill_process_tree

        kill_process_tree(router_proc.pid)
        raise

    logger.info("PD Router ready at %s", base_url)
    return base_url, router_proc


# ---------------------------------------------------------------------------
# Backend fixtures
# ---------------------------------------------------------------------------


def _get_marker_value(
    request: pytest.FixtureRequest,
    marker_name: str,
    arg_index: int = 0,
    default: any = None,
) -> any:
    """Get a value from a pytest marker.

    Args:
        request: The pytest fixture request.
        marker_name: Name of the marker to look for.
        arg_index: Index of positional argument to extract.
        default: Default value if marker not found.

    Returns:
        The marker argument value or default.
    """
    marker = request.node.get_closest_marker(marker_name)
    if marker is None:
        return default
    if marker.args and len(marker.args) > arg_index:
        return marker.args[arg_index]
    return default


def _get_marker_kwargs(
    request: pytest.FixtureRequest,
    marker_name: str,
    defaults: dict[str, any] | None = None,
) -> dict[str, any]:
    """Get keyword arguments from a pytest marker.

    Args:
        request: The pytest fixture request.
        marker_name: Name of the marker to look for.
        defaults: Default values if marker not found or missing kwargs.

    Returns:
        Dict of keyword arguments merged with defaults.
    """
    result = dict(defaults) if defaults else {}
    marker = request.node.get_closest_marker(marker_name)
    if marker is not None:
        result.update(marker.kwargs)
    return result


@pytest.fixture(scope="class")
def setup_backend(request: pytest.FixtureRequest, model_pool: "ModelPool"):
    """Class-scoped fixture that launches a router for each test class.

    Routers are cheap to start (~1-2s) compared to workers (~30-60s), so we
    launch a fresh router per test class for isolation while reusing the
    expensive workers from model_pool.

    Backend types:
    - "http", "grpc": Gets existing worker from model_pool, launches router
    - "pd": Launches prefill/decode workers via model_pool, launches PD router
    - "openai", "xai", etc.: Launches cloud router (no local workers)

    Configuration via markers:
    - @pytest.mark.model("model-id"): Override default model
    - @pytest.mark.workers(n): Number of workers behind router (default: 1)
    - @pytest.mark.pd(num_prefill=1, num_decode=1): PD worker configuration

    Returns:
        Tuple of (backend_name, model_path, openai_client)

    Usage:
        # Simple - uses defaults
        @pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
        class TestBasic:
            ...

        # With model override
        @pytest.mark.model("qwen-7b")
        @pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
        class TestWithModel:
            ...

        # Load balancing with multiple workers
        @pytest.mark.workers(3)
        @pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
        class TestLoadBalancing:
            ...

        # PD with custom configuration
        @pytest.mark.pd(num_prefill=2, num_decode=2)
        @pytest.mark.parametrize("setup_backend", ["pd"], indirect=True)
        class TestPDScaling:
            ...
    """
    import openai
    from infra import kill_process_tree

    backend_name = request.param

    # Skip if requested
    if os.environ.get(ENV_SKIP_BACKEND_SETUP, "").lower() in ("1", "true", "yes"):
        pytest.skip(f"{ENV_SKIP_BACKEND_SETUP} is set")

    # Get model from marker or env var or default
    model_id = _get_marker_value(request, "model")
    if model_id is None:
        model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    # PD disaggregation backend
    if backend_name == "pd":
        # Check PD requirements
        try:
            import sgl_kernel  # noqa: F401
        except ImportError:
            pytest.skip("sgl_kernel not available, required for PD disaggregation")

        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Get PD configuration from marker
        pd_config = _get_marker_kwargs(
            request, "pd", defaults={"num_prefill": 1, "num_decode": 1}
        )
        num_prefill = pd_config["num_prefill"]
        num_decode = pd_config["num_decode"]

        # Check GPU requirements
        required_gpus = num_prefill + num_decode
        gpu_count = torch.cuda.device_count()
        if gpu_count < required_gpus:
            pytest.skip(
                f"PD tests require {required_gpus} GPUs "
                f"({num_prefill} prefill + {num_decode} decode), found {gpu_count}"
            )

        # Try to use pre-launched PD workers, or launch new ones if needed
        from infra import WorkerType

        existing_prefills = model_pool.get_workers_by_type(model_id, WorkerType.PREFILL)
        existing_decodes = model_pool.get_workers_by_type(model_id, WorkerType.DECODE)

        if (
            len(existing_prefills) >= num_prefill
            and len(existing_decodes) >= num_decode
        ):
            # Use pre-launched workers
            prefills = existing_prefills[:num_prefill]
            decodes = existing_decodes[:num_decode]
            logger.info(
                "Using pre-launched PD workers: %d prefill, %d decode",
                len(prefills),
                len(decodes),
            )
        else:
            # Launch new PD workers (custom config or not pre-launched)
            prefills, decodes = model_pool.launch_pd_workers(
                model_id=model_id,
                num_prefill=num_prefill,
                num_decode=num_decode,
                startup_timeout=300,
            )

        model_path = prefills[0].model_path if prefills else None

        # Launch PD router
        base_url, router_proc = launch_pd_router(prefills, decodes)

        client = openai.OpenAI(
            base_url=f"{base_url}/v1",
            api_key="not-used",
        )

        logger.info(
            "Setup PD backend: model=%s, %d prefill + %d decode workers, router=%s",
            model_id,
            len(prefills),
            len(decodes),
            base_url,
        )

        try:
            yield backend_name, model_path, client
        finally:
            logger.info("Tearing down PD router")
            kill_process_tree(router_proc.pid)
        return

    # Check if this is a local backend (grpc, http)
    try:
        connection_mode = ConnectionMode(backend_name)
        is_local = connection_mode in LOCAL_MODES
    except ValueError:
        is_local = False
        connection_mode = None

    # Local backends: use worker from pool + launch router
    if is_local:
        # Get number of workers from marker
        num_workers = _get_marker_value(request, "workers", default=1)

        try:
            instance = model_pool.get(model_id, connection_mode)
        except KeyError:
            pytest.skip(f"Model {model_id}:{backend_name} not available in pool")

        # Build worker URLs list
        # For num_workers > 1, we need multiple workers from the pool
        # For now, we reuse the same worker URL (router will load balance)
        # TODO: Support launching multiple distinct workers for true LB testing
        worker_urls = [instance.worker_url] * num_workers

        # Launch router pointing to the worker(s)
        base_url, router_proc = launch_local_router(
            worker_urls=worker_urls,
            model_path=instance.model_path,
        )

        client = openai.OpenAI(
            base_url=f"{base_url}/v1",
            api_key="not-used",
        )

        logger.info(
            "Setup %s backend: model=%s, workers=%d, router=%s",
            backend_name,
            model_id,
            num_workers,
            base_url,
        )

        try:
            yield backend_name, instance.model_path, client
        finally:
            logger.info("Tearing down router for %s backend", backend_name)
            kill_process_tree(router_proc.pid)
        return

    # Cloud backends: launch cloud router
    from backends import CLOUD_BACKENDS, launch_cloud_backend

    if backend_name not in CLOUD_BACKENDS:
        pytest.fail(f"Unknown backend: {backend_name}")

    cfg = CLOUD_BACKENDS[backend_name]
    api_key_env = cfg.get("api_key_env")

    if api_key_env and not os.environ.get(api_key_env):
        pytest.skip(f"{api_key_env} not set, skipping {backend_name} tests")

    logger.info("Launching cloud backend: %s", backend_name)
    router = launch_cloud_backend(backend_name)

    api_key = os.environ.get(api_key_env) if api_key_env else "not-used"
    client = openai.OpenAI(
        base_url=f"{router.base_url}/v1",
        api_key=api_key,
    )

    try:
        yield backend_name, cfg["model"], client
    finally:
        logger.info("Tearing down cloud backend: %s", backend_name)
        router.shutdown()


@pytest.fixture
def backend_router(request: pytest.FixtureRequest, model_pool: "ModelPool"):
    """Function-scoped fixture for launching a fresh router per test.

    This launches a new router for each test, pointing to workers from the pool.
    Use for tests that need isolated router state.

    Usage:
        @pytest.mark.parametrize("backend_router", ["grpc", "http"], indirect=True)
        def test_router_state(backend_router):
            base_url, router_proc = backend_router
            # Test router-specific behavior
    """
    from infra import kill_process_tree

    backend_name = request.param
    model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    # Convert string to ConnectionMode
    connection_mode = ConnectionMode(backend_name)

    try:
        instance = model_pool.get(model_id, connection_mode)
    except KeyError:
        pytest.skip(f"Model {model_id}:{backend_name} not available in pool")

    base_url, router_proc = launch_local_router(
        worker_urls=[instance.worker_url],
        model_path=instance.model_path,
    )

    try:
        yield base_url, router_proc
    finally:
        kill_process_tree(router_proc.pid)
