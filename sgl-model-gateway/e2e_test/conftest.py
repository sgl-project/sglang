"""Pytest configuration for E2E tests.

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
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    print(f"\n{'=' * LOG_SEPARATOR_WIDTH}")
    print(f"TEST: {test_name}")
    print(f"{'=' * LOG_SEPARATOR_WIDTH}")


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
    LOG_SEPARATOR_WIDTH,
    PARAM_MODEL,
    PARAM_SETUP_BACKEND,
    ConnectionMode,
    WorkerIdentity,
    WorkerType,
)

# ---------------------------------------------------------------------------
# Test collection: scan for required workers
# ---------------------------------------------------------------------------

# Track max worker counts: (model_id, mode, worker_type) -> max_count
# This unified approach handles regular, prefill, and decode workers the same way
_worker_counts: dict[tuple[str, ConnectionMode, WorkerType], int] = {}

# Track first-seen order to preserve test collection order
_first_seen_order: list[tuple[str, ConnectionMode, WorkerType]] = []

# Track max GPU requirement for any single test (for validation)
_max_test_gpu_requirement: int = 0
_max_test_name: str = ""

_needs_default_model: bool = False  # True if any e2e test lacks explicit model marker


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Scan collected tests to determine required workers.

    This runs after test collection but before tests execute.
    It extracts worker requirements from markers in test collection order,
    tracking the max count needed for each (model, mode, worker_type) combination.

    Also tracks the max GPU requirement for any single test for validation.
    """
    global _worker_counts, _first_seen_order, _needs_default_model
    global _max_test_gpu_requirement, _max_test_name

    from infra import MODEL_SPECS

    def track_worker(
        model_id: str, mode: ConnectionMode, worker_type: WorkerType, count: int
    ) -> None:
        """Track a worker requirement, updating max count if needed."""
        key = (model_id, mode, worker_type)
        if key not in _worker_counts:
            _first_seen_order.append(key)
            _worker_counts[key] = count
        else:
            _worker_counts[key] = max(_worker_counts[key], count)

    def calculate_test_gpus(
        model_id: str, prefill: int, decode: int, regular: int
    ) -> int:
        """Calculate GPU requirement for a single test."""
        if model_id not in MODEL_SPECS:
            return 0
        tp = MODEL_SPECS[model_id].get("tp", 1)
        return tp * (prefill + decode + regular)

    for item in items:
        # Extract model from marker or use default
        model_marker = item.get_closest_marker(PARAM_MODEL)
        model_id = model_marker.args[0] if model_marker and model_marker.args else None

        # Check parametrize for model
        if model_id is None:
            for marker in item.iter_markers("parametrize"):
                if marker.args and len(marker.args) >= 2:
                    param_name = marker.args[0]
                    if param_name == PARAM_MODEL or PARAM_MODEL in param_name:
                        param_values = marker.args[1]
                        if isinstance(param_values, (list, tuple)) and param_values:
                            model_id = param_values[0]  # First model in parametrize
                        break

        # Extract backends from parametrize
        backends: list[str] = []
        for marker in item.iter_markers("parametrize"):
            if marker.args and len(marker.args) >= 2:
                param_name = marker.args[0]
                param_values = marker.args[1]
                if param_name == PARAM_SETUP_BACKEND:
                    if isinstance(param_values, (list, tuple)):
                        backends.extend(param_values)

        # Check for workers marker (@pytest.mark.workers(...))
        workers_marker = item.get_closest_marker("workers")
        prefill_count = 0
        decode_count = 0
        regular_count = 1  # Default to 1 regular worker
        if workers_marker:
            prefill_count = workers_marker.kwargs.get("prefill") or 0
            decode_count = workers_marker.kwargs.get("decode") or 0
            regular_count = workers_marker.kwargs.get("count") or 1

        # Track if this test needs default model
        is_e2e = item.get_closest_marker("e2e") is not None
        if model_id is None and is_e2e:
            _needs_default_model = True
            model_id = DEFAULT_MODEL

        # Track worker requirements and calculate this test's GPU requirement
        test_gpus = 0
        if model_id and backends:
            for backend in backends:
                # "pd" backend means PD workers
                if backend == "pd":
                    mode = ConnectionMode.HTTP  # PD uses HTTP mode
                    # Default to 1 prefill + 1 decode if not specified
                    p_count = prefill_count if prefill_count > 0 else 1
                    d_count = decode_count if decode_count > 0 else 1
                    track_worker(model_id, mode, WorkerType.PREFILL, p_count)
                    track_worker(model_id, mode, WorkerType.DECODE, d_count)
                    test_gpus = max(
                        test_gpus, calculate_test_gpus(model_id, p_count, d_count, 0)
                    )
                else:
                    try:
                        mode = ConnectionMode(backend)
                    except ValueError:
                        # Cloud backend (openai, xai, etc.) - skip
                        continue

                    # Check if this backend also has PD workers
                    if prefill_count > 0 or decode_count > 0:
                        track_worker(model_id, mode, WorkerType.PREFILL, prefill_count)
                        track_worker(model_id, mode, WorkerType.DECODE, decode_count)
                        test_gpus = max(
                            test_gpus,
                            calculate_test_gpus(
                                model_id, prefill_count, decode_count, 0
                            ),
                        )
                    else:
                        # Regular worker
                        track_worker(model_id, mode, WorkerType.REGULAR, regular_count)
                        test_gpus = max(
                            test_gpus,
                            calculate_test_gpus(model_id, 0, 0, regular_count),
                        )

        elif model_id and is_e2e:
            # E2E test without explicit backend - will use HTTP by default
            track_worker(model_id, ConnectionMode.HTTP, WorkerType.REGULAR, 1)
            test_gpus = calculate_test_gpus(model_id, 0, 0, 1)

        # Track max GPU requirement across all tests
        if test_gpus > _max_test_gpu_requirement:
            _max_test_gpu_requirement = test_gpus
            _max_test_name = item.nodeid

    # Log results
    if _worker_counts:
        summary = []
        for key in _first_seen_order:
            model_id, mode, worker_type = key
            count = _worker_counts[key]
            if worker_type == WorkerType.REGULAR:
                summary.append(f"{model_id}:{mode.value}x{count}")
            else:
                summary.append(f"{model_id}:{mode.value}:{worker_type.value}x{count}")
        logger.info("Scanned worker requirements (in test order): %s", summary)
        logger.info(
            "Max GPU requirement for single test: %d (%s)",
            _max_test_gpu_requirement,
            _max_test_name,
        )
    else:
        logger.info("Scanned worker requirements: (none)")


def get_pool_requirements() -> list[WorkerIdentity]:
    """Build pool requirements from scanned test markers.

    Returns:
        List of WorkerIdentity objects to pre-launch.
        Each WorkerIdentity has (model_id, mode, worker_type, index).
        Requirements are ordered by first appearance in test collection order,
        so workers needed by earlier tests are launched first.

    Note:
        If a model's first test needs PD workers (prefill/decode), we skip
        pre-launching regular workers for that model (they'd be evicted
        immediately when PD workers are launched).
    """
    # Track which models have PD workers as their first requirement
    # These models shouldn't have regular workers pre-launched
    models_with_pd_first: set[str] = set()
    first_worker_type_per_model: dict[str, WorkerType] = {}

    for model_id, mode, worker_type in _first_seen_order:
        if model_id not in first_worker_type_per_model:
            first_worker_type_per_model[model_id] = worker_type
            if worker_type in (WorkerType.PREFILL, WorkerType.DECODE):
                models_with_pd_first.add(model_id)
                logger.info(
                    "Model %s has PD test first - skipping regular worker pre-launch",
                    model_id,
                )

    # Generate individual WorkerIdentity objects in first-seen order
    requirements: list[WorkerIdentity] = []
    for model_id, mode, worker_type in _first_seen_order:
        # Skip regular workers for models that have PD first
        if model_id in models_with_pd_first and worker_type == WorkerType.REGULAR:
            continue

        count = _worker_counts.get((model_id, mode, worker_type), 1)
        for i in range(count):
            requirements.append(WorkerIdentity(model_id, mode, worker_type, i))

    # Add default if no requirements
    if not requirements:
        requirements.append(WorkerIdentity(DEFAULT_MODEL, ConnectionMode.HTTP))

    return requirements


def validate_gpu_requirements() -> tuple[int, int]:
    """Check if there are enough GPUs for any single test.

    Returns:
        Tuple of (max_required_gpus, available_gpus).

    Note:
        We check the max requirement for any single test, not the sum.
        Workers can be evicted between tests, so we only need enough GPUs
        for the most demanding test.
    """
    # Count available GPUs
    available_gpus = 0
    try:
        import torch

        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
    except ImportError:
        pass

    return _max_test_gpu_requirement, available_gpus


def pytest_collection_finish(session: pytest.Session) -> None:
    """Validate GPU requirements after test collection.

    This runs after all tests are collected but before any tests execute.
    Fails fast if any single test requires more GPUs than available.
    """
    if not _worker_counts:
        return

    # Skip validation if model pool is disabled
    if os.environ.get(ENV_SKIP_MODEL_POOL, "").lower() in ("1", "true", "yes"):
        return

    max_required, available_gpus = validate_gpu_requirements()

    if max_required > available_gpus:
        sep = "=" * LOG_SEPARATOR_WIDTH
        raise pytest.UsageError(
            f"\n{sep}\n"
            f"GPU REQUIREMENTS EXCEEDED\n"
            f"{sep}\n"
            f"Test '{_max_test_name}' requires {max_required} GPUs\n"
            f"Available: {available_gpus} GPUs\n"
            f"\nOptions:\n"
            f"  1. Run tests that fit: pytest -k 'not {_max_test_name.split('::')[0]}'\n"
            f"  2. Reduce workers: @pytest.mark.workers(prefill=1, decode=1)\n"
            f"  3. Skip GPU tests: SKIP_MODEL_POOL=1 pytest\n"
            f"{sep}"
        )

    logger.info(
        "GPU validation passed: max %d required (by %s), %d available",
        max_required,
        _max_test_name,
        available_gpus,
    )


# ---------------------------------------------------------------------------
# Custom pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "model(name): mark test to use a specific model from MODEL_SPECS",
    )
    config.addinivalue_line(
        "markers",
        "backend(name): mark test to use a specific backend (grpc, http, openai, etc.)",
    )
    config.addinivalue_line(
        "markers",
        "workers(count=1, prefill=None, decode=None): "
        "worker configuration - use count for regular workers, "
        "or prefill/decode for PD disaggregation mode",
    )
    config.addinivalue_line(
        "markers",
        "gateway(policy='round_robin', timeout=None, extra_args=None): "
        "gateway/router configuration",
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

        # Create WorkerIdentity objects (regular workers only from env vars)
        requirements = [
            WorkerIdentity(m, b, WorkerType.REGULAR, 0)
            for m in models
            for b in backend_modes
        ]
        logger.info("Using env var requirements: %s", [str(r) for r in requirements])
    else:
        # Use scanned requirements from test markers
        requirements = get_pool_requirements()
        logger.info("Using scanned requirements: %s", [str(r) for r in requirements])

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
# Backend fixtures
# ---------------------------------------------------------------------------


def _get_marker_value(
    request: pytest.FixtureRequest,
    marker_name: str,
    arg_index: int = 0,
    default: Any = None,
) -> Any:
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
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    - @pytest.mark.workers(count=1): Number of regular workers behind router
    - @pytest.mark.workers(prefill=1, decode=1): PD worker configuration
    - @pytest.mark.gateway(policy="round_robin", timeout=60): Gateway configuration

    Returns:
        Tuple of (backend_name, model_path, openai_client, gateway)

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
        @pytest.mark.workers(count=3)
        @pytest.mark.gateway(policy="round_robin")
        @pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
        class TestLoadBalancing:
            ...

        # PD with custom configuration
        @pytest.mark.workers(prefill=2, decode=2)
        @pytest.mark.gateway(policy="round_robin")
        @pytest.mark.parametrize("setup_backend", ["pd"], indirect=True)
        class TestPDScaling:
            ...
    """
    import openai
    from infra import DEFAULT_ROUTER_TIMEOUT, Gateway, WorkerType

    backend_name = request.param

    # Skip if requested
    if os.environ.get(ENV_SKIP_BACKEND_SETUP, "").lower() in ("1", "true", "yes"):
        pytest.skip(f"{ENV_SKIP_BACKEND_SETUP} is set")

    # Get model from marker or env var or default
    model_id = _get_marker_value(request, "model")
    if model_id is None:
        model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    # Get worker configuration from marker
    workers_config = _get_marker_kwargs(
        request, "workers", defaults={"count": 1, "prefill": None, "decode": None}
    )

    # Get gateway configuration from marker
    gateway_config = _get_marker_kwargs(
        request,
        "gateway",
        defaults={
            "policy": "round_robin",
            "timeout": DEFAULT_ROUTER_TIMEOUT,
            "extra_args": None,
        },
    )

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

        # Get PD configuration from workers marker
        num_prefill = workers_config.get("prefill") or 1
        num_decode = workers_config.get("decode") or 1

        # Check GPU requirements
        required_gpus = num_prefill + num_decode
        gpu_count = torch.cuda.device_count()
        if gpu_count < required_gpus:
            pytest.skip(
                f"PD tests require {required_gpus} GPUs "
                f"({num_prefill} prefill + {num_decode} decode), found {gpu_count}"
            )

        # Try to use pre-launched PD workers, or launch additional ones if needed
        existing_prefills = model_pool.get_workers_by_type(model_id, WorkerType.PREFILL)
        existing_decodes = model_pool.get_workers_by_type(model_id, WorkerType.DECODE)

        # Calculate how many more we need (if any)
        missing_prefill = max(0, num_prefill - len(existing_prefills))
        missing_decode = max(0, num_decode - len(existing_decodes))

        if missing_prefill == 0 and missing_decode == 0:
            # Use pre-launched workers (we have enough)
            prefills = existing_prefills[:num_prefill]
            decodes = existing_decodes[:num_decode]
            logger.info(
                "Using pre-launched PD workers: %d prefill, %d decode",
                len(prefills),
                len(decodes),
            )
        else:
            # Build WorkerIdentity list for missing workers
            workers_to_launch: list[WorkerIdentity] = []
            for i in range(missing_prefill):
                workers_to_launch.append(
                    WorkerIdentity(
                        model_id,
                        ConnectionMode.HTTP,
                        WorkerType.PREFILL,
                        len(existing_prefills) + i,
                    )
                )
            for i in range(missing_decode):
                workers_to_launch.append(
                    WorkerIdentity(
                        model_id,
                        ConnectionMode.HTTP,
                        WorkerType.DECODE,
                        len(existing_decodes) + i,
                    )
                )

            logger.info(
                "Have %d/%d prefill, %d/%d decode. Launching %d more workers",
                len(existing_prefills),
                num_prefill,
                len(existing_decodes),
                num_decode,
                len(workers_to_launch),
            )
            new_instances = model_pool.launch_workers(
                workers_to_launch, startup_timeout=300
            )

            # Combine existing + newly launched
            new_prefills = [
                w for w in new_instances if w.worker_type == WorkerType.PREFILL
            ]
            new_decodes = [
                w for w in new_instances if w.worker_type == WorkerType.DECODE
            ]
            prefills = existing_prefills + new_prefills
            decodes = existing_decodes + new_decodes

        model_path = prefills[0].model_path if prefills else None

        # Launch PD gateway with configuration
        gateway = Gateway()
        gateway.start(
            prefill_workers=prefills,
            decode_workers=decodes,
            policy=gateway_config["policy"],
            timeout=gateway_config["timeout"],
            extra_args=gateway_config["extra_args"],
        )

        client = openai.OpenAI(
            base_url=f"{gateway.base_url}/v1",
            api_key="not-used",
        )

        logger.info(
            "Setup PD backend: model=%s, %d prefill + %d decode workers, "
            "gateway=%s, policy=%s",
            model_id,
            len(prefills),
            len(decodes),
            gateway.base_url,
            gateway_config["policy"],
        )

        try:
            yield backend_name, model_path, client, gateway
        finally:
            logger.info("Tearing down PD gateway")
            gateway.shutdown()
        return

    # Check if this is a local backend (grpc, http)
    try:
        connection_mode = ConnectionMode(backend_name)
        is_local = connection_mode in LOCAL_MODES
    except ValueError:
        is_local = False
        connection_mode = None

    # Local backends: use worker from pool + launch gateway
    if is_local:
        # Get number of workers from marker
        num_workers = workers_config.get("count") or 1

        try:
            if num_workers > 1:
                # Check existing workers
                existing = model_pool.get_workers_by_type(model_id, WorkerType.REGULAR)
                existing_for_mode = [w for w in existing if w.mode == connection_mode]

                if len(existing_for_mode) >= num_workers:
                    instances = existing_for_mode[:num_workers]
                else:
                    # Launch missing workers
                    missing = num_workers - len(existing_for_mode)
                    workers_to_launch = [
                        WorkerIdentity(
                            model_id,
                            connection_mode,
                            WorkerType.REGULAR,
                            len(existing_for_mode) + i,
                        )
                        for i in range(missing)
                    ]
                    new_instances = model_pool.launch_workers(
                        workers_to_launch, startup_timeout=300
                    )
                    instances = existing_for_mode + new_instances

                if not instances:
                    pytest.fail(f"Failed to get {num_workers} workers for {model_id}")
                worker_urls = [inst.worker_url for inst in instances]
                model_path = instances[0].model_path
            else:
                # Single worker - use existing get() method
                instance = model_pool.get(model_id, connection_mode)
                worker_urls = [instance.worker_url]
                model_path = instance.model_path
        except RuntimeError as e:
            pytest.fail(str(e))

        # Launch gateway with configuration
        gateway = Gateway()
        gateway.start(
            worker_urls=worker_urls,
            model_path=model_path,
            policy=gateway_config["policy"],
            timeout=gateway_config["timeout"],
            extra_args=gateway_config["extra_args"],
        )

        client = openai.OpenAI(
            base_url=f"{gateway.base_url}/v1",
            api_key="not-used",
        )

        logger.info(
            "Setup %s backend: model=%s, workers=%d, gateway=%s, policy=%s",
            backend_name,
            model_id,
            num_workers,
            gateway.base_url,
            gateway_config["policy"],
        )

        try:
            yield backend_name, model_path, client, gateway
        finally:
            logger.info("Tearing down gateway for %s backend", backend_name)
            gateway.shutdown()
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

    This launches a new Gateway for each test, pointing to workers from the pool.
    Use for tests that need isolated router state.

    Usage:
        @pytest.mark.parametrize("backend_router", ["grpc", "http"], indirect=True)
        def test_router_state(backend_router):
            gateway = backend_router
            # Test gateway-specific behavior
    """
    from infra import Gateway

    backend_name = request.param
    model_id = os.environ.get(ENV_MODEL, DEFAULT_MODEL)

    # Convert string to ConnectionMode
    connection_mode = ConnectionMode(backend_name)

    try:
        instance = model_pool.get(model_id, connection_mode)
    except KeyError:
        pytest.skip(f"Model {model_id}:{backend_name} not available in pool")
    except RuntimeError as e:
        pytest.fail(str(e))

    gateway = Gateway()
    gateway.start(
        worker_urls=[instance.worker_url],
        model_path=instance.model_path,
    )

    try:
        yield gateway
    finally:
        gateway.shutdown()
