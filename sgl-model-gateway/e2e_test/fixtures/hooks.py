"""Pytest hooks for E2E test collection and validation.

This module handles:
- Test collection: Scanning markers to determine required workers
- GPU validation: Ensuring sufficient GPUs for test requirements
- Marker registration: Defining custom pytest markers
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from infra import ConnectionMode, WorkerIdentity, WorkerType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test collection state
# ---------------------------------------------------------------------------

# Track max worker counts: (model_id, mode, worker_type) -> max_count
_worker_counts: dict[tuple["ConnectionMode", "WorkerType"], int] = {}

# Track first-seen order to preserve test collection order
_first_seen_order: list[tuple[str, "ConnectionMode", "WorkerType"]] = []

# Track max GPU requirement for any single test (for validation)
_max_test_gpu_requirement: int = 0
_max_test_name: str = ""

_needs_default_model: bool = False


def reset_collection_state() -> None:
    """Reset collection state (useful for testing)."""
    global _worker_counts, _first_seen_order
    global _max_test_gpu_requirement, _max_test_name, _needs_default_model
    _worker_counts = {}
    _first_seen_order = []
    _max_test_gpu_requirement = 0
    _max_test_name = ""
    _needs_default_model = False


def get_worker_counts() -> dict:
    """Get the worker counts dictionary."""
    return _worker_counts


def get_first_seen_order() -> list:
    """Get the first-seen order list."""
    return _first_seen_order


def get_max_gpu_requirement() -> tuple[int, str]:
    """Get the max GPU requirement and test name."""
    return _max_test_gpu_requirement, _max_test_name


def needs_default_model() -> bool:
    """Check if any test needs the default model."""
    return _needs_default_model


# ---------------------------------------------------------------------------
# Test collection hook
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Scan collected tests to determine required workers.

    This runs after test collection but before tests execute.
    It extracts worker requirements from markers in test collection order,
    tracking the max count needed for each (model, mode, worker_type) combination.
    """
    global _worker_counts, _first_seen_order, _needs_default_model
    global _max_test_gpu_requirement, _max_test_name

    from infra import (
        DEFAULT_MODEL,
        LOG_SEPARATOR_WIDTH,
        MODEL_SPECS,
        PARAM_MODEL,
        PARAM_SETUP_BACKEND,
        ConnectionMode,
        WorkerType,
    )

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
        # First check the class directly (handles inheritance correctly)
        model_id = None
        if hasattr(item, "cls") and item.cls is not None:
            for marker in (
                item.cls.pytestmark if hasattr(item.cls, "pytestmark") else []
            ):
                if marker.name == PARAM_MODEL and marker.args:
                    model_id = marker.args[0]
                    break
        # Fall back to get_closest_marker for method-level markers
        if model_id is None:
            model_marker = item.get_closest_marker(PARAM_MODEL)
            model_id = (
                model_marker.args[0] if model_marker and model_marker.args else None
            )

        # Check parametrize for model
        if model_id is None:
            for marker in item.iter_markers("parametrize"):
                if marker.args and len(marker.args) >= 2:
                    param_name = marker.args[0]
                    if param_name == PARAM_MODEL or PARAM_MODEL in param_name:
                        param_values = marker.args[1]
                        if isinstance(param_values, (list, tuple)) and param_values:
                            model_id = param_values[0]
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

        # Check for workers marker
        workers_marker = item.get_closest_marker("workers")
        prefill_count = 0
        decode_count = 0
        regular_count = 1
        if workers_marker:
            prefill_count = workers_marker.kwargs.get("prefill") or 0
            decode_count = workers_marker.kwargs.get("decode") or 0
            regular_count = workers_marker.kwargs.get("count") or 1

        # Track if this test needs default model
        is_e2e = item.get_closest_marker("e2e") is not None
        if model_id is None and is_e2e:
            _needs_default_model = True
            model_id = DEFAULT_MODEL

        # Track worker requirements
        test_gpus = 0
        if model_id and backends:
            for backend in backends:
                if backend == "pd":
                    mode = ConnectionMode.HTTP
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
                        continue

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
                        track_worker(model_id, mode, WorkerType.REGULAR, regular_count)
                        test_gpus = max(
                            test_gpus,
                            calculate_test_gpus(model_id, 0, 0, regular_count),
                        )

        elif model_id and is_e2e:
            track_worker(model_id, ConnectionMode.HTTP, WorkerType.REGULAR, 1)
            test_gpus = calculate_test_gpus(model_id, 0, 0, 1)

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


# ---------------------------------------------------------------------------
# Pool requirements
# ---------------------------------------------------------------------------


def get_pool_requirements() -> list["WorkerIdentity"]:
    """Build pool requirements from scanned test markers.

    Returns:
        List of WorkerIdentity objects to pre-launch.
    """
    from infra import DEFAULT_MODEL, ConnectionMode, WorkerIdentity, WorkerType

    # Track which models have PD workers as their first requirement
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
        if model_id in models_with_pd_first and worker_type == WorkerType.REGULAR:
            continue

        count = _worker_counts.get((model_id, mode, worker_type), 1)
        for i in range(count):
            requirements.append(WorkerIdentity(model_id, mode, worker_type, i))

    if not requirements:
        requirements.append(WorkerIdentity(DEFAULT_MODEL, ConnectionMode.HTTP))

    return requirements


# ---------------------------------------------------------------------------
# GPU validation
# ---------------------------------------------------------------------------


def _count_gpus_without_cuda() -> int:
    """Count available GPUs without initializing CUDA.

    Uses nvidia-smi to avoid CUDA initialization, which is critical for
    pytest-parallel compatibility. CUDA cannot be re-initialized after a fork.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return len([line for line in result.stdout.strip().split("\n") if line])
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return 0


def validate_gpu_requirements() -> tuple[int, int]:
    """Check if there are enough GPUs for any single test.

    Uses nvidia-smi instead of torch.cuda to avoid CUDA initialization,
    which would break pytest-parallel (CUDA cannot be re-initialized after fork).

    Returns:
        Tuple of (max_required_gpus, available_gpus).
    """
    available_gpus = _count_gpus_without_cuda()
    return _max_test_gpu_requirement, available_gpus


def pytest_collection_finish(session: pytest.Session) -> None:
    """Validate GPU requirements after test collection."""
    from infra import ENV_SKIP_MODEL_POOL, LOG_SEPARATOR_WIDTH

    if not _worker_counts:
        return

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
# Marker registration
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
    config.addinivalue_line(
        "markers",
        "thread_unsafe: mark test as incompatible with parallel thread execution",
    )
    config.addinivalue_line(
        "markers",
        "storage(backend): mark test to use a specific history storage backend "
        "(memory, oracle). Default is memory.",
    )


# ---------------------------------------------------------------------------
# Parallel execution support
# ---------------------------------------------------------------------------


def is_parallel_execution(config: pytest.Config) -> bool:
    """Check if tests are running in parallel mode (pytest-parallel).

    Returns True if --tests-per-worker > 1, indicating concurrent thread execution.
    """
    # pytest-parallel adds the 'tests_per_worker' option
    tests_per_worker = getattr(config.option, "tests_per_worker", None)
    if tests_per_worker is None:
        return False

    if tests_per_worker == "auto":
        return True

    try:
        return int(tests_per_worker) > 1
    except (ValueError, TypeError):
        return False


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip thread_unsafe tests when running in parallel mode."""
    if is_parallel_execution(item.config):
        marker = item.get_closest_marker("thread_unsafe")
        if marker:
            reason = marker.kwargs.get("reason", "Test is not thread-safe")
            pytest.skip(f"Skipping in parallel mode: {reason}")
