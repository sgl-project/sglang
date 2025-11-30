"""
FlashInfer-Bench integration for SGLang.

This module provides integration with FlashInfer-Bench for automatic kernel
optimization based on production workloads.

Features:
    - Automatic workload tracing
    - Kernel benchmarking
    - Dynamic kernel substitution
    - Zero-code integration via environment variables

Quick Start:
    # 1. Install FlashInfer-Bench
    pip install flashinfer-bench

    # 2. Enable tracing to collect workloads
    FIB_ENABLE_TRACING=1 python -m sglang.launch_server --model-path meta-llama/Llama-3-8b

    # 3. Enable kernel substitution for optimization
    FIB_ENABLE_APPLY=1 python -m sglang.launch_server --model-path meta-llama/Llama-3-8b

Environment Variables:
    FIB_ENABLE_TRACING: Enable workload collection (default: False)
    FIB_ENABLE_APPLY: Enable kernel substitution (default: False)
    FIB_DATASET_PATH: Path to store/load traces (default: ~/.cache/flashinfer_bench/dataset)

Command-Line Arguments:
    --enable-flashinfer-bench-tracing: Enable tracing
    --enable-flashinfer-bench-apply: Enable kernel substitution
    --flashinfer-bench-dataset-path: Custom dataset path

Example:
    # Collect workloads during production
    FIB_ENABLE_TRACING=1 FIB_DATASET_PATH=./traces python -m sglang.launch_server ...

    # Benchmark (offline)
    flashinfer-bench run --local ./traces

    # Deploy optimizations
    FIB_ENABLE_APPLY=1 FIB_DATASET_PATH=./traces python -m sglang.launch_server ...

How It Works:
    FlashInfer-Bench automatically patches FlashInfer's internal functions when
    enable_tracing() or enable_apply() is called. This means:

    1. When tracing is enabled, FlashInfer attention calls are intercepted and
       workload parameters (tensor shapes, batch sizes, etc.) are collected.

    2. When apply is enabled, FlashInfer attention calls are routed to the
       best-performing kernel based on previously benchmarked results.

    No manual kernel wrapping is needed - it's all automatic!

For more information:
    - Documentation: docs/flashinfer_bench_integration.md
    - FlashInfer-Bench: https://bench.flashinfer.ai
"""

import logging
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Check if FlashInfer-Bench is available
HAS_FLASHINFER_BENCH = False

try:
    from flashinfer_bench import (
        disable_apply,
        disable_tracing,
        enable_apply,
        enable_tracing,
    )

    HAS_FLASHINFER_BENCH = True
except ImportError:
    logger.debug("FlashInfer-Bench not installed, kernel optimization disabled")


# Track initialization state
_initialized = False
_tracing_enabled = False
_apply_enabled = False


def initialize_flashinfer_bench(
    tracing: Optional[bool] = None,
    apply: Optional[bool] = None,
    dataset_path: Optional[str] = None,
) -> bool:
    """Initialize FlashInfer-Bench integration for SGLang.

    FlashInfer-Bench automatically patches FlashInfer's internal functions
    (BatchPrefillWithPagedKVCacheWrapper, BatchDecodeWithPagedKVCacheWrapper, etc.)
    when tracing or apply is enabled. No manual kernel wrapping is needed.

    Args:
        tracing: Whether to enable workload tracing. If None, uses FIB_ENABLE_TRACING env var.
        apply: Whether to enable kernel substitution. If None, uses FIB_ENABLE_APPLY env var.
        dataset_path: Path to store/load traces. If None, uses FIB_DATASET_PATH env var.

    Returns:
        True if FlashInfer-Bench was successfully initialized, False otherwise.

    Note:
        This should be called early in the server startup process, before any
        FlashInfer operations are performed.
    """
    global _initialized, _tracing_enabled, _apply_enabled

    if not HAS_FLASHINFER_BENCH:
        logger.debug("FlashInfer-Bench not available, skipping initialization")
        return False

    # Use environment variables as defaults
    do_tracing = tracing if tracing is not None else envs.FIB_ENABLE_TRACING.get()
    do_apply = apply if apply is not None else envs.FIB_ENABLE_APPLY.get()
    path = dataset_path or envs.FIB_DATASET_PATH.get()

    if not (do_tracing or do_apply):
        logger.debug("FlashInfer-Bench not enabled (neither tracing nor apply)")
        return False

    try:
        # Enable tracing if requested
        # Pass tracing_configs=None to use FULL_TRACING_CONFIGS default,
        # which contains proper definition-name mappings for FlashInfer kernels
        if do_tracing:
            enable_tracing(dataset_path=path, tracing_configs=None)
            _tracing_enabled = True
            logger.info(f"FlashInfer-Bench tracing enabled, dataset path: {path}")

        # Enable kernel substitution if requested
        if do_apply:
            enable_apply(dataset_path=path)
            _apply_enabled = True
            logger.info(
                f"FlashInfer-Bench kernel substitution enabled, dataset path: {path}"
            )

        _initialized = True
        return True

    except Exception as e:
        logger.warning(f"Failed to initialize FlashInfer-Bench: {e}")
        return False


def shutdown_flashinfer_bench() -> None:
    """Shutdown FlashInfer-Bench integration.

    This flushes any pending trace data and disables tracing/apply.
    Safe to call even if FlashInfer-Bench was not initialized.
    """
    global _initialized, _tracing_enabled, _apply_enabled

    if not HAS_FLASHINFER_BENCH:
        return

    try:
        if _tracing_enabled:
            disable_tracing()
            _tracing_enabled = False
            logger.debug("FlashInfer-Bench tracing disabled")

        if _apply_enabled:
            disable_apply()
            _apply_enabled = False
            logger.debug("FlashInfer-Bench apply disabled")

        _initialized = False
    except Exception as e:
        logger.debug(f"Error during FlashInfer-Bench shutdown: {e}")


def is_flashinfer_bench_enabled() -> bool:
    """Check if FlashInfer-Bench integration is enabled.

    Returns:
        True if either tracing or apply is enabled.
    """
    return _initialized and (_tracing_enabled or _apply_enabled)


def is_tracing_enabled() -> bool:
    """Check if FlashInfer-Bench tracing is enabled.

    Returns:
        True if tracing is enabled.
    """
    return _tracing_enabled


def is_apply_enabled() -> bool:
    """Check if FlashInfer-Bench kernel substitution is enabled.

    Returns:
        True if kernel substitution is enabled.
    """
    return _apply_enabled
