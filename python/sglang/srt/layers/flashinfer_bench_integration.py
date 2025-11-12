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

For more information:
    - Documentation: docs/flashinfer_bench_integration.md
    - FlashInfer-Bench: https://bench.flashinfer.ai
"""

import logging
from functools import wraps
from typing import Any, Dict, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Check if FlashInfer-Bench is available
try:
    from flashinfer_bench import (
        TracingConfig,
        apply,
        disable_apply,
        disable_tracing,
        enable_tracing,
    )

    HAS_FLASHINFER_BENCH = True
except ImportError:
    HAS_FLASHINFER_BENCH = False
    logger.debug("FlashInfer-Bench not installed, kernel optimization disabled")


class SGLangFlashInferBenchIntegration:
    """Integration manager for FlashInfer-Bench in SGLang."""

    def __init__(self):
        self.enabled = False
        self.tracing_enabled = False
        self.apply_enabled = False
        self.dataset_path = None
        self.config = None
        self.tracing_runtime = None

    def initialize(
        self,
        enable_tracing: bool = False,
        enable_apply: bool = False,
        dataset_path: Optional[str] = None,
        tracing_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize FlashInfer-Bench integration.

        Args:
            enable_tracing: Whether to enable workload tracing
            enable_apply: Whether to enable kernel substitution
            dataset_path: Path to store/load traces
            tracing_config: Custom tracing configuration
        """
        if not HAS_FLASHINFER_BENCH:
            return

        # Use environment variables as defaults
        enable_tracing = enable_tracing or envs.FIB_ENABLE_TRACING.get()
        enable_apply = enable_apply or envs.FIB_ENABLE_APPLY.get()

        if not (enable_tracing or enable_apply):
            return

        self.enabled = True
        self.dataset_path = dataset_path or envs.FIB_DATASET_PATH.get()

        # Configure and enable tracing
        if enable_tracing:
            tracing_config = tracing_config or {}
            config = TracingConfig(
                input_dump_policy=tracing_config.get(
                    "input_dump_policy", "dump_non_float"
                ),
                filter_policy=tracing_config.get("filter_policy", "keep_first_by_axes"),
            )
            self.config = config

            # Actually enable tracing - this installs FlashInfer integrations!
            from flashinfer_bench import enable_tracing as fib_enable_tracing

            self.tracing_runtime = fib_enable_tracing(
                dataset_path=self.dataset_path,
                tracing_configs={"default": config},
            )
            self.tracing_enabled = True
            logger.info(
                f"FlashInfer-Bench tracing enabled, dataset path: {self.dataset_path}"
            )

        # Configure and enable kernel substitution
        if enable_apply:
            # Actually enable apply - this activates kernel substitution!
            from flashinfer_bench import enable_apply as fib_enable_apply

            fib_enable_apply()
            self.apply_enabled = True
            logger.info(
                f"FlashInfer-Bench kernel substitution enabled, dataset path: {self.dataset_path}"
            )

    def wrap_kernel(self, kernel_name: str):
        """Decorator to wrap a kernel with FlashInfer-Bench tracing/substitution.

        Args:
            kernel_name: Name of the kernel for tracing/substitution

        Returns:
            Decorator function that wraps the kernel
        """

        def decorator(func):
            if not self.enabled or not HAS_FLASHINFER_BENCH:
                return func

            @wraps(func)
            def wrapper(*args, **kwargs):
                # If tracing or apply is enabled, use flashinfer_bench.apply()
                # The apply() function handles both tracing and kernel substitution
                if self.tracing_enabled or self.apply_enabled:
                    try:
                        return apply(
                            def_name_or_resolver=kernel_name,
                            runtime_kwargs={"args": args, "kwargs": kwargs},
                            fallback=lambda **kw: func(*kw["args"], **kw["kwargs"]),
                        )
                    except Exception as e:
                        logger.debug(
                            f"FlashInfer-Bench operation failed for {kernel_name}: {e}"
                        )

                # Fall back to original implementation
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def start_tracing_context(self):
        """Start a tracing context for collecting workloads."""
        if not self.enabled or not self.tracing_enabled or not HAS_FLASHINFER_BENCH:
            return None

        return enable_tracing(
            dataset_path=self.dataset_path,
            tracing_configs={"default": self.config} if self.config else None,
        )

    def stop_tracing(self):
        """Stop tracing."""
        if HAS_FLASHINFER_BENCH:
            disable_tracing()

    def stop_apply(self):
        """Stop kernel substitution."""
        if HAS_FLASHINFER_BENCH:
            disable_apply()


# Global integration instance
_integration = SGLangFlashInferBenchIntegration()


def initialize_flashinfer_bench(**kwargs):
    """Initialize FlashInfer-Bench integration for SGLang.

    This should be called early in the server startup process.
    """
    _integration.initialize(**kwargs)


def get_integration():
    """Get the global FlashInfer-Bench integration instance."""
    return _integration


def wrap_attention_kernel(kernel_name: str):
    """Decorator to wrap attention kernels with FlashInfer-Bench.

    Example:
        @wrap_attention_kernel("flashinfer_prefill_attention")
        def forward_extend(self, q, k, v, ...):
            ...
    """
    return _integration.wrap_kernel(kernel_name)
