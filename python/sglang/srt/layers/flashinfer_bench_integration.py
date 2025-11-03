"""
FlashInfer-Bench integration for SGLang.

This module provides integration with FlashInfer-Bench for:
1. Automatic workload tracing
2. Kernel benchmarking
3. Dynamic kernel substitution

Usage:
    # Enable tracing to collect workloads
    FIB_ENABLE_TRACING=1 python -m sglang.launch_server ...

    # Enable kernel substitution for optimization
    FIB_ENABLE_APPLY=1 python -m sglang.launch_server ...

    # Set custom dataset path
    FIB_DATASET_PATH=/path/to/traces python -m sglang.launch_server ...
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

# Check if FlashInfer-Bench is available
try:
    import flashinfer_bench
    from flashinfer_bench import (
        TracingConfig,
        enable_tracing,
        enable_apply,
        disable_tracing,
        disable_apply,
        apply
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
        enable_tracing = enable_tracing or os.environ.get("FIB_ENABLE_TRACING") == "1"
        enable_apply = enable_apply or os.environ.get("FIB_ENABLE_APPLY") == "1"
        dataset_path = dataset_path or os.environ.get("FIB_DATASET_PATH")

        if not (enable_tracing or enable_apply):
            return

        self.enabled = True
        self.dataset_path = dataset_path or os.path.expanduser("~/.cache/flashinfer_bench/dataset")

        # Configure tracing
        if enable_tracing:
            config = TracingConfig(
                input_dump_policy=tracing_config.get("input_dump_policy", "dump_non_float") if tracing_config else "dump_non_float",
                filter_policy=tracing_config.get("filter_policy", "keep_first_by_axes") if tracing_config else "keep_first_by_axes"
            )
            self.config = config
            self.tracing_enabled = True
            logger.info(f"FlashInfer-Bench tracing enabled, dataset path: {self.dataset_path}")

        # Configure kernel substitution
        if enable_apply:
            self.apply_enabled = True
            logger.info(f"FlashInfer-Bench kernel substitution enabled, dataset path: {self.dataset_path}")

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
                # If apply is enabled, try to substitute with optimized kernel
                if self.apply_enabled:
                    try:
                        return apply(
                            def_name_or_resolver=kernel_name,
                            runtime_kwargs={"args": args, "kwargs": kwargs},
                            fallback=lambda **kw: func(*kw["args"], **kw["kwargs"])
                        )
                    except Exception as e:
                        logger.debug(f"Kernel substitution failed for {kernel_name}: {e}")

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
            tracing_configs={"default": self.config} if self.config else None
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