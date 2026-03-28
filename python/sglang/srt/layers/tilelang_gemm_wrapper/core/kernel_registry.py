"""TileLang GEMM kernel registry."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

KERNEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
_initialized = False


def _initialize_registry():
    """Lazily initialize the kernel registry."""
    global KERNEL_REGISTRY, _initialized

    if _initialized:
        return

    try:
        import tilelang  # noqa: F401
    except ImportError:
        logger.warning("tilelang not installed, kernel registry will be empty")
        _initialized = True
        return

    try:
        from sglang.srt.layers.tilelang_gemm_wrapper.core.kernels import (
            base_kernel_factory,
            splitK_kernel_factory,
            splitK_swapAB_kernel_factory,
            swapAB_kernel_factory,
        )

        KERNEL_REGISTRY.update(
            {
                "base": {
                    "factory": base_kernel_factory,
                    "has_split_k": False,
                    "is_swap_ab": False,
                    "scale_shm_key": "a_scale_shm",
                },
                "swapAB": {
                    "factory": swapAB_kernel_factory,
                    "has_split_k": False,
                    "is_swap_ab": True,
                    "scale_shm_key": "b_scale_shm",
                },
                "splitK": {
                    "factory": splitK_kernel_factory,
                    "has_split_k": True,
                    "is_swap_ab": False,
                    "scale_shm_key": "a_scale_shm",
                },
                "splitK_swapAB": {
                    "factory": splitK_swapAB_kernel_factory,
                    "has_split_k": True,
                    "is_swap_ab": True,
                    "scale_shm_key": "b_scale_shm",
                },
            }
        )

        logger.info(f"Loaded {len(KERNEL_REGISTRY)} kernel types")

    except ImportError as e:
        logger.warning(f"Failed to import kernel factories: {e}")

    _initialized = True


def get_kernel_factory(kernel_type: str) -> dict:
    """Get kernel factory info by type."""
    _initialize_registry()

    if kernel_type not in KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown kernel type: {kernel_type}. "
            f"Available types: {list(KERNEL_REGISTRY.keys())}"
        )
    return KERNEL_REGISTRY[kernel_type]


def list_kernel_types() -> List[str]:
    """List all available kernel types."""
    _initialize_registry()
    return list(KERNEL_REGISTRY.keys())


def is_registry_available() -> bool:
    """Check if the kernel registry is available."""
    _initialize_registry()
    return len(KERNEL_REGISTRY) > 0
