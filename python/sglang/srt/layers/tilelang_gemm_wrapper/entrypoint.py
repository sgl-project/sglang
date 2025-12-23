"""TileLang GEMM Wrapper entry point."""
import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.tilelang_gemm_wrapper.configurer import (
    ENABLE_TILELANG_GEMM,
    TILELANG_GEMM_CONFIG_DIR,
)

try:
    from sglang.srt.entrypoints.warmup import warmup as warmup_decorator
    _WARMUP_DECORATOR_AVAILABLE = True
except ImportError:
    _WARMUP_DECORATOR_AVAILABLE = False
    warmup_decorator = lambda name: lambda fn: fn  # noqa: E731

logger = logging.getLogger(__name__)

_wrapper = None
_wrapper_initialized = False


def _get_wrapper():
    """Get or initialize TileLangGEMMWrapper instance."""
    global _wrapper, _wrapper_initialized

    if _wrapper_initialized:
        return _wrapper

    if not ENABLE_TILELANG_GEMM:
        _wrapper_initialized = True
        return None

    try:
        from sglang.srt.layers.tilelang_gemm_wrapper.core import TileLangGEMMWrapper

        config_dir = TILELANG_GEMM_CONFIG_DIR if TILELANG_GEMM_CONFIG_DIR else None
        _wrapper = TileLangGEMMWrapper(config_dir=config_dir)
        logger.info(f"TileLangGEMMWrapper initialized (config_dir={config_dir})")
    except ImportError as e:
        logger.warning(f"Failed to import TileLangGEMMWrapper: {e}")
        _wrapper = None
    except Exception as e:
        logger.warning(f"Failed to initialize TileLangGEMMWrapper: {e}")
        _wrapper = None

    _wrapper_initialized = True
    return _wrapper


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
) -> None:
    """Execute FP8 GEMM: out = A @ B^T with blockwise scaling.

    Args:
        lhs: (A_fp8, A_scale) tuple
        rhs: (B_fp8, B_scale) tuple
        out: (M, N), bfloat16, output buffer
    """
    wrapper = _get_wrapper()
    if wrapper is None:
        raise RuntimeError(
            "TileLang GEMM is not available. "
            "Please ensure tilelang is installed and SGLANG_ENABLE_TILELANG_GEMM=1"
        )

    A_fp8, A_scale = lhs
    B_fp8, B_scale = rhs
    wrapper.gemm(A_fp8, B_fp8, A_scale, B_scale, out)


def warmup(shapes: List[Tuple[int, int, int]]) -> None:
    """Pre-compile kernels for specified shapes."""
    wrapper = _get_wrapper()
    if wrapper is None:
        logger.warning("TileLang GEMM warmup skipped: wrapper not available")
        return

    logger.info(f"TileLang GEMM warming up {len(shapes)} shapes...")
    wrapper.warmup(shapes)
    logger.info("TileLang GEMM warmup complete")


def get_kernel_info(M: int, N: int, K: int) -> Optional[dict]:
    """Get kernel info for given shape (for debugging)."""
    wrapper = _get_wrapper()
    if wrapper is None:
        return None
    return wrapper.get_kernel_info(M, N, K)


def clear_cache() -> None:
    """Clear all kernel caches."""
    wrapper = _get_wrapper()
    if wrapper is not None:
        wrapper.clear_cache()
        logger.info("TileLang GEMM cache cleared")


def is_available() -> bool:
    """Check if TileLang GEMM is available."""
    return ENABLE_TILELANG_GEMM and _get_wrapper() is not None


def list_available_configs() -> List[Tuple[int, int]]:
    """List all available (N, K) configurations."""
    wrapper = _get_wrapper()
    if wrapper is None:
        return []
    return wrapper.list_available_configs()


def warmup_common_shapes(
    m_values: Optional[List[int]] = None,
    nk_shapes: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """Warmup common shapes."""
    wrapper = _get_wrapper()
    if wrapper is None:
        logger.warning("TileLang GEMM warmup skipped: wrapper not available")
        return

    if m_values is None:
        m_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    if nk_shapes is None:
        nk_shapes = wrapper.list_available_configs()

    if not nk_shapes:
        logger.warning("No TileLang GEMM configurations available for warmup")
        return

    shapes = [(M, N, K) for N, K in nk_shapes for M in m_values]

    logger.info(f"TileLang GEMM warming up {len(shapes)} shapes "
                f"({len(nk_shapes)} (N,K) x {len(m_values)} M values)...")

    wrapper.warmup(shapes)
    logger.info("TileLang GEMM warmup complete")


if _WARMUP_DECORATOR_AVAILABLE:
    @warmup_decorator("compile-tilelang-gemm")
    async def _sglang_warmup_tilelang_gemm(disaggregation_mode: str, tokenizer_manager):
        """Warmup function for sglang's warmup system."""
        logger.info("Running TileLang GEMM warmup via sglang warmup system...")
        warmup_common_shapes()
