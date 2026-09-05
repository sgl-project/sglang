"""
This module provides JIT-compiled CUDA kernels for fusing multiple tensor
copy operations into single kernel launches, reducing kernel launch overhead
and improving CUDA graph replay performance.

The kernels are compiled on-demand using TVM FFI and cached for subsequent use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def _jit_dsa_fused_store_module(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> Module:
    """
    Build a JIT module that exposes the ordinary fused cache store plus the
    packed CP transport helpers.
    """
    args = make_cpp_args(key_dtype, indices_dtype, page_size, is_arch_support_pdl())
    return load_jit(
        "fused_store_index_k_cache",
        *args,
        cuda_files=["dsa/fused_store_index_cache.cuh"],
        cuda_wrappers=[
            (
                "fused_store_index_k_cache",
                # - Float  = bf16_t (sgl_kernel/type.cuh)
                # - IndicesT = int64_t (out_cache_loc is int64 in SGLang SetKAndS)
                # - kPageSize = 64 (CUDA DSA)
                f"FusedStoreCacheIndexerKernel<{args}>::run",
            ),
            (
                "fused_quantize_index_k_packed",
                f"FusedQuantizePackedIndexerKernel<{args}>::run",
            ),
            (
                "store_packed_index_k_cache",
                f"StorePackedCacheIndexerKernel<{args}>::run",
            ),
            (
                "store_rank_major_packed_index_k_cache",
                f"StoreRankMajorPackedCacheIndexerKernel<{args}>::run",
            ),
        ],
    )


@cache_once
def can_use_dsa_fused_store(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> bool:
    logger = logging.getLogger(__name__)
    try:
        _jit_dsa_fused_store_module(key_dtype, indices_dtype, page_size)
        return True
    except Exception as e:
        logger.warning(f"Failed to load dsa fused store JIT kernel: {e}")
        return False


@debug_kernel_api
def fused_store_index_k_cache(
    key: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int = 64,
) -> None:
    """
    Fused: quantize bf16 key (N,128) -> fp8 + fp32 scale and write into DSATokenToKVPool.index_k_with_scale_buffer.

    key:            (num_tokens, 128) bf16 (or reshapeable to it)
    index_k_with_scale:  (num_pages, 64*(128+4)) uint8
    out_cache_loc:       (num_tokens,) int64 token indices in TokenToKVPool
    """
    assert key.is_cuda
    assert index_k_with_scale.is_cuda
    assert out_cache_loc.is_cuda

    # 1) normalize shapes
    if key.dim() != 2:
        key = key.view(-1, key.shape[-1])
    assert key.shape[1] == 128, f"expected key last-dim=128, got {key.shape}"

    # 2) dtypes
    assert key.dtype == torch.bfloat16, f"{key.dtype=}"
    assert index_k_with_scale.dtype == torch.uint8, f"{index_k_with_scale.dtype=}"
    assert out_cache_loc.dtype == torch.int64, f"{out_cache_loc.dtype=}"

    # 3) contiguity
    if not key.is_contiguous():
        key = key.contiguous()
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()
    if not index_k_with_scale.is_contiguous():
        index_k_with_scale = index_k_with_scale.contiguous()

    module = _jit_dsa_fused_store_module(key.dtype, out_cache_loc.dtype, page_size)
    module.fused_store_index_k_cache(key, index_k_with_scale, out_cache_loc)


@debug_kernel_api
def fused_quantize_index_k_packed(
    key: torch.Tensor,
    packed: Optional[torch.Tensor] = None,
    page_size: int = 64,
) -> torch.Tensor:
    """Quantize bf16 index-K rows into contiguous ``[N, 132]`` records.

    Each record contains 128 fp8 bytes followed by the exact fp32 scale used
    by :func:`fused_store_index_k_cache`.  The row-contiguous representation
    is intended for a single CP all-gather.
    """
    assert key.is_cuda
    if key.dim() != 2:
        key = key.view(-1, key.shape[-1])
    assert key.shape[1] == 128, f"expected key last-dim=128, got {key.shape}"
    assert key.dtype == torch.bfloat16, f"{key.dtype=}"
    if not key.is_contiguous():
        key = key.contiguous()
    if packed is None:
        packed = torch.empty((key.shape[0], 132), dtype=torch.uint8, device=key.device)
    assert packed.is_cuda and packed.device == key.device
    assert packed.dtype == torch.uint8, f"{packed.dtype=}"
    assert packed.shape == (key.shape[0], 132), (
        f"expected packed shape {(key.shape[0], 132)}, got {packed.shape}"
    )
    assert packed.is_contiguous()

    # The module specialization includes the established int64/page-64 cache
    # ABI even though this first helper only consumes ``key`` and ``packed``.
    module = _jit_dsa_fused_store_module(key.dtype, torch.int64, page_size)
    module.fused_quantize_index_k_packed(key, packed)
    return packed


@debug_kernel_api
def store_packed_index_k_cache(
    packed: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int = 64,
) -> None:
    """Store contiguous packed index-K rows into the paged cache unchanged."""
    assert packed.is_cuda
    assert index_k_with_scale.is_cuda
    assert out_cache_loc.is_cuda
    assert packed.dtype == torch.uint8, f"{packed.dtype=}"
    assert packed.dim() == 2 and packed.shape[1] == 132, f"{packed.shape=}"
    assert index_k_with_scale.dtype == torch.uint8, f"{index_k_with_scale.dtype=}"
    assert out_cache_loc.dtype == torch.int64, f"{out_cache_loc.dtype=}"
    assert packed.shape[0] == out_cache_loc.shape[0]
    assert packed.is_contiguous()
    assert index_k_with_scale.is_contiguous()
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()

    module = _jit_dsa_fused_store_module(torch.bfloat16, out_cache_loc.dtype, page_size)
    module.store_packed_index_k_cache(packed, index_k_with_scale, out_cache_loc)


@debug_kernel_api
def store_rank_major_packed_index_k_cache(
    packed: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    cp_size: int,
    page_size: int = 64,
) -> None:
    """Directly store an equal-shard interleave CP all-gather result.

    ``packed`` stays in rank-major all-gather order.  The CUDA kernel folds
    the interleave source-row lookup into the ordinary paged cache scatter,
    avoiding a global ``index_select`` and reordered transport temporary.
    The production adapter calls this only from the default-off packed CP
    experiment after validating equal power-of-two Interleave shards.
    """
    assert packed.is_cuda
    assert index_k_with_scale.is_cuda
    assert out_cache_loc.is_cuda
    assert packed.dtype == torch.uint8, f"{packed.dtype=}"
    assert packed.dim() == 2 and packed.shape[1] == 132, f"{packed.shape=}"
    assert index_k_with_scale.dtype == torch.uint8, f"{index_k_with_scale.dtype=}"
    assert out_cache_loc.dtype == torch.int64, f"{out_cache_loc.dtype=}"
    assert packed.shape[0] == out_cache_loc.shape[0]
    assert int(cp_size) > 1
    assert int(cp_size) & (int(cp_size) - 1) == 0
    assert packed.shape[0] % int(cp_size) == 0
    assert packed.is_contiguous()
    assert index_k_with_scale.is_contiguous()
    if not out_cache_loc.is_contiguous():
        out_cache_loc = out_cache_loc.contiguous()

    module = _jit_dsa_fused_store_module(torch.bfloat16, out_cache_loc.dtype, page_size)
    module.store_rank_major_packed_index_k_cache(
        packed, index_k_with_scale, out_cache_loc, int(cp_size)
    )
