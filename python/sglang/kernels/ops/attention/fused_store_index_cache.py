"""
This module provides JIT-compiled CUDA kernels for fusing multiple tensor
copy operations into single kernel launches, reducing kernel launch overhead
and improving CUDA graph replay performance.

The kernels are compiled on-demand using TVM FFI and cached for subsequent use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
    Build a JIT module that exposes:
      module.fused_store_index_k_cache(input_bf16, index_k_with_scale_u8, loc_i64)
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
            )
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


@cache_once
def _jit_dsa_sharded_store_module(page_size: int) -> Module:
    args = make_cpp_args(torch.bfloat16, torch.int64, page_size, is_arch_support_pdl())
    return load_jit(
        "fused_store_sharded_index_k_cache",
        *args,
        cuda_files=["dsa/fused_store_index_cache.cuh"],
        cuda_wrappers=[("store", f"FusedStoreCacheIndexerKernel<{args}>::run_sharded")],
    )


@cache_once
def can_use_dsa_sharded_store(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> bool:
    """Probe the optional CUDA BF16/FP8 dual-store kernel before dispatch."""
    if (
        key_dtype != torch.bfloat16
        or indices_dtype != torch.int64
        or page_size != 64
        or not torch.cuda.is_available()
        or torch.version.hip is not None
    ):
        return False
    try:
        _jit_dsa_sharded_store_module(page_size)
        return True
    except Exception as exc:
        logger.warning("Failed to load sharded DSA fused store JIT kernel: %s", exc)
        return False


@debug_kernel_api
def fused_store_sharded_index_k_cache(
    key: torch.Tensor,
    scratch: torch.Tensor,
    scratch_loc: torch.Tensor,
    shard_cache: torch.Tensor,
    logical_loc: torch.Tensor,
    shard_size: int,
    shard_rank: int,
    page_size: int = 64,
) -> None:
    """Quantize once, stage every row and persist only this rank's pages.

    ``key`` contains BF16 rows with 128 elements. Quantization uses the normal
    FP32 abs-max scale, not a rounded/power-of-two scale. Both destination
    buffers have shape ``(num_pages, page_size * 132)`` and retain the DSA
    layout: all FP8 K rows, followed by all FP32 scales within each page.

    Locations are valid, nonnegative token addresses (including reserved row
    zero). Padding must already point to a valid scratch trash/reserved row;
    there is no negative-index skip convention. Non-owned rows still update
    scratch. Like the ordinary store, concurrent duplicate destinations with
    different keys are not supported. Inputs may be strided; output buffers
    must be contiguous and disjoint.
    """
    assert 1 < shard_size <= 2**32 - 1 and 0 <= shard_rank < shard_size
    assert page_size == 64
    assert key.ndim >= 2 and key.shape[-1] == 128
    assert key.dtype == torch.bfloat16
    assert scratch.dtype == shard_cache.dtype == torch.uint8
    assert scratch_loc.dtype == logical_loc.dtype == torch.int64
    assert scratch_loc.ndim == logical_loc.ndim == 1
    assert scratch.ndim == shard_cache.ndim == 2
    assert scratch.shape[1] == shard_cache.shape[1] == page_size * 132
    assert scratch.is_contiguous() and shard_cache.is_contiguous()
    assert key.numel() // 128 == scratch_loc.numel() == logical_loc.numel()
    assert key.is_cuda
    assert all(
        tensor.device == key.device
        for tensor in (scratch, scratch_loc, shard_cache, logical_loc)
    )
    if logical_loc.numel() == 0:
        return
    key = key.reshape(-1, 128).contiguous()
    _jit_dsa_sharded_store_module(page_size).store(
        key,
        scratch,
        scratch_loc.contiguous(),
        shard_cache,
        logical_loc.contiguous(),
        shard_size,
        shard_rank,
    )
