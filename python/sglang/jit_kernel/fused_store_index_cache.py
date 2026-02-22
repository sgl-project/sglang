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

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def _jit_nsa_fused_store_module() -> Module:
    """
    Build a JIT module that exposes:
      module.fused_store_index_k_cache(input_bf16, index_k_with_scale_u8, loc_i64)
    """
    return load_jit(
        "fused_store_index_k_cache",
        cuda_files=["nsa/fused_store_index_cache.cuh"],
        cuda_wrappers=[
            (
                "fused_store_index_k_cache",
                # - Float  = bf16_t (sgl_kernel/type.cuh)
                # - IndicesT = int64_t (out_cache_loc is int64 in SGLang SetKAndS)
                # - kPageSize = 64 (CUDA NSA)
                "FusedStoreCacheIndexerKernel<bf16_t, int64_t, 64>::run",
            )
        ],
    )


@cache_once
def can_use_nsa_fused_store() -> bool:
    """
    Similar spirit to can_use_store_cache(): compile once and cache result.
    """
    try:
        _jit_nsa_fused_store_module()
        return True
    except Exception as e:
        logger.warning(f"Failed to load nsa fused store JIT kernel: {e}")
        return False


def fused_store_index_k_cache(
    key_bf16: torch.Tensor,
    index_k_with_scale_u8: torch.Tensor,
    out_cache_loc_i64: torch.Tensor,
) -> None:
    """
    Fused: quantize bf16 key (N,128) -> fp8 + fp32 scale and write into NSATokenToKVPool.index_k_with_scale_buffer.

    key_bf16:            (num_tokens, 128) bf16 (or reshapeable to it)
    index_k_with_scale:  (num_pages, 64*(128+4)) uint8
    out_cache_loc:       (num_tokens,) int64 token indices in TokenToKVPool
    """
    assert key_bf16.is_cuda
    assert index_k_with_scale_u8.is_cuda
    assert out_cache_loc_i64.is_cuda

    # 1) normalize shapes
    if key_bf16.dim() != 2:
        key_bf16 = key_bf16.view(-1, key_bf16.shape[-1])
    assert key_bf16.shape[1] == 128, f"expected key last-dim=128, got {key_bf16.shape}"

    # 2) dtypes
    assert key_bf16.dtype == torch.bfloat16, f"{key_bf16.dtype=}"
    assert index_k_with_scale_u8.dtype == torch.uint8, f"{index_k_with_scale_u8.dtype=}"
    assert out_cache_loc_i64.dtype == torch.int64, f"{out_cache_loc_i64.dtype=}"

    # 3) contiguity
    if not key_bf16.is_contiguous():
        key_bf16 = key_bf16.contiguous()
    if not out_cache_loc_i64.is_contiguous():
        out_cache_loc_i64 = out_cache_loc_i64.contiguous()
    if not index_k_with_scale_u8.is_contiguous():
        index_k_with_scale_u8 = index_k_with_scale_u8.contiguous()

    module = _jit_nsa_fused_store_module()
    module.fused_store_index_k_cache(key_bf16, index_k_with_scale_u8, out_cache_loc_i64)
