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

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def _jit_nsa_fused_store_module(
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
        cuda_files=["nsa/fused_store_index_cache.cuh"],
        cuda_wrappers=[
            (
                "fused_store_index_k_cache",
                # - Float  = bf16_t (sgl_kernel/type.cuh)
                # - IndicesT = int64_t (out_cache_loc is int64 in SGLang SetKAndS)
                # - kPageSize = 64 (CUDA NSA)
                f"FusedStoreCacheIndexerKernel<{args}>::run",
            )
        ],
    )


@cache_once
def can_use_nsa_fused_store(
    key_dtype: torch.dtype, indices_dtype: torch.dtype, page_size: int
) -> bool:
    logger = logging.getLogger(__name__)
    try:
        _jit_nsa_fused_store_module(key_dtype, indices_dtype, page_size)
        return True
    except Exception as e:
        logger.warning(f"Failed to load nsa fused store JIT kernel: {e}")
        return False


def _fused_store_index_k_cache_fake_impl(
    key: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_num_tokens: torch.Tensor,
    page_size: int = 64,
) -> None:
    return None


@register_custom_op(
    mutates_args=["index_k_with_scale"],
    fake_impl=_fused_store_index_k_cache_fake_impl,
)
@debug_kernel_api
def fused_store_index_k_cache(
    key: torch.Tensor,
    index_k_with_scale: torch.Tensor,
    out_cache_loc: torch.Tensor,
    extend_num_tokens: torch.Tensor,
    page_size: int = 64,
) -> None:
    """
    Fused: quantize bf16 key (N,128) -> fp8 + fp32 scale and write into NSATokenToKVPool.index_k_with_scale_buffer.

    key:                 (num_tokens, 128) bf16, contiguous
    index_k_with_scale:  (num_pages, 64*(128+4)) uint8, contiguous
    out_cache_loc:       (num_tokens,) int64, contiguous; token indices in TokenToKVPool
    extend_num_tokens:   (1,) int32, contiguous; per-replay real token count.
                         Under piecewise CUDA graph, num_tokens (= key.shape[0]) is the
                         bucket size baked at capture, while extend_num_tokens is read
                         dynamically and bounds the actual writes to skip the padding
                         region. For non-PCG callers, set this to a 1-elem tensor whose
                         value equals num_tokens.
    """
    assert key.is_cuda and key.is_contiguous(), "key must be contiguous CUDA tensor"
    assert (
        index_k_with_scale.is_cuda and index_k_with_scale.is_contiguous()
    ), "index_k_with_scale must be contiguous CUDA tensor"
    assert (
        out_cache_loc.is_cuda and out_cache_loc.is_contiguous()
    ), "out_cache_loc must be contiguous CUDA tensor"
    assert (
        extend_num_tokens.is_cuda and extend_num_tokens.is_contiguous()
    ), "extend_num_tokens must be contiguous CUDA tensor"

    assert (
        key.dim() == 2 and key.shape[1] == 128
    ), f"expected key shape (N,128), got {key.shape}"
    assert key.dtype == torch.bfloat16, f"{key.dtype=}"
    assert index_k_with_scale.dtype == torch.uint8, f"{index_k_with_scale.dtype=}"
    assert out_cache_loc.dtype == torch.int64, f"{out_cache_loc.dtype=}"
    assert (
        extend_num_tokens.dtype == torch.int32 and extend_num_tokens.numel() == 1
    ), f"extend_num_tokens must be a 1-elem int32 tensor, got dtype={extend_num_tokens.dtype} numel={extend_num_tokens.numel()}"

    module = _jit_nsa_fused_store_module(key.dtype, out_cache_loc.dtype, page_size)
    module.fused_store_index_k_cache(
        key, index_k_with_scale, out_cache_loc, extend_num_tokens
    )
