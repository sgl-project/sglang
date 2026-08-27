"""
Fused metadata copy kernel for NSA backend CUDA graph replay.

This module provides JIT-compiled CUDA kernels for fusing multiple tensor
copy operations into single kernel launches, reducing kernel launch overhead
and improving CUDA graph replay performance.

The kernels are compiled on-demand using TVM FFI and cached for subsequent use.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

logger = logging.getLogger(__name__)


# ============================================================================
# JIT Module Compilation
# ============================================================================


@cache_once
def _jit_fused_metadata_copy_module(
    forward_mode: int, has_real_page_table: bool, has_flashmla: bool
):
    """Compile JIT module for single-backend fused metadata copy.

    Args:
        forward_mode: 0=DECODE, 1=TARGET_VERIFY, 2=DRAFT_EXTEND
        has_real_page_table: Whether real_page_table tensors are used
        has_flashmla: Whether FlashMLA metadata tensors are used
    """
    args = make_cpp_args(forward_mode, has_real_page_table, has_flashmla)
    try:
        return load_jit(
            "fused_metadata_copy",
            *args,
            cuda_files=["elementwise/fused_metadata_copy.cuh"],
            cuda_wrappers=[
                (
                    "fused_metadata_copy",
                    f"FusedMetadataCopyKernel<{args}>::run",
                )
            ],
        )
    except Exception as e:
        logger.error(
            f"Failed to compile JIT fused metadata copy kernel "
            f"(forward_mode={forward_mode}, has_real_page_table={has_real_page_table}, "
            f"has_flashmla={has_flashmla}): {e}"
        )
        raise


@cache_once
def _jit_fused_metadata_copy_multi_module(
    has_real_page_table: bool, has_flashmla: bool
):
    """Compile JIT module for multi-backend fused metadata copy (DECODE mode only).

    Args:
        has_real_page_table: Whether real_page_table tensors are used
        has_flashmla: Whether FlashMLA metadata tensors are used
    """
    args = make_cpp_args(has_real_page_table, has_flashmla)
    try:
        return load_jit(
            "fused_metadata_copy_multi",
            *args,
            cuda_files=["elementwise/fused_metadata_copy.cuh"],
            cuda_wrappers=[
                (
                    "fused_metadata_copy_multi",
                    f"FusedMetadataCopyMultiKernel<{args}>::run",
                )
            ],
        )
    except Exception as e:
        logger.error(
            f"Failed to compile JIT fused metadata copy multi kernel "
            f"(has_real_page_table={has_real_page_table}, has_flashmla={has_flashmla}): {e}"
        )
        raise


# ============================================================================
# Public API
# ============================================================================


def fused_metadata_copy_cuda(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    nsa_cache_seqlens_src: torch.Tensor,
    seqlens_expanded_src: Optional[torch.Tensor],
    nsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: Optional[torch.Tensor],
    flashmla_num_splits_src: Optional[torch.Tensor],
    flashmla_metadata_src: Optional[torch.Tensor],
    cache_seqlens_dst: torch.Tensor,
    cu_seqlens_k_dst: torch.Tensor,
    page_table_1_dst: torch.Tensor,
    nsa_cache_seqlens_dst: torch.Tensor,
    seqlens_expanded_dst: Optional[torch.Tensor],
    nsa_cu_seqlens_k_dst: torch.Tensor,
    real_page_table_dst: Optional[torch.Tensor],
    flashmla_num_splits_dst: Optional[torch.Tensor],
    flashmla_metadata_dst: Optional[torch.Tensor],
    forward_mode: int,
    bs: int,
    max_len: int,
    max_seqlen_k: int,
    seqlens_expanded_size: int,
) -> None:
    """
    Fused metadata copy kernel for NSA backend CUDA graph replay.

    This function fuses multiple tensor copy operations into a single kernel launch,
    reducing kernel launch overhead and improving performance.

    Args:
        cache_seqlens_src: Source cache sequence lengths [bs]
        cu_seqlens_k_src: Source cumulative sequence lengths [bs+1]
        page_indices_src: Source page indices [rows, max_len]
        nsa_cache_seqlens_src: Source NSA cache sequence lengths [size]
        seqlens_expanded_src: Optional source expanded sequence lengths [size] (required for TARGET_VERIFY/DRAFT_EXTEND)
        nsa_cu_seqlens_k_src: Source NSA cumulative sequence lengths [size+1]
        real_page_table_src: Optional source real page table [rows, cols]
        flashmla_num_splits_src: Optional source FlashMLA num_splits [size+1]
        flashmla_metadata_src: Optional source FlashMLA metadata tensor
        cache_seqlens_dst: Destination cache sequence lengths [bs]
        cu_seqlens_k_dst: Destination cumulative sequence lengths [bs+1]
        page_table_1_dst: Destination page table [rows, stride]
        nsa_cache_seqlens_dst: Destination NSA cache sequence lengths [size]
        seqlens_expanded_dst: Optional destination expanded sequence lengths [size] (required for TARGET_VERIFY/DRAFT_EXTEND)
        nsa_cu_seqlens_k_dst: Destination NSA cumulative sequence lengths [size+1]
        real_page_table_dst: Optional destination real page table [rows, cols]
        flashmla_num_splits_dst: Optional destination FlashMLA num_splits [size+1]
        flashmla_metadata_dst: Optional destination FlashMLA metadata tensor
        forward_mode: Forward mode (0=DECODE, 1=TARGET_VERIFY, 2=DRAFT_EXTEND)
        bs: Batch size
        max_len: Maximum length for decode/draft_extend mode
        max_seqlen_k: Maximum sequence length for target_verify mode
        seqlens_expanded_size: Size of expanded sequence lengths
    """
    # Determine template parameters for kernel specialization
    has_real_page_table = real_page_table_src is not None
    has_flashmla = flashmla_num_splits_src is not None

    # Get JIT-compiled module for this configuration (cached after first use)
    module = _jit_fused_metadata_copy_module(
        forward_mode, has_real_page_table, has_flashmla
    )

    # Ensure all required source tensors are contiguous (required for kernel's linear indexing)
    # This matches the CHECK_INPUT checks in the verified sgl-kernel implementation
    cache_seqlens_src = cache_seqlens_src.contiguous()
    cu_seqlens_k_src = cu_seqlens_k_src.contiguous()
    page_indices_src = page_indices_src.contiguous()
    nsa_cache_seqlens_src = nsa_cache_seqlens_src.contiguous()
    if seqlens_expanded_src is not None:
        seqlens_expanded_src = seqlens_expanded_src.contiguous()
    nsa_cu_seqlens_k_src = nsa_cu_seqlens_k_src.contiguous()

    # Call JIT-compiled kernel (None values are passed as Optional with no value)
    module.fused_metadata_copy(
        cache_seqlens_src,
        cu_seqlens_k_src,
        page_indices_src,
        nsa_cache_seqlens_src,
        seqlens_expanded_src,
        nsa_cu_seqlens_k_src,
        real_page_table_src,
        flashmla_num_splits_src,
        flashmla_metadata_src,
        cache_seqlens_dst,
        cu_seqlens_k_dst,
        page_table_1_dst,
        nsa_cache_seqlens_dst,
        seqlens_expanded_dst,
        nsa_cu_seqlens_k_dst,
        real_page_table_dst,
        flashmla_num_splits_dst,
        flashmla_metadata_dst,
        bs,
        max_len,
        max_seqlen_k,
        seqlens_expanded_size,
    )


def fused_metadata_copy_multi_cuda(
    cache_seqlens_src: torch.Tensor,
    cu_seqlens_k_src: torch.Tensor,
    page_indices_src: torch.Tensor,
    nsa_cache_seqlens_src: torch.Tensor,
    nsa_cu_seqlens_k_src: torch.Tensor,
    real_page_table_src: Optional[torch.Tensor],
    flashmla_num_splits_src: Optional[torch.Tensor],
    flashmla_metadata_src: Optional[torch.Tensor],
    cache_seqlens_dst0: torch.Tensor,
    cu_seqlens_k_dst0: torch.Tensor,
    page_table_1_dst0: torch.Tensor,
    nsa_cache_seqlens_dst0: torch.Tensor,
    nsa_cu_seqlens_k_dst0: torch.Tensor,
    real_page_table_dst0: Optional[torch.Tensor],
    flashmla_num_splits_dst0: Optional[torch.Tensor],
    flashmla_metadata_dst0: Optional[torch.Tensor],
    cache_seqlens_dst1: torch.Tensor,
    cu_seqlens_k_dst1: torch.Tensor,
    page_table_1_dst1: torch.Tensor,
    nsa_cache_seqlens_dst1: torch.Tensor,
    nsa_cu_seqlens_k_dst1: torch.Tensor,
    real_page_table_dst1: Optional[torch.Tensor],
    flashmla_num_splits_dst1: Optional[torch.Tensor],
    flashmla_metadata_dst1: Optional[torch.Tensor],
    cache_seqlens_dst2: torch.Tensor,
    cu_seqlens_k_dst2: torch.Tensor,
    page_table_1_dst2: torch.Tensor,
    nsa_cache_seqlens_dst2: torch.Tensor,
    nsa_cu_seqlens_k_dst2: torch.Tensor,
    real_page_table_dst2: Optional[torch.Tensor],
    flashmla_num_splits_dst2: Optional[torch.Tensor],
    flashmla_metadata_dst2: Optional[torch.Tensor],
    bs: int,
    max_len: int,
    seqlens_expanded_size: int,
) -> None:
    """
    Multi-backend fused metadata copy kernel for NSA backend CUDA graph replay.

    This function copies metadata from one source to THREE destinations in a single
    kernel launch, eliminating the overhead of 3 separate kernel calls. Currently
    only supports DECODE mode, which is the most common case.

    Args:
        cache_seqlens_src: Source cache sequence lengths [bs]
        cu_seqlens_k_src: Source cumulative sequence lengths [bs+1]
        page_indices_src: Source page indices [bs, max_len]
        nsa_cache_seqlens_src: Source NSA cache sequence lengths [bs]
        nsa_cu_seqlens_k_src: Source NSA cumulative sequence lengths [bs+1]
        real_page_table_src: Optional source real page table [bs, cols]
        flashmla_num_splits_src: Optional source FlashMLA num_splits [bs+1]
        flashmla_metadata_src: Optional source FlashMLA metadata tensor
        cache_seqlens_dst0-2: Destination cache sequence lengths for backends 0-2
        cu_seqlens_k_dst0-2: Destination cumulative sequence lengths for backends 0-2
        page_table_1_dst0-2: Destination page tables for backends 0-2
        nsa_cache_seqlens_dst0-2: Destination NSA cache sequence lengths for backends 0-2
        nsa_cu_seqlens_k_dst0-2: Destination NSA cumulative sequence lengths for backends 0-2
        real_page_table_dst0-2: Optional destination real page tables for backends 0-2
        flashmla_num_splits_dst0-2: Optional destination FlashMLA num_splits for backends 0-2
        flashmla_metadata_dst0-2: Optional destination FlashMLA metadata tensors for backends 0-2
        bs: Batch size
        max_len: Maximum length for decode mode
        seqlens_expanded_size: Size of expanded sequence lengths
    """
    # Determine template parameters for kernel specialization
    has_real_page_table = real_page_table_src is not None
    has_flashmla = flashmla_num_splits_src is not None

    # Get JIT-compiled module for this configuration (cached after first use)
    module = _jit_fused_metadata_copy_multi_module(has_real_page_table, has_flashmla)

    # Ensure all source tensors are contiguous (required for kernel's linear indexing)
    # This matches the CHECK_INPUT checks in the verified sgl-kernel implementation
    cache_seqlens_src = cache_seqlens_src.contiguous()
    cu_seqlens_k_src = cu_seqlens_k_src.contiguous()
    page_indices_src = page_indices_src.contiguous()
    nsa_cache_seqlens_src = nsa_cache_seqlens_src.contiguous()
    nsa_cu_seqlens_k_src = nsa_cu_seqlens_k_src.contiguous()

    # Call JIT-compiled kernel (None values are passed as Optional with no value)
    module.fused_metadata_copy_multi(
        cache_seqlens_src,
        cu_seqlens_k_src,
        page_indices_src,
        nsa_cache_seqlens_src,
        nsa_cu_seqlens_k_src,
        real_page_table_src,
        flashmla_num_splits_src,
        flashmla_metadata_src,
        cache_seqlens_dst0,
        cu_seqlens_k_dst0,
        page_table_1_dst0,
        nsa_cache_seqlens_dst0,
        nsa_cu_seqlens_k_dst0,
        real_page_table_dst0,
        flashmla_num_splits_dst0,
        flashmla_metadata_dst0,
        cache_seqlens_dst1,
        cu_seqlens_k_dst1,
        page_table_1_dst1,
        nsa_cache_seqlens_dst1,
        nsa_cu_seqlens_k_dst1,
        real_page_table_dst1,
        flashmla_num_splits_dst1,
        flashmla_metadata_dst1,
        cache_seqlens_dst2,
        cu_seqlens_k_dst2,
        page_table_1_dst2,
        nsa_cache_seqlens_dst2,
        nsa_cu_seqlens_k_dst2,
        real_page_table_dst2,
        flashmla_num_splits_dst2,
        flashmla_metadata_dst2,
        bs,
        max_len,
        seqlens_expanded_size,
    )
