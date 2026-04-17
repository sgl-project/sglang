"""Fused MoE Align Block Size + Count-and-Sort JIT kernel.

Replaces two separate sgl-kernel C++ kernels:
  1. moe_align_block_size_kernel (prefix sum + expert_ids + fill)
  2. count_and_sort_expert_tokens_kernel (atomic scatter)

with a single JIT kernel. Supports up to 256 experts.
No sgl-kernel rebuild required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import triton

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fused_moe_align_sort_module(id_dtype: str) -> Module:
    """Compile JIT module for given index dtype (int32 or int64)."""
    cpp_type = "int32_t" if id_dtype == "int32" else "int64_t"
    return load_jit(
        "fused_moe_align_sort",
        cpp_type,
        cuda_files=["moe/fused_moe_align_sort.cuh"],
        cuda_wrappers=[
            ("fused_moe_align_sort", f"fused_moe_align_sort<{cpp_type}>"),
        ],
    )


def get_fused_moe_align_sort_output_shapes(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[int, int]:
    numel = topk_ids.numel()
    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    return max_num_tokens_padded, max_num_m_blocks


def fused_moe_align_sort_into(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Write fused moe align+sort outputs into caller-provided buffers."""
    assert topk_ids.is_cuda
    assert (
        num_experts <= 256
    ), f"JIT fused kernel supports up to 256 experts, got {num_experts}"

    max_num_tokens_padded, max_num_m_blocks = get_fused_moe_align_sort_output_shapes(
        topk_ids, block_size, num_experts
    )
    assert sorted_ids.is_cuda and sorted_ids.dtype == torch.int32
    assert expert_ids.is_cuda and expert_ids.dtype == torch.int32
    assert num_tokens_post_pad.is_cuda and num_tokens_post_pad.dtype == torch.int32
    assert sorted_ids.numel() >= max_num_tokens_padded
    assert expert_ids.numel() >= max_num_m_blocks
    assert num_tokens_post_pad.numel() >= 1

    id_dtype = "int32" if topk_ids.dtype == torch.int32 else "int64"
    module = _jit_fused_moe_align_sort_module(id_dtype)
    module.fused_moe_align_sort(
        topk_ids.flatten(),
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        num_experts,
        block_size,
    )


def fused_moe_align_sort(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused MoE align block size + token sorting.

    Replaces moe_align_block_size() from sgl_kernel with a JIT version
    that fuses fill + align + sort into one kernel launch.

    Parameters
    ----------
    topk_ids   : [total_tokens, top_k] int32/int64 — expert indices per token
    block_size : MoE block size for padding
    num_experts : total number of experts (including EP offset expert -1)

    Returns
    -------
    (sorted_ids, expert_ids, num_tokens_post_pad) — same as moe_align_block_size
    """
    assert topk_ids.is_cuda
    assert (
        num_experts <= 256
    ), f"JIT fused kernel supports up to 256 experts, got {num_experts}"

    max_num_tokens_padded, max_num_m_blocks = get_fused_moe_align_sort_output_shapes(
        topk_ids, block_size, num_experts
    )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=topk_ids.device)

    fused_moe_align_sort_into(
        topk_ids,
        block_size,
        num_experts,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad
