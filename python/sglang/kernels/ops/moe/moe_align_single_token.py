"""CUDA JIT single-warp moe_align_block_size for M == 1 decode batches."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_align_single_token_module(write_inactive_prefix_count: bool = False) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), write_inactive_prefix_count)
    return load_jit(
        "moe_align_single_token",
        *args,
        cuda_files=["moe/align_single_token.cuh"],
        cuda_wrappers=[("run", f"AlignSingleTokenKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def moe_align_single_token(
    topk_ids: torch.Tensor,
    block_size: int,
    *,
    return_inactive_prefix_count: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """moe_align_block_size for a single token (distinct expert ids).

    Returns (sorted_token_ids, expert_ids, num_tokens_post_padded) with the
    same layout as moe_align_block_size: experts ascending, one block per
    expert, padding value = numel. With ``return_inactive_prefix_count``,
    the count is int32 ``[padded_tokens, inactive_blocks]``.
    """
    topk = topk_ids.shape[1]
    device = topk_ids.device
    sorted_ids = torch.empty((topk * block_size,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((topk,), dtype=torch.int32, device=device)
    num_post = torch.empty(
        (2 if return_inactive_prefix_count else 1,), dtype=torch.int32, device=device
    )
    _jit_align_single_token_module(return_inactive_prefix_count).run(
        topk_ids, sorted_ids, expert_ids, num_post, block_size
    )
    return sorted_ids, expert_ids, num_post
