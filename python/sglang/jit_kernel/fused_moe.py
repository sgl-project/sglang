from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import torch
from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@lru_cache(maxsize=None)
def _jit_merlin_module(has_zp: bool, is_zp_float: bool) -> Module:
    args = make_cpp_args(
        has_zp,
        is_zp_float,
    )
    return load_jit(
        "merlin",
        cuda_files=["marlin_moe_wna16/ops.cuh"],
        cuda_wrappers=[
            ("moe_wna16_marlin_gemm", f"marlin::moe_wna16_marlin_gemm<{args}>"),
        ],
    )


def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    perm_or_none: Optional[torch.Tensor],
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    is_zp_float: bool,
):
    c = (
        c_or_none
        if c_or_none is not None
        else torch.empty((size_m * top_k, size_n), device=a.device, dtype=a.dtype)
    )
    _jit_merlin_module(b_zeros_or_none is not None, is_zp_float).moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias_or_none,
        b_scales,
        global_scale_or_none,
        b_zeros_or_none,
        g_idx_or_none,
        perm_or_none,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
    )
    return c
