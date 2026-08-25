from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_get_block_table_module(
    topk: int, head_group_num: int, block_size: int
) -> Module:
    """Compile and cache the JIT module for a given sparse topk value.

    One module is built per topk value, replacing the original runtime
    ``VALUE_SPLITS_SWITCH(topk, ...)`` dispatch with a compile-time template
    argument ``kSparseTopK``.
    """
    args = make_cpp_args(topk, head_group_num, block_size)
    wrappers = [
        (
            "get_block_table_blockwise",
            f"minicpm_sala::get_block_table<false, {args}>",
        ),
    ]
    if block_size == 64 and topk % 16 == 0:
        wrappers.append(
            (
                "get_block_table_elementwise",
                f"minicpm_sala::get_block_table<true, {args}>",
            )
        )
    return load_jit(
        f"get_block_table_strategies_topk{topk}_g{head_group_num}_b{block_size}",
        *args,
        cuda_files=["minicpm_sala/get_block_table.cuh"],
        cuda_wrappers=wrappers,
    )


def get_block_table(
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    token_to_bs: torch.Tensor,
    token_pos_in_bs: torch.Tensor,
    seqlen_q: torch.Tensor,
    head_group_num: int = 2,
    block_size: int = 64,
    *,
    elementwise: bool,
) -> torch.Tensor:
    if topk_idx.dim() != 3:
        raise RuntimeError(
            f"topk_idx must be 3D [head_group, token_num, topk], got shape {tuple(topk_idx.shape)}"
        )
    token_num = topk_idx.shape[1]
    topk = topk_idx.shape[2]
    if topk <= 0 or block_size <= 0:
        raise RuntimeError(
            f"topk and block_size must be positive, got {topk=} and {block_size=}"
        )
    kernel_name = (
        "get_block_table_elementwise"
        if elementwise and block_size == 64 and topk % 16 == 0
        else "get_block_table_blockwise"
    )

    out = torch.empty(
        (token_num, head_group_num, topk * block_size),
        dtype=torch.int32,
        device=topk_idx.device,
    )
    module = _jit_get_block_table_module(topk, head_group_num, block_size)
    getattr(module, kernel_name)(
        out, topk_idx, block_table, token_to_bs, token_pos_in_bs, seqlen_q
    )
    return out
