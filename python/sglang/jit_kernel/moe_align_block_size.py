from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_align_block_size_module() -> Module:
    return load_jit(
        "moe_align_block_size",
        cuda_files=["moe/moe_align_block_size.cuh"],
        cuda_wrappers=[("moe_align_block_size", "moe_align_block_size")],
    )


@register_custom_op(
    op_name="jit_moe_align_block_size",
    mutates_args=["sorted_token_ids", "expert_ids", "num_tokens_post_pad", "cumsum"],
)
def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum: torch.Tensor,
    pad_sorted_token_ids: bool,
) -> None:
    module = _jit_moe_align_block_size_module()
    module.moe_align_block_size(
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum,
        num_experts,
        block_size,
        pad_sorted_token_ids,
        sorted_token_ids.shape[0],
    )
