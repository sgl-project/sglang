from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_permute_prepare_module() -> Module:
    return load_jit(
        "moe_permute_prepare",
        cuda_files=["moe/moe_permute_prepare.cu"],
        header_only=False,
    )


@register_custom_op(
    op_name="moe_permute_prepare_out",
    mutates_args=["expert_offsets", "src2dst"],
)
def _moe_permute_prepare_out(
    sorted_topk_ids: torch.Tensor,
    reorder_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    src2dst: torch.Tensor,
    num_experts: int,
    use_int64_offset: bool,
    is_ep: bool,
) -> None:
    module = _jit_moe_permute_prepare_module()
    module.moe_permute_prepare(
        sorted_topk_ids,
        reorder_ids,
        expert_offsets,
        src2dst,
        num_experts,
        use_int64_offset,
        is_ep,
    )


def moe_permute_prepare(
    topk_ids: torch.Tensor,
    num_experts: int,
    use_int64_offset: bool = False,
    is_ep: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if topk_ids.dtype != torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if not topk_ids.is_cuda:
        raise ValueError("topk_ids must be a CUDA tensor")

    sorted_topk_ids, reorder_ids = torch.sort(topk_ids.flatten())
    offset_dtype = torch.int64 if use_int64_offset else torch.int32
    expert_offsets = torch.empty(
        (num_experts + 1,), dtype=offset_dtype, device=topk_ids.device
    )
    src2dst = torch.empty(
        (topk_ids.numel(),), dtype=torch.int32, device=topk_ids.device
    )

    _moe_permute_prepare_out(
        sorted_topk_ids,
        reorder_ids,
        expert_offsets,
        src2dst,
        num_experts,
        use_int64_offset,
        is_ep,
    )
    return expert_offsets, src2dst
