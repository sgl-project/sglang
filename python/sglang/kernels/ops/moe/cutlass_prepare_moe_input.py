"""Complete routing maps and grouped-GEMM shapes for CUTLASS MoE."""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_cutlass_prepare_moe_input_module() -> Module:
    return load_jit(
        "cutlass_prepare_moe_input",
        cuda_files=["moe/cutlass_prepare_moe_input.cuh"],
        cuda_wrappers=[("prepare", "cutlass_prepare_moe_input")],
    )


@register_custom_op(
    op_name="cutlass_prepare_moe_input",
    mutates_args=[
        "expert_offsets",
        "problem_sizes1",
        "problem_sizes2",
        "input_permutation",
        "output_permutation",
        "blockscale_offsets",
    ],
)
def cutlass_prepare_moe_input(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
    blockscale_offsets: Optional[torch.Tensor] = None,
) -> None:
    """Write contiguous int32 CUTLASS metadata, including every unused map entry.

    Routes outside [0, num_experts) have output permutation -1. Input-map
    capacity beyond expert_offsets[-1] points at token zero for full gathers.
    Optional blockscale offsets count per-expert rows rounded up to 128.
    """
    cursors = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
    _jit_cutlass_prepare_moe_input_module().prepare(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        cursors,
        num_experts,
        n,
        k,
        blockscale_offsets,
    )
