"""Complete routing maps and grouped-GEMM shapes for CUTLASS MoE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_cutlass_prepare_moe_input_module(
    transpose_problem_sizes: bool = False,
) -> Module:
    args = make_cpp_args(transpose_problem_sizes)
    return load_jit(
        "cutlass_prepare_moe_input",
        *args,
        cuda_files=["moe/cutlass_prepare_moe_input.cuh"],
        cuda_wrappers=[("prepare", f"cutlass_prepare_moe_input<{args}>")],
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
    *,
    transpose_problem_sizes: bool = False,
) -> None:
    """Write contiguous int32 CUTLASS metadata, including every unused map entry.

    Routes outside [0, num_experts) have output permutation -1. Input-map
    capacity beyond expert_offsets[-1] points at token zero for full gathers.
    Optional blockscale offsets count per-expert rows rounded up to 128.
    Transposed problem sizes put the token count in N instead of M.
    """
    _jit_cutlass_prepare_moe_input_module(transpose_problem_sizes).prepare(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        n,
        k,
        blockscale_offsets,
    )
