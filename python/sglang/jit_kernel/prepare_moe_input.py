from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_prepare_moe_input_module() -> Module:
    return load_jit(
        "prepare_moe_input",
        cuda_files=["moe/prepare_moe_input.cuh"],
        cuda_wrappers=[
            ("prepare_moe_input", "prepare_moe_input"),
            ("shuffle_rows", "shuffle_rows"),
            ("apply_shuffle_mul_sum", "apply_shuffle_mul_sum"),
        ],
    )


@register_custom_op(
    op_name="prepare_moe_input_out",
    mutates_args=[
        "expert_offsets",
        "problem_sizes1",
        "problem_sizes2",
        "input_permutation",
        "output_permutation",
    ],
)
def prepare_moe_input_out(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: Optional[torch.Tensor],
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    input_permutation: torch.Tensor,
    output_permutation: torch.Tensor,
    num_experts: int,
    n: int,
    k: int,
) -> None:
    """
    Compute MoE routing metadata and sorted token permutations.

    Args:
        topk_ids:            [num_tokens, topk] int32 — expert indices per token
        expert_offsets:      [num_experts + 1] int32 — output cumulative token offsets
        blockscale_offsets:  [num_experts + 1] int32 (optional) — rounded offsets for FP8 block scales
        problem_sizes1:      [num_experts, 3] int32 — GEMM problem sizes (gate/up projection)
        problem_sizes2:      [num_experts, 3] int32 — GEMM problem sizes (down projection)
        input_permutation:   [num_tokens * topk] int32 — token index per expert slot
        output_permutation:  [num_tokens * topk] int32 — expert slot index per token
        num_experts:         total number of experts
        n:                   hidden size / 2 (intermediate size per partition)
        k:                   input hidden size
    """
    atomic_buffer = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    module = _jit_prepare_moe_input_module()
    module.prepare_moe_input(
        topk_ids,
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        atomic_buffer,
        num_experts,
        n,
        k,
    )


def prepare_moe_input(
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
    """
    Compute MoE routing metadata and sorted token permutations.

    Matches the call signature of ``sgl_kernel.prepare_moe_input``.

    Args:
        topk_ids:            [num_tokens, topk] int32
        expert_offsets:      [num_experts + 1] int32 — written in-place
        problem_sizes1:      [num_experts, 3] int32 — written in-place
        problem_sizes2:      [num_experts, 3] int32 — written in-place
        input_permutation:   [num_tokens * topk] int32 — written in-place
        output_permutation:  [num_tokens * topk] int32 — written in-place
        num_experts:         total number of experts
        n:                   intermediate size (half of gate/up projection output)
        k:                   input hidden size
        blockscale_offsets:  optional [num_experts + 1] int32 for FP8 block-scale offsets
    """
    prepare_moe_input_out(
        topk_ids,
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        n,
        k,
    )


@register_custom_op(
    op_name="shuffle_rows_out",
    mutates_args=["output"],
)
def shuffle_rows_out(
    input: torch.Tensor,
    dst2src_map: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """
    Shuffle rows of ``input`` according to ``dst2src_map``, writing to ``output``.

    Args:
        input:       [num_src_rows, num_cols]
        dst2src_map: [num_dst_rows] int32 — dst_row[i] = input[dst2src_map[i]]
        output:      [num_dst_rows, num_cols] — pre-allocated output
    """
    module = _jit_prepare_moe_input_module()
    module.shuffle_rows(input, dst2src_map, output)


def shuffle_rows(
    input: torch.Tensor,
    dst2src_map: torch.Tensor,
    output_shape: tuple,
) -> torch.Tensor:
    """
    Shuffle rows of ``input`` according to ``dst2src_map``.

    Matches the call signature of ``sgl_kernel.shuffle_rows``.

    Args:
        input:         [num_src_rows, num_cols]
        dst2src_map:   [num_dst_rows] int32
        output_shape:  (num_dst_rows, num_cols) — shape of the returned tensor

    Returns:
        output tensor [num_dst_rows, num_cols], same dtype as input
    """
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    shuffle_rows_out(input, dst2src_map, output)
    return output


@register_custom_op(
    op_name="apply_shuffle_mul_sum_out",
    mutates_args=["output"],
)
def apply_shuffle_mul_sum_out(
    input: torch.Tensor,
    output: torch.Tensor,
    permutation: torch.Tensor,
    factors: Optional[torch.Tensor],
) -> None:
    """
    Gather rows of ``input`` by ``permutation``, scale by ``factors``, and sum over topk.

    Equivalent to:
        (input[permutation].view(m, topk, k) * factors.view(m, topk, 1)).sum(dim=1)

    Args:
        input:       [m * topk, k] fp32/fp16/bf16
        output:      [m, k] — written in-place
        permutation: [m * topk] int32 — maps output positions to input rows
        factors:     [m * topk] optional scaling factors (topk weights)
    """
    module = _jit_prepare_moe_input_module()
    module.apply_shuffle_mul_sum(input, output, permutation, factors)


def apply_shuffle_mul_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    permutation: torch.Tensor,
    factors: Optional[torch.Tensor] = None,
) -> None:
    """
    Gather-scale-reduce: matches the call signature of ``sgl_kernel.apply_shuffle_mul_sum``.

    Args:
        input:       [m * topk, k] fp32/fp16/bf16
        output:      [m, k] — written in-place
        permutation: [m * topk] int32
        factors:     optional [m * topk] scaling factors
    """
    apply_shuffle_mul_sum_out(input, output, permutation, factors)
