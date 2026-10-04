# SPDX-License-Identifier: Apache-2.0
"""SM90 MoE with MXFP4 weights and bf16 activations, on a CUTLASS grouped GEMM.

The mixed-input wgmma pipeline decodes E2M1 nibbles and folds the group-32 E8M0 scale into
the A fragment, so unlike the marlin path there is no activation quantize / dequantize glue
around either GEMM. Measured 1.50x marlin summed over both DSV4-Flash TP4 MoE GEMMs.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.ep_moe_kernels import (
    post_reorder_for_cutlass_moe,
    pre_reorder_for_cutlass_moe,
)
from sglang.kernels.ops.moe.mxfp4_a16_moe_mm import mxfp4_a16_moe_mm
from sglang.kernels.ops.moe.mxfp4_moe_grouped_metadata import (
    MXFP4_MOE_FUSED_METADATA_MAX_EXPERTS,
    mxfp4_moe_grouped_metadata,
)


def park_foreign_experts(topk_ids: torch.Tensor, *, num_local_experts: int) -> torch.Tensor:
    """Move the dispatcher's ``-1`` pad to the sentinel the reorder kernels test for.

    Under EP the standard dispatcher hands this rank local expert ids and pads the
    remaining top-k slots of a token with ``-1``. Both reorder kernels skip a slot by
    testing ``expert_id != num_local_experts``, so a raw ``-1`` is *not* skipped and the
    gather follows it to row -1 of the pooled buffer. Remapping to ``num_local_experts``
    is also what the metadata kernel expects: ids outside ``[0, num_experts)`` -- parked
    ids included -- leave ``src2dst`` at -1.
    """
    return torch.where(topk_ids == -1, num_local_experts, topk_ids).to(torch.int32)


def cutlass_mxfp4_moe(
    *,
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    routed_scaling_factor: float,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Runs one MXFP4 MoE layer: gather -> grouped GEMM -> SiLU -> grouped GEMM -> scatter.

    :param hidden_states: [num_tokens, hidden] bf16.
    :param w13_weight: [E, 2 * intermediate, hidden // 2] uint8 E2M1 pairs, checkpoint layout.
    :param w13_weight_scale: [E, hidden // 128, 2 * intermediate * 4] uint8 E8M0, +126 baked in.
    :param w2_weight: [E, hidden, intermediate // 2] uint8 E2M1 pairs.
    :param w2_weight_scale: [E, intermediate // 128, hidden * 4] uint8 E8M0, +126 baked in.
    :param expert_offsets: [E + 1] int32 scratch, filled here.
    :param problem_sizes1: [E, 3] int32 scratch, filled here.
    :param problem_sizes2: [E, 3] int32 scratch, filled here.
    :param routed_scaling_factor: applied once per token by the post-reorder kernel.
    :param swiglu_limit: clamps gate to <= L and up to [-L, L] before the SiLU, matching
        the reference MLP. DeepSeek-V4 sets it (10.0) and every other DSV4 MoE path
        forwards it, so dropping it here both diverges from the checkpoint's semantics
        and, on the rare row whose pre-activation exceeds the rail, injects a
        deterministic error into the layer output.
    """
    num_local_experts = w13_weight.size(0)
    num_tokens = hidden_states.size(0)
    hidden = w13_weight.size(2) * 2
    intermediate = w2_weight.size(2) * 2
    topk = topk_ids.size(1)
    device = hidden_states.device
    rows = num_tokens * topk

    topk_ids = park_foreign_experts(topk_ids, num_local_experts=num_local_experts)
    if num_local_experts > MXFP4_MOE_FUSED_METADATA_MAX_EXPERTS:
        raise NotImplementedError(
            "moe_runner_backend=cutlass_mxfp4 scans the experts in one CTA, so it caps at "
            f"{MXFP4_MOE_FUSED_METADATA_MAX_EXPERTS} experts per rank, got "
            f"{num_local_experts}."
        )
    src2dst = torch.empty(rows, dtype=torch.int32, device=device)
    mxfp4_moe_grouped_metadata(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        src2dst,
        intermediate,
        hidden,
    )

    # The expert-sorted activations and the second GEMM's output are both [rows, hidden], and
    # the former is dead once GEMM1 has read it, so one buffer serves both: the reuse is
    # ordered by the stream, and the peak stays one [rows, hidden] block below two.
    pool = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)
    # a1_scales=None makes this a pure gather -- the CUTLASS mainloop takes bf16 activations.
    pre_reorder_for_cutlass_moe(
        hidden_states,
        pool,
        src2dst,
        topk_ids,
        None,
        num_local_experts,
        topk,
        num_tokens,
        hidden,
    )

    gateup = torch.empty((rows, 2 * intermediate), device=device, dtype=torch.bfloat16)
    mxfp4_a16_moe_mm(
        gateup,
        pool,
        w13_weight,
        w13_weight_scale,
        expert_offsets[:-1],
        problem_sizes1,
    )

    down_input = torch.empty((rows, intermediate), device=device, dtype=torch.bfloat16)
    if swiglu_limit is None:
        silu_and_mul(gateup, down_input)
    else:
        from sglang.kernels.ops.attention.dsv4 import silu_and_mul_clamp

        silu_and_mul_clamp(gateup, down_input, swiglu_limit)

    down_output = pool
    mxfp4_a16_moe_mm(
        down_output,
        down_input,
        w2_weight,
        w2_weight_scale,
        expert_offsets[:-1],
        problem_sizes2,
    )

    output = torch.empty_like(hidden_states)
    post_reorder_for_cutlass_moe(
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        num_local_experts,
        topk,
        num_tokens,
        hidden,
        routed_scaling_factor,
    )
    return output
