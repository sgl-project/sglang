from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class CutlassMxfp4MoeQuantInfo(MoeQuantInfo):
    """Checkpoint-layout MXFP4 weights plus the grouped GEMM's per-expert metadata.

    The scales are not the checkpoint's ``[E, N, K/32]`` bytes: the collective TMA-loads them
    as ``Array<uint8_t, 4>``, so ``process_weights_after_loading`` transposes them to
    ``[E, K/128, N*4]`` and bakes the +126 exponent bias in.
    """

    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor
    expert_offsets: torch.Tensor
    problem_sizes1: torch.Tensor
    problem_sizes2: torch.Tensor


@register_fused_func("none", "cutlass_mxfp4")
def fused_experts_none_to_cutlass_mxfp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: CutlassMxfp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.cutlass_mxfp4_moe import cutlass_mxfp4_moe
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    # The second GEMM's input comes straight from silu_and_mul over a [gate; up] buffer, so
    # the clamped-swiglu family (gemm1_alpha / limits, GPT-OSS) has no path here.
    if (
        runner_config.activation != "silu"
        or not runner_config.is_gated
        or runner_config.gemm1_alpha is not None
        or runner_config.apply_router_weight_on_input
    ):
        raise NotImplementedError(
            "moe_runner_backend=cutlass_mxfp4 only implements gated silu without "
            f"router-weight-on-input, got activation={runner_config.activation!r}, "
            f"is_gated={runner_config.is_gated}, gemm1_alpha={runner_config.gemm1_alpha}, "
            f"apply_router_weight_on_input={runner_config.apply_router_weight_on_input}."
        )

    topk_output = dispatch_output.topk_output
    output = cutlass_mxfp4_moe(
        hidden_states=dispatch_output.hidden_states,
        w13_weight=quant_info.w13_weight,
        w13_weight_scale=quant_info.w13_weight_scale,
        w2_weight=quant_info.w2_weight,
        w2_weight_scale=quant_info.w2_weight_scale,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        expert_offsets=quant_info.expert_offsets,
        problem_sizes1=quant_info.problem_sizes1,
        problem_sizes2=quant_info.problem_sizes2,
        routed_scaling_factor=runner_config.routed_scaling_factor or 1.0,
    )
    return StandardCombineInput(hidden_states=output)
