from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

_FP4_SF_VEC_SIZE = 16


@dataclass
class CuteDslFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer CuteDSL FP4 MoE kernels."""

    # Lazily-created CuteDslMoEWrapper (stashed on layer)
    wrapper: Any

    # Weights (uint8 FP4 packed)
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor

    # Block-scale factors
    w13_weight_sf: torch.Tensor
    w2_weight_sf: torch.Tensor

    # Per-expert GEMM scales
    w1_alpha: torch.Tensor
    w2_alpha: torch.Tensor

    # Intermediate quantization scale (fc2 input)
    fc2_input_scale: torch.Tensor

    # Activation quantization scale (scalarized)
    input_scale: torch.Tensor


@register_fused_func("none", "flashinfer_cutedsl")
def fused_experts_none_to_flashinfer_cutedsl_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from flashinfer import fp4_quantize

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert runner_config.activation == "silu", "Only silu is supported for CuteDSL MoE."

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    x_fp4, x_sf = fp4_quantize(
        hidden_states,
        quant_info.input_scale,
        sf_vec_size=_FP4_SF_VEC_SIZE,
        is_sf_swizzled_layout=False,
    )

    output = quant_info.wrapper.run(
        x=x_fp4,
        x_sf=x_sf,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        w1_weight=quant_info.w13_weight,
        w1_weight_sf=quant_info.w13_weight_sf,
        w1_alpha=quant_info.w1_alpha,
        fc2_input_scale=quant_info.fc2_input_scale,
        w2_weight=quant_info.w2_weight,
        w2_weight_sf=quant_info.w2_weight_sf,
        w2_alpha=quant_info.w2_alpha,
    )

    return StandardCombineInput(hidden_states=output)
