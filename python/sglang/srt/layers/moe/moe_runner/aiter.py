from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.utils import get_int_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.base import (
        CombineInput,
        DispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


class AiterQuantType(str, Enum):
    NONE = "No"
    PER_TOKEN = "per_Token"
    PER_128X128 = "per_128x128"
    PER_1X32 = "per_1x32"


@dataclass
class AiterMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_type: AiterQuantType = AiterQuantType.NONE
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    expert_mask: Optional[torch.Tensor] = None
    doweight_stage1: bool = False
    hidden_pad: int = 0
    intermediate_pad: int = 0


_AITER_ACTIVATIONS = {"silu": "Silu", "swiglu": "Swiglu"}


def get_aiter_expert_mask(layer) -> Optional[torch.Tensor]:
    # DeepEPMoE / MoriEPMoE set self.expert_mask in __init__; standard FusedMoE
    # exposes it via the dispatcher.
    if getattr(layer, "expert_mask", None) is not None:
        return layer.expert_mask
    return getattr(layer.dispatcher, "expert_mask_gpu", None)


def _aiter_activation(activation: str):
    from aiter import ActivationType

    return getattr(ActivationType, _AITER_ACTIVATIONS.get(activation, "Gelu"))


def _aiter_quant_type(quant_type: AiterQuantType):
    from aiter import QuantType

    return getattr(QuantType, quant_type.value)


@register_fused_func("none", "aiter")
def fused_experts_none_to_aiter(
    dispatch_output: StandardDispatchOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from aiter.fused_moe import fused_moe

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert not runner_config.no_combine, "no_combine=True is not supported by AITER"

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(torch.float32)

    if runner_config.apply_router_weight_on_input and not quant_info.doweight_stage1:
        # Pre-scale at the Python level for kernels that don't honor doweight_stage1.
        assert (
            topk_weights.dim() == 2 and topk_weights.shape[-1] == 1
        ), "apply_router_weight_on_input requires topk=1"
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        topk_weights = torch.ones_like(topk_weights)

    output = fused_moe(
        hidden_states=hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_weight=topk_weights,
        topk_ids=topk_ids.to(torch.int32),
        quant_type=_aiter_quant_type(quant_info.quant_type),
        activation=_aiter_activation(runner_config.activation),
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        bias1=quant_info.b13,
        bias2=quant_info.b2,
        expert_mask=quant_info.expert_mask,
        doweight_stage1=quant_info.doweight_stage1,
        hidden_pad=quant_info.hidden_pad,
        intermediate_pad=quant_info.intermediate_pad,
    )
    return StandardCombineInput(hidden_states=output)


def _run_aiter_for_deepep(
    dispatch_output: "DeepEPNormalDispatchOutput | DeepEPLLDispatchOutput",
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    from aiter.fused_moe import fused_moe

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids
    topk_weights = dispatch_output.topk_weights

    if hidden_states.shape[0] == 0:
        return hidden_states

    # DeepEP marks invalid slots with idx == -1; AITER cannot accept negative ids,
    # so we reroute them to the sink slot at index num_local_experts (masked off
    # by quant_info.expert_mask, which has shape (num_local_experts + 1,)).
    topk_ids = topk_ids.to(torch.int32)
    topk_ids = torch.where(
        topk_ids == -1,
        torch.full_like(topk_ids, runner_config.num_local_experts),
        topk_ids,
    )

    return fused_moe(
        hidden_states=hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_weight=topk_weights.to(torch.float32),
        topk_ids=topk_ids,
        quant_type=_aiter_quant_type(quant_info.quant_type),
        activation=_aiter_activation(runner_config.activation),
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        bias1=quant_info.b13,
        bias2=quant_info.b2,
        expert_mask=quant_info.expert_mask,
        doweight_stage1=quant_info.doweight_stage1,
        hidden_pad=quant_info.hidden_pad,
        intermediate_pad=quant_info.intermediate_pad,
    )


def _build_deepep_combine_input(
    dispatch_output: "DispatchOutput",
    output: torch.Tensor,
) -> "CombineInput":
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )

    cls = (
        DeepEPNormalCombineInput
        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
        else DeepEPLLCombineInput
    )
    return cls(
        hidden_states=output,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )


@register_fused_func("deepep", "aiter")
@register_fused_func("mooncake", "aiter")
@register_fused_func("nixl", "aiter")
def fused_experts_deepep_to_aiter(
    dispatch_output: "DeepEPNormalDispatchOutput | DeepEPLLDispatchOutput",
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> "CombineInput":
    output = _run_aiter_for_deepep(dispatch_output, quant_info, runner_config)
    return _build_deepep_combine_input(dispatch_output, output)


@register_fused_func("mori", "aiter")
def fused_experts_mori_to_aiter(
    dispatch_output: "MoriEPNormalDispatchOutput | MoriEPLLDispatchOutput",
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> "CombineInput":
    from aiter.fused_moe import fused_moe

    from sglang.srt.layers.moe.rocm_moe_utils import upscale, upscale_mxfp4
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLCombineInput,
        MoriEPNormalCombineInput,
    )

    dispatch_a1 = dispatch_output.hidden_states
    dispatch_scale = dispatch_output.hidden_states_scale
    dispatch_ids = dispatch_output.topk_ids
    dispatch_weights = dispatch_output.topk_weights
    dispatch_recv_token_num = dispatch_output.num_recv_tokens_per_expert
    output_dtype = dispatch_output.out_dtype

    # Truncate dispatch tensors to reduce MoE computation on padding rows.
    # mori combine only reads [0, totalRecvTokenNum), so the truncated output
    # can be passed directly without padding back.
    mori_max = get_int_env_var("SGLANG_MORI_MOE_MAX_INPUT_TOKENS", 0)
    if mori_max > 0:
        dispatch_a1 = dispatch_a1[:mori_max]
        if dispatch_scale is not None:
            dispatch_scale = dispatch_scale[:mori_max]
        dispatch_ids = dispatch_ids[:mori_max]
        dispatch_weights = dispatch_weights[:mori_max]

    # Infer weight quant family from the quant method's choice.
    weight_quant = quant_info.quant_type
    is_fp8_quant = weight_quant in (
        AiterQuantType.PER_128X128,
        AiterQuantType.PER_TOKEN,
    )
    is_w4a4 = weight_quant == AiterQuantType.PER_1X32

    # Decide the effective activation quant_type based on dispatch dtype/scale,
    # upscaling the dispatched tensor when there is a kernel-unsupported
    # weight/activation dtype mismatch.
    quant_type = AiterQuantType.NONE
    if (
        not is_fp8_quant
        and dispatch_scale is not None
        and dispatch_a1.dtype != torch.float4_e2m1fn_x2
    ):
        if is_w4a4:
            # W4A4 weights with FP8 dispatch: dequant FP8->BF16 first; the FP4
            # per_1x32 path needs BF16 input.
            dispatch_a1 = upscale(
                dispatch_a1, dispatch_scale, dispatch_recv_token_num, output_dtype
            )
            dispatch_scale = None
        else:
            # BF16 weights with FP8 dispatch: pass FP8 through, no requantize.
            quant_type = AiterQuantType.PER_128X128

    if dispatch_a1.dtype == torch.float4_e2m1fn_x2 and dispatch_scale is not None:
        if is_fp8_quant:
            # FP8 weights + FP4 dispatch: no kernel for fp4x2/fp8 pair, dequant
            # FP4->BF16 first; fused_moe will re-quantize to FP8 internally.
            dispatch_a1 = upscale_mxfp4(
                dispatch_a1, dispatch_scale, dispatch_recv_token_num, output_dtype
            )
            dispatch_scale = None
        elif quant_type == AiterQuantType.NONE:
            # BF16 / W4A4 weights with FP4 dispatch: pass FP4 through.
            quant_type = AiterQuantType.PER_1X32

    if is_w4a4:
        quant_type = AiterQuantType.PER_1X32
    elif is_fp8_quant and quant_type == AiterQuantType.NONE:
        quant_type = weight_quant

    output = fused_moe(
        hidden_states=dispatch_a1,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        a1_scale=dispatch_scale,
        topk_weight=dispatch_weights.to(torch.float32),
        topk_ids=dispatch_ids.to(torch.int32),
        quant_type=_aiter_quant_type(quant_type),
        activation=_aiter_activation(runner_config.activation),
        expert_mask=quant_info.expert_mask,
        num_local_tokens=dispatch_recv_token_num,
        dtype=output_dtype,
    )

    cls = (
        MoriEPNormalCombineInput
        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
        else MoriEPLLCombineInput
    )
    return cls(
        hidden_states=output,
        topk_ids=dispatch_output.origin_topk_ids,
        topk_weights=dispatch_output.origin_topk_weights,
    )
