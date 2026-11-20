"""MoE runner for ModelOpt NVFP4 experts on Hopper: the grouped NVFP4 W4A16 op of
``sglang.kernels.ops.moe.nvfp4_w4a16_sm90`` on weights repacked at load. Opt in with
``--moe-runner-backend nvfp4_w4a16_sm90``; standard dispatch only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.moe.nvfp4_w4a16_sm90 import nvfp4_w4a16_sm90_moe
from sglang.kernels.ops.quantization.nvfp4_sm90_repack import Nvfp4Sm90Experts
from sglang.srt.layers import zero_copy_context
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
class Nvfp4W4a16Sm90MoeQuantInfo(MoeQuantInfo):
    experts: Nvfp4Sm90Experts


@register_fused_func("none", "nvfp4_w4a16_sm90")
def fused_experts_none_to_nvfp4_w4a16_sm90(
    dispatch_output: StandardDispatchOutput,
    quant_info: Nvfp4W4a16Sm90MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.moe_runner.marlin import _fused_unpack_packed_topk
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    if TopKOutputChecker.format_is_packed(topk_output):
        # The fused gate+topk kernel packs (expert_id << 16) | bf16 weight bits.
        topk_ids, topk_weights = _fused_unpack_packed_topk(topk_output.packed_topk_ids)
    else:
        topk_ids, topk_weights = topk_output.topk_ids, topk_output.topk_weights
    out = zero_copy_context.get_moe_output(hidden_states)
    output = nvfp4_w4a16_sm90_moe(
        hidden_states,
        topk_ids.to(torch.int32).contiguous(),
        quant_info.experts,
        topk_weights=topk_weights.to(torch.float32).contiguous(),
        out=out,
        scale=runner_config.routed_scaling_factor or 1.0,
        swiglu_limit=runner_config.swiglu_limit,
    )
    return StandardCombineInput(hidden_states=output)
