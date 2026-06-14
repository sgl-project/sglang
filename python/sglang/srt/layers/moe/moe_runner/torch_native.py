from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

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
class TorchNativeMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor = None
    w2_weight: torch.Tensor = None
    w13_bias: Optional[torch.Tensor] = None
    w2_bias: Optional[torch.Tensor] = None


@register_fused_func("none", "torch_native")
def fused_experts_none_to_torch_native(
    dispatch_output: StandardDispatchOutput,
    quant_info: TorchNativeMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.fused_moe_native import moe_forward_native
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    proxy = types.SimpleNamespace(
        w13_weight=quant_info.w13_weight,
        w2_weight=quant_info.w2_weight,
        w13_weight_bias=quant_info.w13_bias,
        w2_weight_bias=quant_info.w2_bias,
        num_experts=quant_info.w13_weight.shape[0],
    )

    output = moe_forward_native(proxy, hidden_states, topk_output, runner_config)
    return StandardCombineInput(hidden_states=output)
