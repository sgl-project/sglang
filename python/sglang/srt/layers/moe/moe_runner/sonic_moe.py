from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class SonicMoeRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.SONIC_MOE


@dataclass
class SonicMoeRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.SONIC_MOE


@dataclass
class SonicMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    # No quantization flags for initial implementation


class SonicMoeRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        # Trivial initialization only, no complex setup

    def run(
        self,
        runner_input: SonicMoeRunnerInput,
        quant_info: SonicMoeQuantInfo,
        running_state: dict,
    ) -> SonicMoeRunnerOutput:
        raise NotImplementedError(
            "SonicMoE uses fused function path only. "
            "This run() method should not be called."
        )

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.SONIC_MOE


@register_fused_func("none", "sonic_moe")
def fused_experts_none_to_sonic_moe(
    dispatch_output: StandardDispatchOutput,
    quant_info: SonicMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """
    Fused expert computation for SonicMoE backend.
    Follows similar pattern to Triton implementation but uses SonicMoE kernels.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    w13_weight = quant_info.w13_weight
    w2_weight = quant_info.w2_weight

    output = _sonic_moe_forward_placeholder(
        hidden_states=hidden_states,
        w13=w13_weight,
        w2=w2_weight,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        config=runner_config,
    )

    return StandardCombineInput(hidden_states=output)


def _sonic_moe_forward_placeholder(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    config: MoeRunnerConfig,
) -> torch.Tensor:
    """
    Placeholder for actual SonicMoE kernel.
    Replace this with the real SonicMoE implementation.
    """
    raise NotImplementedError("SonicMoE fused kernel is not implemented yet.")
