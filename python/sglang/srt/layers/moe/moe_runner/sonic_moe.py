from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

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

from sglang.srt.layers.moe.sonic_moe.enums import ActivationType


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
    from sglang.srt.layers.moe.sonic_moe.functional import sonic_moe_forward_placeholder
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    w13_weight = quant_info.w13_weight
    w2_weight = quant_info.w2_weight
    b13 = quant_info.b13
    b2 = quant_info.b2

    if runner_config.is_gated and runner_config.activation == "silu":
        activation_type = ActivationType("swiglu")
    else:
        raise NotImplementedError(
            "Only SiLU gated activation is supported in SonicMoE fused path."
        )
    # for debugging purpose
    use_torch = False

    if use_torch:
        output = _sonic_moe_forward_placeholder_torch(
            hidden_states=hidden_states,
            w13=w13_weight,
            b13=b13,
            w2=w2_weight,
            b2=b2,
            router_weights=topk_output.topk_weights,
            selected_experts=topk_output.topk_ids,
            config=runner_config,
        )
    else:

        output = sonic_moe_forward_placeholder(
            hidden_states=hidden_states,
            w13=w13_weight.permute(1, 2, 0),
            b13=b13,
            w2=w2_weight.permute(1, 2, 0),
            b2=b2,
            router_weights=topk_output.topk_weights,
            selected_experts=topk_output.topk_ids,
            activation_type=activation_type,
            config=runner_config,
        )

    return StandardCombineInput(hidden_states=output)


def _sonic_moe_forward_placeholder_torch(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    b13: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    router_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    config: MoeRunnerConfig,
) -> torch.Tensor:
    """
    Placeholder for actual SonicMoE kernel.
    Replace this with the real SonicMoE implementation.
    """
    assert hidden_states.size(0) == selected_experts.size(0), "Batch size mismatch"
    assert hidden_states.size(1) == config.hidden_size, "Hidden size mismatch"
    assert w13.size(0) == config.num_experts, "W13 expert count mismatch"
    assert w13.size(1) == (
        config.intermediate_size_per_partition * 2
        if config.is_gated
        else config.intermediate_size_per_partition
    ), "W13 intermediate size mismatch"
    assert w13.size(2) == config.hidden_size, "W13 hidden size mismatch"
    assert w2.size(0) == config.num_experts, "W2 expert count mismatch"
    assert w2.size(1) == config.hidden_size, "W2 hidden size mismatch"
    assert (
        w2.size(2) == config.intermediate_size_per_partition
    ), "W2 intermediate size mismatch"
    assert selected_experts.size(1) == config.top_k, "Input feature size mismatch"

    T = hidden_states.size(0)

    selected_experts = selected_experts.flatten()

    sorted_expert_idxs, sorted_scattered_idxs = selected_experts.sort()

    expert_frequency = selected_experts.bincount(minlength=config.num_experts).to(
        torch.int32
    )

    # sort and group input tokens according to expert assignment
    fan_in_index = sorted_scattered_idxs // config.top_k

    # gather the gate values for grouped input tokens
    router_weights = router_weights.flatten().to(hidden_states.dtype)
    batch_gates = router_weights[sorted_scattered_idxs]

    hidden_states = hidden_states[fan_in_index]

    hidden_states = _torch_forward(
        hidden_states=hidden_states,
        expert_frequency=expert_frequency,
        weight=w13,
        bias=b13,
        return_list=True,
    )

    hidden_states = [_swiglu(i) for i in hidden_states]

    hidden_states = _torch_forward(
        hidden_states=hidden_states,
        expert_frequency=None,
        weight=w2,
        bias=b2,
        return_list=False,
    )

    hidden_states = hidden_states * batch_gates.unsqueeze(-1)

    zeros = torch.zeros(
        (T, config.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device
    )

    hidden_states = zeros.index_add(0, fan_in_index, hidden_states)

    return hidden_states


def _torch_forward(
    hidden_states: torch.Tensor,
    expert_frequency: torch.Tensor | None,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    return_list: bool = False,
) -> list[torch.Tensor] | torch.Tensor:
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.split(expert_frequency.tolist(), dim=0)
    else:
        assert expert_frequency is None

    hidden_states = [
        F.linear(hidden_states[i], weight[i], None if bias is None else bias[i])
        for i in range(weight.size(0))
    ]

    if not return_list:
        hidden_states = torch.cat(hidden_states, dim=0)

    return hidden_states


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return u * F.silu(g)
