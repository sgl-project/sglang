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

from sonicmoe.count_cumsum import count_cumsum
from sonicmoe.enums import ActivationType
from sonicmoe.functional import TC_topk_router_metadata, _DownProjection, _UpProjection


# import sonic_moe
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

    if runner_config.is_gated and runner_config.activation == "silu":
        activation_type = ActivationType("swiglu")
    else:
        raise NotImplementedError(
            "Only SiLU gated activation is supported in SonicMoE fused path."
        )
    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    w13_weight = quant_info.w13_weight
    w2_weight = quant_info.w2_weight
    b13 = quant_info.b13
    b2 = quant_info.b2

    s = torch.cuda.current_stream()
    with torch.no_grad():
        output = _sonic_moe_forward_placeholder(
            hidden_states=hidden_states,
            w13=w13_weight.permute(1, 2, 0),
            b13=b13,
            w2=w2_weight.permute(1, 2, 0),
            b2=b2,
            router_weights=topk_output.topk_weights,
            selected_experts=topk_output.topk_ids,
            activation_type=activation_type,
            config=runner_config,
            stream=s,
        )

    return StandardCombineInput(hidden_states=output)


def _sonic_moe_forward_placeholder(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    b13: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    router_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    activation_type: ActivationType,
    config: MoeRunnerConfig,
    stream: torch.cuda.Stream,
) -> torch.Tensor:
    """
    Placeholder for actual SonicMoE kernel.
    Replace this with the real SonicMoE implementation.
    """
    assert hidden_states.size(0) == selected_experts.size(0), "Batch size mismatch"
    assert hidden_states.size(1) == config.hidden_size, "Hidden size mismatch"
    assert w13.size(2) == config.num_experts, "W13 expert count mismatch"
    assert w13.size(0) == (
        config.intermediate_size_per_partition * 2
        if config.is_gated
        else config.intermediate_size_per_partition
    ), "W13 intermediate size mismatch"
    assert w13.size(1) == config.hidden_size, "W13 hidden size mismatch"
    assert w2.size(2) == config.num_experts, "W2 expert count mismatch"
    assert w2.size(0) == config.hidden_size, "W2 hidden size mismatch"
    assert (
        w2.size(1) == config.intermediate_size_per_partition
    ), "W2 intermediate size mismatch"
    assert selected_experts.size(1) == config.top_k, "Input feature size mismatch"

    T = hidden_states.size(0)
    K = config.top_k

    expert_frequency, expert_frequency_offset = count_cumsum(
        selected_experts.view(-1), config.num_experts, do_cumsum=True
    )
    (
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    ) = TC_topk_router_metadata(selected_experts, expert_frequency_offset, K)

    y1, z = _UpProjection.apply(
        hidden_states,
        w13,
        b13,
        expert_frequency_offset,
        T * K,
        K,
        stream.cuda_stream,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        False,  # is_varlen_K
        activation_type,
        True,  # inference_mode_enabled
    )

    hidden_states = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        router_weights,
        expert_frequency_offset,
        T,
        K,
        stream.cuda_stream,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        False,  # is_varlen_K
        activation_type,
    )
    return hidden_states
