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
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class MarlinRunnerInput(RunnerInput):
    """Input bundle passed to the Marlin runner core."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinRunnerOutput(RunnerOutput):
    """Output bundle returned from the Marlin runner core."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@dataclass
class MarlinMoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the Marlin backend."""

    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_g_idx_sort_indices: Optional[torch.Tensor]
    w2_g_idx_sort_indices: Optional[torch.Tensor]
    weight_bits: int

    # GPTQ specific (Optional)
    w13_g_idx: Optional[torch.Tensor] = None
    w2_g_idx: Optional[torch.Tensor] = None
    is_k_full: bool = True

    # AWQ specific (Optional)
    w13_qzeros: Optional[torch.Tensor] = None
    w2_qzeros: Optional[torch.Tensor] = None

    # Optional
    expert_map: Optional[torch.Tensor] = None


class MarlinRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        self.workspace: Optional[torch.Tensor] = None

    def run(
        self,
        runner_input: MarlinRunnerInput,
        quant_info: MarlinMoeQuantInfo,
        running_state: dict,
    ) -> MarlinRunnerOutput:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )

        x = runner_input.hidden_states

        assert self.config.activation == "silu", "Only SiLU activation is supported."

        if self.workspace is None:
            self.workspace = marlin_make_workspace(x.device, max_blocks_per_sm=4)

        output = fused_marlin_moe(
            hidden_states=x,
            w1=quant_info.w13_qweight,
            w2=quant_info.w2_qweight,
            w1_scale=quant_info.w13_scales,
            w2_scale=quant_info.w2_scales,
            gating_output=runner_input.router_logits,
            topk_weights=runner_input.topk_weights,
            topk_ids=runner_input.topk_ids,
            expert_map=quant_info.expert_map,
            g_idx1=quant_info.w13_g_idx,
            g_idx2=quant_info.w2_g_idx,
            sort_indices1=quant_info.w13_g_idx_sort_indices,
            sort_indices2=quant_info.w2_g_idx_sort_indices,
            w1_zeros=quant_info.w13_qzeros,
            w2_zeros=quant_info.w2_qzeros,
            workspace=self.workspace,
            num_bits=quant_info.weight_bits,
            is_k_full=quant_info.is_k_full,
            inplace=self.config.inplace,
            routed_scaling_factor=self.config.routed_scaling_factor,
        ).to(x.dtype)

        return MarlinRunnerOutput(hidden_states=output)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN


@register_pre_permute("standard", "marlin")
def pre_permute_standard_to_marlin(
    dispatch_output: StandardDispatchOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> MarlinRunnerInput:
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    assert TopKOutputChecker.format_is_standard(
        topk_output
    ), "Marlin runner expects StandardTopKOutput"

    return MarlinRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        router_logits=topk_output.router_logits,
    )


@register_post_permute("marlin", "standard")
def post_permute_marlin_to_standard(
    runner_output: MarlinRunnerOutput,
    quant_info: MarlinMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )
