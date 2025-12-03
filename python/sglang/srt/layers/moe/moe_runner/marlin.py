from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from python.sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
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
    w13_g_idx: torch.Tensor
    w2_g_idx: torch.Tensor
    w13_g_idx_sort_indices: torch.Tensor
    w2_g_idx_sort_indices: torch.Tensor
    weight_bits: int
    is_k_full: bool

class MarlinRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
    
    def run(
        self,
        runner_input: MarlinRunnerInput,
        quant_info: MarlinMoeQuantInfo,
        running_state: dict,
    ) -> MarlinRunnerOutput:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = runner_input.hidden_states
        
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."
        
        # The input must currently be float16
        orig_dtype = x.dtype
        x = x.half()

        output = fused_marlin_moe(
            x,
            quant_info.w13_qweight,
            quant_info.w2_qweight,
            quant_info.w13_scales,
            quant_info.w2_scales,
            runner_input.router_logits,
            runner_input.topk_weights,
            runner_input.topk_ids,
            g_idx1=quant_info.w13_g_idx,
            g_idx2=quant_info.w2_g_idx,
            sort_indices1=quant_info.w13_g_idx_sort_indices,
            sort_indices2=quant_info.w2_g_idx_sort_indices,
            num_bits=quant_info.weight_bits,
            is_k_full=quant_info.is_k_full,
        ).to(orig_dtype)

        return MarlinRunnerOutput(
            hidden_states=output
        )

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.MARLIN
        
@register_pre_permute("standard", "marlin")
def pre_permute_standard_to_marlin(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> MarlinRunnerInput:
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    assert TopKOutputChecker.format_is_standard(topk_output), (
        "Marlin runner expects StandardTopKOutput"
    )

    return MarlinRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        router_logits=topk_output.router_logits,
    )

@register_post_permute("triton_kernel", "standard")
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