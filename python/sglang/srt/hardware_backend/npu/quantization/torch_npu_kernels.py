"""Torch_npu kernels MoE runner backend skeleton."""

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

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


# ---------------------------------------------------------------------------
# Runner IO dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TorchNpuKernelsRunnerInput(RunnerInput):
    """Input bundle passed to the torch-npu-kernels runner core."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TORCH_NPU_KERNELS


@dataclass
class TorchNpuKernelsRunnerOutput(RunnerOutput):
    """Output bundle returned from the torch-npu-kernels runner core."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TORCH_NPU_KERNELS


@dataclass
class TorchNpuKernelsQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the torch-npu-kernels backend."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Runner core
# ---------------------------------------------------------------------------


class TorchNpuKernelsRunnerCore(MoeRunnerCore):
    """Execute MoE experts via the external torch_npu_kernels package."""

    def run(
        self,
        runner_input: TorchNpuKernelsRunnerInput,
        quant_info: TorchNpuKernelsQuantInfo,
        running_state: dict,
    ) -> TorchNpuKernelsRunnerOutput:
        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import npu_fused_experts

        hidden_states = runner_input.hidden_states
        topk_ids = runner_input.topk_ids
        topk_weights = runner_input.topk_weights

        '''common_kwargs = dict(
            routing_data=runner_input.routing_data,
            gather_indx=runner_input.gather_indx,
            scatter_indx=None if self.config.no_combine else runner_input.scatter_indx,
            inplace=False,
            activation=self.config.activation,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            global_num_experts=quant_info.global_num_experts,
        )'''


        if True:
            output = npu_fused_experts(
                hidden_states=hidden_states,
                w13=quant_info.w13_weight,
                w13_scale=quant_info.w13_scale,
                w2=quant_info.w2_weight,
                w2_scale=quant_info.w2_scale,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=topk_ids.shape[1],
            )
        else:
            output = torch_npu_fused_moe_without_routing_weights_bf16(
                hidden_states=hidden_states,
                w1=quant_info.w13_weight,
                w2=quant_info.w2_weight,
                **common_kwargs,
            )

        '''if self.config.no_combine:
            tokens = runner_input.hidden_states.shape[0]
            hidden = runner_input.hidden_states.shape[-1]
            total_rows = output.shape[0]
            top_k = total_rows // tokens
            output = output.view(tokens, top_k, hidden)'''

        return TorchNpuKernelsRunnerOutput(hidden_states=output)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TORCH_NPU_KERNELS


# ---------------------------------------------------------------------------
# Permute / fused hooks
# ---------------------------------------------------------------------------


@register_pre_permute("standard", "torch_npu")
def pre_permute_standard_to_torch_npu_kernels(
    dispatch_output: "StandardDispatchOutput",
    quant_info: TorchNpuKernelsQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TorchNpuKernelsRunnerInput:
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    #assert TopKOutputChecker.format_is_triton_kernels(
    #    topk_output
    #), "Torch-npu-kernel runner expects TorchNpuKernelTopKOutput"

    topk_weights, topk_ids, _ = topk_output
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(hidden_states.dtype)

    return TorchNpuKernelsRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


@register_post_permute("torch_npu", "standard")
def post_permute_torch_npu_kernels_to_standard(
    runner_output: TorchNpuKernelsRunnerOutput,
    quant_info: TorchNpuKernelsQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    hidden_states = runner_output.hidden_states

    return StandardCombineInput(hidden_states=hidden_states)
