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

from sglang.srt.hardware_backend.npu.moe.npu_fused_experts import npu_fused_experts_w4a8, npu_fused_experts_w8a8

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
    w13_offset: Optional[torch.Tensor] = None
    w2_offset: Optional[torch.Tensor] = None
    w13_scale_bias: Optional[torch.Tensor] = None
    w2_scale_bias: Optional[torch.Tensor] = None

# ---------------------------------------------------------------------------
# Runner core
# ---------------------------------------------------------------------------

def output_w4a8(hidden_states, quant_info, topk_weights, topk_ids):
    output = npu_fused_experts_w4a8(
            hidden_states=hidden_states,
            w13=quant_info.w13_weight,
            w13_scale=quant_info.w13_scale,
            w13_scale_bias=quant_info.w13_scale_bias,
            w2=quant_info.w2_weight,
            w2_scale=quant_info.w2_scale,
            w2_scale_bias=quant_info.w2_scale_bias,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
        )
    return output

def output_w4a16(hidden_states, quant_info, topk_weights, topk_ids):
    output = npu_fused_experts_w4a8(
            hidden_states=hidden_states,
            w13=quant_info.w13_weight,
            w13_scale=quant_info.w13_scale,
            w13_offset=quant_info.w13_offset,
            w2=quant_info.w2_weight,
            w2_scale=quant_info.w2_scale,
            w2_offset=quant_info.w2_offset,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
        )
    return output

def output_w8a8(hidden_states, quant_info, topk_weights, topk_ids):
    output = npu_fused_experts_w8a8(
            hidden_states=hidden_states,
            w13=quant_info.w13_weight,
            w13_scale=quant_info.w13_scale,
            w13_offset=quant_info.w13_offset,
            w2=quant_info.w2_weight,
            w2_scale=quant_info.w2_scale,
            w2_offset=quant_info.w2_offset,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=topk_ids.shape[1],
        )
    return output


class TorchNpuKernelsRunnerCore(MoeRunnerCore):
    """Execute MoE experts via the external torch_npu_kernels package."""

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        
        if config.quantization == "ModelSlimW4A8Int8MoE":
            self.selected_run = output_w4a8
        elif config.quantization == "NPUCompressedTensorsW4A8Int4DynamicMoEMethod":
            self.selected_run = output_w4a8
        elif config.quantization == "NPUCompressedTensorsW4A16Int4DynamicMoEMethod":
            self.selected_run = output_w4a16
        elif config.quantization == "ModelSlimW8A8Int8MoE":
            self.selected_run = output_w8a8

    def run(
        self,
        runner_input: TorchNpuKernelsRunnerInput,
        quant_info: TorchNpuKernelsQuantInfo,
        running_state: dict,
    ) -> TorchNpuKernelsRunnerOutput:

        hidden_states = runner_input.hidden_states
        topk_ids = runner_input.topk_ids
        topk_weights = runner_input.topk_weights

        output = self.selected_run(hidden_states, quant_info, topk_weights, topk_ids)

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

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
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

