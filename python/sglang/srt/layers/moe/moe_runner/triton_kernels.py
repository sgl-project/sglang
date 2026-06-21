"""Triton kernels MoE runner backend skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

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
    from triton_kernels.matmul import PrecisionConfig
    from triton_kernels.tensor_details.ragged_tensor import RaggedTensorMetadata

    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


# ---------------------------------------------------------------------------
# Runner IO dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TritonKernelsRunnerInput(RunnerInput):
    """Input bundle passed to the triton-kernels runner core."""

    hidden_states: torch.Tensor
    a_ragged_metadata: RaggedTensorMetadata
    gather_indx: torch.Tensor
    scatter_indx: torch.Tensor
    gate_scal: torch.Tensor
    n_expts_act: int

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON_KERNELS


@dataclass
class TritonKernelsRunnerOutput(RunnerOutput):
    """Output bundle returned from the triton-kernels runner core."""

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON_KERNELS


@dataclass
class TritonKernelsQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by the triton-kernels backend."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_bias: Optional[torch.Tensor] = None
    w2_bias: Optional[torch.Tensor] = None
    w13_precision_config: Optional[PrecisionConfig] = None
    w2_precision_config: Optional[PrecisionConfig] = None
    global_num_experts: int = -1


# ---------------------------------------------------------------------------
# Runner core
# ---------------------------------------------------------------------------


class TritonKernelsRunnerCore(MoeRunnerCore):
    """Execute MoE experts via the external triton_kernels package."""

    def run(
        self,
        runner_input: TritonKernelsRunnerInput,
        quant_info: TritonKernelsQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TritonKernelsRunnerOutput:
        from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
            triton_kernel_fused_experts,
            triton_kernel_fused_experts_with_bias,
        )

        assert (
            self.config.is_gated
        ), "Only gated MoEs are supported for Triton Kernels runner"

        hidden_states = runner_input.hidden_states

        common_kwargs = dict(
            a_ragged_metadata=runner_input.a_ragged_metadata,
            gather_indx=runner_input.gather_indx,
            scatter_indx=None if self.config.no_combine else runner_input.scatter_indx,
            gate_scal=runner_input.gate_scal,
            n_expts_act=runner_input.n_expts_act,
            inplace=False,
            activation=self.config.activation,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            global_num_experts=quant_info.global_num_experts,
        )

        has_bias = quant_info.w13_bias is not None or quant_info.w2_bias is not None

        if has_bias:
            assert (
                quant_info.w13_bias is not None and quant_info.w2_bias is not None
            ), "Bias execution requires both w13_bias and w2_bias"
            output = triton_kernel_fused_experts_with_bias(
                hidden_states=hidden_states,
                w1=quant_info.w13_weight,
                w1_pcg=quant_info.w13_precision_config,
                b1=quant_info.w13_bias,
                w2=quant_info.w2_weight,
                w2_pcg=quant_info.w2_precision_config,
                b2=quant_info.w2_bias,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_clamp_limit=self.config.gemm1_clamp_limit,
                **common_kwargs,
            )
        else:
            output = triton_kernel_fused_experts(
                hidden_states=hidden_states,
                w1=quant_info.w13_weight,
                w2=quant_info.w2_weight,
                **common_kwargs,
            )

        tokens = runner_input.hidden_states.shape[0]
        hidden = runner_input.hidden_states.shape[-1]
        total_rows = output.shape[0]
        top_k = runner_input.n_expts_act
        assert total_rows == tokens * top_k, (
            "Triton-kernels MoE output rows must match tokens * active experts, "
            f"got {total_rows=} for {tokens=} and {top_k=}"
        )

        if self.config.no_combine:
            output = output.view(tokens, top_k, hidden)
        else:
            output = output.view(tokens, top_k, hidden).sum(dim=1)

        return TritonKernelsRunnerOutput(hidden_states=output)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON_KERNELS


# ---------------------------------------------------------------------------
# Permute / fused hooks
# ---------------------------------------------------------------------------


@register_pre_permute("standard", "triton_kernel")
def pre_permute_standard_to_triton_kernels(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonKernelsQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonKernelsRunnerInput:
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    assert TopKOutputChecker.format_is_triton_kernels(
        topk_output
    ), "Triton-kernel runner expects TritonKernelTopKOutput"

    a_ragged_metadata, gather_indx, scatter_indx, gate_scal, n_expts_act = topk_output

    return TritonKernelsRunnerInput(
        hidden_states=hidden_states,
        a_ragged_metadata=a_ragged_metadata,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        gate_scal=gate_scal,
        n_expts_act=n_expts_act,
    )


@register_post_permute("triton_kernel", "standard")
def post_permute_triton_kernels_to_standard(
    runner_output: TritonKernelsRunnerOutput,
    quant_info: TritonKernelsQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = runner_output.hidden_states

    if (
        runner_config.routed_scaling_factor is not None
        and runner_config.routed_scaling_factor != 1.0
        and not runner_config.no_combine
    ):
        hidden_states.mul_(runner_config.routed_scaling_factor)

    return StandardCombineInput(hidden_states=hidden_states)
