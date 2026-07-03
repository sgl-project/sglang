from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

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
from sglang.srt.utils import is_cuda, is_gfx95_supported, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import DeepEPv2DispatchOutput
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class TritonRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    use_mxfp8: bool = False
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    per_channel_quant: bool = False
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_zp: Optional[torch.Tensor] = None
    w2_zp: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TritonRunnerOutput:
        if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
            from sglang.srt.layers.moe.moe_runner.triton_utils.mxfp8_moe_amd_gfx95 import (
                fused_experts_mxfp8,
            )

            out = fused_experts_mxfp8(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w2_weight,
                runner_input.topk_weights,
                runner_input.topk_ids,
                quant_info.w13_scale,
                quant_info.w2_scale,
                b1=quant_info.b13,
                b2=quant_info.b2,
                activation=self.config.activation,
                is_gated=self.config.is_gated,
                no_combine=self.config.no_combine,
                inplace=self.config.inplace,
                apply_router_weight_on_input=self.config.apply_router_weight_on_input,
                routed_scaling_factor=self.config.routed_scaling_factor,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_limit=self.config.gemm1_clamp_limit,
                swiglu_limit=self.config.swiglu_limit,
                gate_up_interleaved=self.config.gate_up_interleaved,
            )
            return TritonRunnerOutput(hidden_states=out)

        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
        )

        filter_expert = (
            self.config.num_experts is None
            or self.config.num_experts != self.config.num_local_experts
        )

        out = _fused_moe_kernel_sequence(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            runner_input.topk_weights,
            runner_input.topk_ids,
            runner_input.sorted_token_ids,
            runner_input.expert_ids,
            runner_input.num_tokens_post_padded,
            running_state["config"],
            running_state.get("down_config"),
            running_state.get("down_moe_use_tma", False),
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
            activation=self.config.activation,
            is_gated=self.config.is_gated,
            no_combine=self.config.no_combine,
            inplace=self.config.inplace,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            routed_scaling_factor=running_state.get(
                "deepep_v2_routed_scaling_factor", self.config.routed_scaling_factor
            ),
            gemm1_alpha=self.config.gemm1_alpha,
            gemm1_limit=self.config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            hooks=hooks,
            swiglu_limit=self.config.swiglu_limit,
        )

        return TritonRunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
        from sglang.srt.layers.moe.moe_runner.triton_utils.mxfp8_moe_amd_gfx95 import (
            fused_experts_mxfp8,
        )

        topk_weights, topk_ids, _ = dispatch_output.topk_output
        output = fused_experts_mxfp8(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            b1=quant_info.b13,
            b2=quant_info.b2,
            activation=runner_config.activation,
            is_gated=runner_config.is_gated,
            no_combine=runner_config.no_combine,
            inplace=runner_config.inplace,
            apply_router_weight_on_input=runner_config.apply_router_weight_on_input,
            routed_scaling_factor=runner_config.routed_scaling_factor,
            gemm1_alpha=runner_config.gemm1_alpha,
            gemm1_limit=runner_config.gemm1_clamp_limit,
            swiglu_limit=runner_config.swiglu_limit,
            gate_up_interleaved=runner_config.gate_up_interleaved,
        )
    else:
        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )

        output = fused_experts(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_output=dispatch_output.topk_output,
            moe_runner_config=runner_config,
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
        )

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("standard", "triton")
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_output.topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "standard")
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )


def _prepare_triton_runner_input(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_info: TritonMoeQuantInfo,
    running_state: dict,
) -> TritonRunnerInput:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )
    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma
    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_pre_permute("deepep_v2", "triton")
def pre_permute_deepep_v2_to_triton(
    dispatch_output: DeepEPv2DispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    hidden_states, hidden_states_scale, topk_ids, topk_weights, *_ = dispatch_output
    if hidden_states_scale is not None or hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(
            "DeepEP v2 -> Triton expects BF16 dispatch output without activation scales. "
            "Use --deepep-v2-dispatcher-output-dtype bf16."
        )
    # A2A EP combine inputs are kept unscaled. The model-level MoE forward
    # applies the routed scaling factor once after combine.
    running_state["deepep_v2_routed_scaling_factor"] = None
    valid_rows = (topk_ids >= 0).any(dim=1)
    running_state["deepep_v2_output_shape"] = hidden_states.shape
    running_state["deepep_v2_valid_rows"] = valid_rows
    running_state["deepep_v2_topk_ids"] = topk_ids
    running_state["deepep_v2_topk_weights"] = topk_weights
    hidden_states = hidden_states[valid_rows].contiguous()
    topk_ids = topk_ids[valid_rows].contiguous()
    topk_weights = topk_weights[valid_rows].contiguous()
    return _prepare_triton_runner_input(
        hidden_states, topk_ids, topk_weights, quant_info, running_state
    )


@register_post_permute("triton", "deepep_v2")
def post_permute_triton_to_deepep_v2(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import DeepEPv2CombineInput

    valid_rows = running_state["deepep_v2_valid_rows"]
    output = torch.zeros(
        running_state["deepep_v2_output_shape"],
        device=runner_output.hidden_states.device,
        dtype=runner_output.hidden_states.dtype,
    )
    output[valid_rows] = runner_output.hidden_states
    return DeepEPv2CombineInput(
        hidden_states=output,
        topk_ids=running_state["deepep_v2_topk_ids"],
        topk_weights=running_state["deepep_v2_topk_weights"],
    )
