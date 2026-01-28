from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
)
from sglang.srt.utils.common import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


def align_fp8_moe_weights_for_flashinfer_trtllm(layer: Module) -> None:
    """Prepare FP8 MoE weights/scales for FlashInfer TRT-LLM kernels."""
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a

    # Note: No need to swap W13 halves, they are already in the correct order:
    # [Gate, Up]
    w13_weight = cast(torch.Tensor, layer.w13_weight)
    w2_weight = cast(torch.Tensor, layer.w2_weight)
    num_experts, two_n, hidden = w13_weight.shape

    w13_interleaved_list = [
        reorder_rows_for_gated_act_gemm(w13_weight[i]) for i in range(num_experts)
    ]
    w13_interleaved: torch.Tensor = torch.stack(w13_interleaved_list).reshape(
        num_experts, two_n, hidden
    )

    # Shuffle weights for transposed MMA output (both W13, W2)
    epilogue_tile_m = 128
    w13_shuffled = [
        shuffle_matrix_a(w13_interleaved[i].view(torch.uint8), epilogue_tile_m)
        for i in range(num_experts)
    ]
    w2_shuffled = [
        shuffle_matrix_a(w2_weight[i].view(torch.uint8), epilogue_tile_m)
        for i in range(num_experts)
    ]

    layer.w13_weight = Parameter(
        torch.stack(w13_shuffled).view(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.w2_weight = Parameter(
        torch.stack(w2_shuffled).view(torch.float8_e4m3fn),
        requires_grad=False,
    )

    # Precompute and register per-expert output scaling factors for FI MoE.
    # Note: w13_input_scale and w2_input_scale are scalar Parameters post-reduction.
    assert hasattr(layer, "w13_input_scale") and layer.w13_input_scale is not None
    assert hasattr(layer, "w2_input_scale") and layer.w2_input_scale is not None
    assert hasattr(layer, "w13_weight_scale") and layer.w13_weight_scale is not None
    assert hasattr(layer, "w2_weight_scale") and layer.w2_weight_scale is not None

    input_scale = cast(torch.Tensor, layer.w13_input_scale).to(torch.float32)
    activation_scale = cast(torch.Tensor, layer.w2_input_scale).to(torch.float32)
    w13_weight_scale = cast(torch.Tensor, layer.w13_weight_scale).to(torch.float32)
    w2_weight_scale = cast(torch.Tensor, layer.w2_weight_scale).to(torch.float32)

    output1_scales_scalar = w13_weight_scale * input_scale * (1.0 / activation_scale)
    output1_scales_gate_scalar = w13_weight_scale * input_scale
    output2_scales_scalar = activation_scale * w2_weight_scale

    layer.output1_scales_scalar = Parameter(output1_scales_scalar, requires_grad=False)
    layer.output1_scales_gate_scalar = Parameter(
        output1_scales_gate_scalar, requires_grad=False
    )
    layer.output2_scales_scalar = Parameter(output2_scales_scalar, requires_grad=False)


@dataclass
class FlashInferTrtllmFp8MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM FP8 MoE kernels."""

    # Weights
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor

    # Expert-parallel metadata
    global_num_experts: int
    local_expert_offset: int
    local_num_experts: int

    routing_method_type: int

    # Block-quant path
    block_quant: bool
    weight_block_k: int | None = None
    w13_weight_scale_inv: torch.Tensor | None = None
    w2_weight_scale_inv: torch.Tensor | None = None

    # Per-tensor path
    w13_input_scale: torch.Tensor | None = None
    output1_scales_scalar: torch.Tensor | None = None
    output1_scales_gate_scalar: torch.Tensor | None = None
    output2_scales_scalar: torch.Tensor | None = None


@register_fused_func("none", "flashinfer_trtllm")
def fused_experts_none_to_flashinfer_trtllm_fp8(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from flashinfer.fused_moe import (
        trtllm_fp8_block_scale_moe,
        trtllm_fp8_per_tensor_scale_moe,
    )

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    assert runner_config.activation == "silu", "Only silu is supported."
    assert not runner_config.no_combine, "no_combine is not supported for flashinfer."

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_bypassed(topk_output)

    router_logits = topk_output.router_logits
    topk_config = topk_output.topk_config
    correction_bias = (
        None
        if topk_config.correction_bias is None
        else topk_config.correction_bias.to(hidden_states.dtype)
    )

    routing_method_type = quant_info.routing_method_type

    if quant_info.block_quant:
        assert quant_info.weight_block_k is not None
        assert quant_info.w13_weight_scale_inv is not None
        assert quant_info.w2_weight_scale_inv is not None

        a_q, a_sf = per_token_group_quant_fp8(hidden_states, quant_info.weight_block_k)
        a_sf_t = a_sf.t().contiguous()

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            # FIXME: there is a bug in the trtllm_fp8_block_scale_moe.
            # It ignored the `output` argument. https://github.com/flashinfer-ai/flashinfer/blob/da01b1bd8f9f22aec8c0eea189ad54860b034947/flashinfer/fused_moe/core.py#L1323-L1325
            # so we put the whole function under the ``use_symmetric_memory`` context manager.
            # If the bug is fixed, we can only put the output tensor allocation under the context manager.
            output = trtllm_fp8_block_scale_moe(
                routing_logits=(
                    router_logits.to(torch.float32)
                    if routing_method_type == RoutingMethodType.DeepSeekV3
                    else router_logits
                ),
                routing_bias=correction_bias,
                hidden_states=a_q,
                hidden_states_scale=a_sf_t,
                gemm1_weights=quant_info.w13_weight,
                gemm1_weights_scale=quant_info.w13_weight_scale_inv,
                gemm2_weights=quant_info.w2_weight,
                gemm2_weights_scale=quant_info.w2_weight_scale_inv,
                num_experts=quant_info.global_num_experts,
                top_k=topk_config.top_k,
                n_group=topk_config.num_expert_group,
                topk_group=topk_config.topk_group,
                intermediate_size=int(quant_info.w2_weight.shape[2]),
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),
                routing_method_type=routing_method_type,
                use_shuffled_weight=False,
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
            )
    else:
        assert quant_info.w13_input_scale is not None
        assert quant_info.output1_scales_scalar is not None
        assert quant_info.output1_scales_gate_scalar is not None
        assert quant_info.output2_scales_scalar is not None

        a_q, _ = scaled_fp8_quant(hidden_states, quant_info.w13_input_scale)
        routing_bias_cast = (
            None if correction_bias is None else correction_bias.to(torch.bfloat16)
        )

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            output = trtllm_fp8_per_tensor_scale_moe(
                routing_logits=router_logits.to(torch.bfloat16),
                routing_bias=routing_bias_cast,
                hidden_states=a_q,
                gemm1_weights=quant_info.w13_weight,
                output1_scales_scalar=quant_info.output1_scales_scalar,
                output1_scales_gate_scalar=quant_info.output1_scales_gate_scalar,
                gemm2_weights=quant_info.w2_weight,
                output2_scales_scalar=quant_info.output2_scales_scalar,
                num_experts=quant_info.global_num_experts,
                top_k=topk_config.top_k,
                n_group=topk_config.num_expert_group,
                topk_group=topk_config.topk_group,
                intermediate_size=int(quant_info.w2_weight.shape[2]),
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),
                use_routing_scales_on_input=False,
                routing_method_type=routing_method_type,
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
            )

    return StandardCombineInput(hidden_states=output)
