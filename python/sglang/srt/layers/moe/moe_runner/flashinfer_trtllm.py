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
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.utils.common import (
    is_cuda_alike,
    is_flashinfer_available,
    is_sm120_supported,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

if is_flashinfer_available() and is_sm120_supported():
    from flashinfer import fp4_quantize
elif is_cuda_alike():
    from sgl_kernel import scaled_fp4_quant as fp4_quantize
else:
    fp4_quantize = None


def align_fp8_moe_weights_for_flashinfer_trtllm(
    layer: Module, swap_w13_halves: bool = False
) -> None:
    """Prepare FP8 MoE weights/scales for FlashInfer TRT-LLM kernels.

    Args:
        layer: The MoE layer to process.
        swap_w13_halves: If True, swap W13 halves from [Up, Gate] to [Gate, Up].
            This is needed for ModelOpt FP8 checkpoints which store weights in
            [Up, Gate] order, while regular FP8 checkpoints store them in [Gate, Up].
    """
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a

    w13_weight = cast(torch.Tensor, layer.w13_weight)
    w2_weight = cast(torch.Tensor, layer.w2_weight)
    num_experts, two_n, hidden = w13_weight.shape

    # Optionally swap W13 halves: [Up, Gate] -> [Gate, Up]
    if swap_w13_halves:
        inter = two_n // 2
        w13_weight = (
            w13_weight.reshape(num_experts, 2, inter, hidden)
            .flip(dims=[1])
            .reshape(num_experts, two_n, hidden)
        )

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


def align_fp4_moe_weights_for_flashinfer_trtllm(layer: Module) -> None:
    """Prepare FP4 MoE weights/scales for FlashInfer TRT-LLM kernels.

    This function handles the weight transformation needed for FP4 TRTLLM MoE:
    - Reorders weights for gated activation GEMM
    - Shuffles weights and scales for transposed MMA output
    - Computes the output scale factors
    """
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    w13_weight = cast(torch.Tensor, layer.w13_weight)
    w2_weight = cast(torch.Tensor, layer.w2_weight)
    w13_weight_scale = cast(torch.Tensor, layer.w13_weight_scale)
    w2_weight_scale = cast(torch.Tensor, layer.w2_weight_scale)

    (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    ) = prepare_static_weights_for_trtllm_fp4_moe(
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w2_weight.size(-2),  # hidden_size
        w13_weight.size(-2) // 2,  # intermediate_size
        w13_weight.size(0),  # num_experts
    )

    # Set flashinfer parameters
    copy_or_rebind_param(
        layer, "gemm1_weights_fp4_shuffled", gemm1_weights_fp4_shuffled
    )
    copy_or_rebind_param(
        layer, "gemm2_weights_fp4_shuffled", gemm2_weights_fp4_shuffled
    )
    copy_or_rebind_param(layer, "gemm1_scales_fp4_shuffled", gemm1_scales_fp4_shuffled)
    copy_or_rebind_param(layer, "gemm2_scales_fp4_shuffled", gemm2_scales_fp4_shuffled)

    # Compute additional scaling factor needed for TRT-LLM
    w2_input_scale_quant = cast(torch.Tensor, layer.w2_input_scale_quant)
    g1_alphas = cast(torch.Tensor, layer.g1_alphas)
    copy_or_rebind_param(
        layer,
        "g1_scale_c",
        (w2_input_scale_quant * g1_alphas).to(torch.float32),
    )

    # Clean up weights that won't be used by TRT-LLM
    del (
        layer.w2_weight,
        layer.w2_weight_scale,
        layer.w13_weight,
        layer.w13_weight_scale,
    )


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
    intermediate_size: int

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
    use_routing_scales_on_input: bool = False


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
                n_group=(
                    topk_config.num_expert_group if topk_config.num_expert_group else 0
                ),
                topk_group=topk_config.topk_group if topk_config.topk_group else 0,
                intermediate_size=quant_info.intermediate_size,
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
                n_group=(
                    topk_config.num_expert_group if topk_config.num_expert_group else 0
                ),
                topk_group=topk_config.topk_group if topk_config.topk_group else 0,
                intermediate_size=quant_info.intermediate_size,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),
                use_routing_scales_on_input=quant_info.use_routing_scales_on_input,
                routing_method_type=routing_method_type,
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
            )

    return StandardCombineInput(hidden_states=output)


@dataclass
class FlashInferTrtllmFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM FP4 MoE kernels."""

    # Shuffled FP4 weights (processed by align_fp4_moe_weights_for_flashinfer_trtllm)
    gemm1_weights_fp4_shuffled: torch.Tensor
    gemm2_weights_fp4_shuffled: torch.Tensor
    gemm1_scales_fp4_shuffled: torch.Tensor
    gemm2_scales_fp4_shuffled: torch.Tensor

    # Scaling factors
    g1_scale_c: torch.Tensor
    g1_alphas: torch.Tensor
    g2_alphas: torch.Tensor
    w13_input_scale_quant: torch.Tensor

    # Expert-parallel metadata
    global_num_experts: int
    local_expert_offset: int
    local_num_experts: int
    intermediate_size_per_partition: int

    routing_method_type: int


def quantize_hidden_states_fp4(
    hidden_states: torch.Tensor,
    input_scale_quant: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize hidden states to FP4 for TRTLLM MoE.

    Global scale factor is set by ModelOptNvFp4FusedMoEMethod during weight loading.
    Only block scales are computed at runtime for efficiency.

    Returns (packed_fp4_uint8, scale_float8_e4m3fn_runtime)
    """

    # flashinfer.fp4_quantize returns (packed_uint8, scale_fp8)
    # Only the block scales are computed at runtime
    hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
        hidden_states,
        input_scale_quant,
        16,  # sf_vec_size
        False,  # use_ue8m0
        False,  # is_sf_swizzled_layout
    )

    seq_len, hidden_size = hidden_states.shape
    hs_fp4 = hs_fp4_bytes.reshape(seq_len, hidden_size // 2)
    # TRT-LLM expects hidden state scales shaped as [seq_len, hidden_size // 16]
    hs_sf = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(seq_len, hidden_size // 16)

    return hs_fp4, hs_sf


def fused_experts_none_to_flashinfer_trtllm_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """FlashInfer TRTLLM FP4 MoE forward pass.

    This function handles the FP4 TRTLLM MoE path that was previously in
    FlashInferFP4MoE.forward_impl and ModelOptNvFp4FusedMoEMethod.apply.
    """
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    assert runner_config.activation == "silu", "Only silu is supported for FP4 MoE."
    assert runner_config.is_gated, "Only gated MoEs are supported for FP4 MoE."

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_bypassed(topk_output)

    router_logits = topk_output.router_logits
    topk_config = topk_output.topk_config
    routing_method_type = quant_info.routing_method_type

    # Quantize hidden states to FP4
    hs_fp4, hs_scale_linear = quantize_hidden_states_fp4(
        hidden_states, quant_info.w13_input_scale_quant
    )

    # DeepSeekV3 style routing requires float32 router logits
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        router_logits = router_logits.to(torch.float32)

    correction_bias = (
        None
        if topk_config.correction_bias is None
        else topk_config.correction_bias.to(hidden_states.dtype)
    )

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        num_tokens = hs_fp4.shape[0]
        hidden_size = (
            hs_fp4.shape[-1] * 2 if hs_fp4.dtype == torch.uint8 else hs_fp4.shape[-1]
        )
        symm_output = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hs_fp4.device
        )

    result = trtllm_fp4_block_scale_moe(
        routing_logits=router_logits,
        routing_bias=correction_bias,
        hidden_states=hs_fp4,
        hidden_states_scale=hs_scale_linear.view(torch.float8_e4m3fn).flatten(),
        gemm1_weights=quant_info.gemm1_weights_fp4_shuffled,
        gemm1_weights_scale=quant_info.gemm1_scales_fp4_shuffled.view(
            torch.float8_e4m3fn
        ),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=quant_info.gemm2_weights_fp4_shuffled,
        gemm2_weights_scale=quant_info.gemm2_scales_fp4_shuffled.view(
            torch.float8_e4m3fn
        ),
        gemm2_bias=None,
        output1_scale_scalar=quant_info.g1_scale_c,
        output1_scale_gate_scalar=quant_info.g1_alphas,
        output2_scale_scalar=quant_info.g2_alphas,
        num_experts=quant_info.global_num_experts,
        top_k=topk_config.top_k,
        n_group=topk_config.num_expert_group,
        topk_group=topk_config.topk_group,
        intermediate_size=quant_info.intermediate_size_per_partition,
        local_expert_offset=quant_info.local_expert_offset,
        local_num_experts=quant_info.local_num_experts,
        routed_scaling_factor=runner_config.routed_scaling_factor,
        tile_tokens_dim=None,
        routing_method_type=(
            routing_method_type
            if routing_method_type is not None
            else RoutingMethodType.Default
        ),
        do_finalize=True,
        tune_max_num_tokens=next_power_of_2(hs_fp4.shape[0]),
        output=symm_output,
    )[0]

    return StandardCombineInput(hidden_states=result)


@dataclass
class FlashInferTrtllmBf16MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM BF16 MoE kernels."""

    gemm1_weights: torch.Tensor
    gemm2_weights: torch.Tensor

    # Expert-parallel metadata
    global_num_experts: int
    local_expert_offset: int


def fused_experts_none_to_flashinfer_trtllm_bf16(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmBf16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    # lazy import
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    try:
        from flashinfer.fused_moe import trtllm_bf16_moe
    except ImportError as e:
        raise ImportError(
            "Can't import trtllm_bf16_moe from flashinfer. "
            "Please check flashinfer version to use bf16 with flashinfer_trtllm backend."
        ) from e

    assert (
        runner_config.activation == "silu"
    ), "Only silu is supported for flashinfer trtllm moe"
    assert (
        dispatch_output.topk_output.topk_config.renormalize
    ), "Renormalize is required for flashinfer trtllm moe"
    assert (
        runner_config.num_fused_shared_experts == 0
    ), "Fused shared experts are not supported for flashinfer trtllm moe"
    assert (
        runner_config.is_gated
    ), "Only gated MoEs are supported for flashinfer trtllm moe"
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert TopKOutputChecker.format_is_bypassed(dispatch_output.topk_output)

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_config = topk_output.topk_config

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):

        # Call the fused kernel
        final_hidden_states = trtllm_bf16_moe(
            routing_logits=topk_output.router_logits,
            routing_bias=topk_config.correction_bias,
            hidden_states=hidden_states,
            gemm1_weights=quant_info.gemm1_weights,
            gemm2_weights=quant_info.gemm2_weights,
            num_experts=quant_info.global_num_experts,
            top_k=topk_config.top_k,
            n_group=topk_config.num_expert_group,
            topk_group=topk_config.topk_group,
            intermediate_size=runner_config.intermediate_size_per_partition,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=runner_config.num_local_experts,
            routing_method_type=runner_config.routing_method_type,
            routed_scaling_factor=runner_config.routed_scaling_factor,
            tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),
        )

    return StandardCombineInput(hidden_states=final_hidden_states)


@register_fused_func("none", "flashinfer_trtllm")
def fused_experts_none_to_flashinfer_trtllm(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Dispatch to FP8 or FP4 FlashInfer TRT-LLM MoE based on quant_info type."""
    if isinstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch_output, quant_info, runner_config
        )
    if isinstance(quant_info, FlashInferTrtllmFp8MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_fp8(
            dispatch_output, quant_info, runner_config
        )
    if isinstance(quant_info, FlashInferTrtllmBf16MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_bf16(
            dispatch_output, quant_info, runner_config
        )
    raise TypeError(
        f"Unexpected quant_info type for flashinfer_trtllm: {type(quant_info)}"
    )
