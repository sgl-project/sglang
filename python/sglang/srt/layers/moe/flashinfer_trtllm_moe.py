from typing import Optional

import torch

from sglang.srt.utils.custom_op import register_custom_op


def _fake_fp8_block_scale_moe_out(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Optional[int] = None,
    activation_type: Optional[int] = None,
) -> None:
    return None


@register_custom_op(
    fake_impl=_fake_fp8_block_scale_moe_out,
    mutates_args=["output"],
)
def trtllm_fp8_block_scale_moe_out_wrapper(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Optional[int] = None,
    activation_type: Optional[int] = None,
) -> None:
    try:
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
    except ImportError as e:
        raise ImportError(
            "Can't import trtllm_fp8_block_scale_moe from flashinfer. "
            "Please check flashinfer version."
        ) from e

    kwargs = {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm1_alpha": gemm1_alpha,
        "gemm1_beta": gemm1_beta,
        "gemm1_clamp_limit": gemm1_clamp_limit,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "output": output,
        "num_experts": num_experts,
        "top_k": top_k,
        "n_group": n_group,
        "topk_group": topk_group,
        "intermediate_size": intermediate_size,
        "local_expert_offset": local_expert_offset,
        "local_num_experts": local_num_experts,
        "routed_scaling_factor": routed_scaling_factor,
        "routing_method_type": routing_method_type,
        "use_shuffled_weight": use_shuffled_weight,
        "weight_layout": weight_layout,
        "enable_pdl": enable_pdl,
        "tune_max_num_tokens": tune_max_num_tokens,
    }
    if fp8_quantization_type is not None:
        from flashinfer.fused_moe import Fp8QuantizationType

        kwargs["fp8_quantization_type"] = Fp8QuantizationType(fp8_quantization_type)

    if activation_type is not None:
        from flashinfer.fused_moe.core import ActivationType

        kwargs["activation_type"] = ActivationType(activation_type)

    trtllm_fp8_block_scale_moe(**kwargs)


def _fake_fp8_block_scale_routed_moe_out(
    topk_ids: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    output: torch.Tensor,
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Optional[int] = None,
    activation_type: Optional[int] = None,
) -> None:
    return None


@register_custom_op(
    fake_impl=_fake_fp8_block_scale_routed_moe_out,
    mutates_args=["output"],
)
def trtllm_fp8_block_scale_routed_moe_out_wrapper(
    topk_ids: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    output: torch.Tensor,
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Optional[int] = None,
    activation_type: Optional[int] = None,
) -> None:
    try:
        from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe
    except ImportError as e:
        raise ImportError(
            "Can't import trtllm_fp8_block_scale_routed_moe from flashinfer. "
            "Please check flashinfer version."
        ) from e

    kwargs = {
        "topk_ids": topk_ids if topk_weights is None else (topk_ids, topk_weights),
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm1_alpha": gemm1_alpha,
        "gemm1_beta": gemm1_beta,
        "gemm1_clamp_limit": gemm1_clamp_limit,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "output": output,
        "num_experts": num_experts,
        "top_k": top_k,
        "n_group": n_group,
        "topk_group": topk_group,
        "intermediate_size": intermediate_size,
        "local_expert_offset": local_expert_offset,
        "local_num_experts": local_num_experts,
        "routed_scaling_factor": routed_scaling_factor,
        "routing_method_type": routing_method_type,
        "use_shuffled_weight": use_shuffled_weight,
        "weight_layout": weight_layout,
        "enable_pdl": enable_pdl,
        "tune_max_num_tokens": tune_max_num_tokens,
    }
    if fp8_quantization_type is not None:
        from flashinfer.fused_moe import Fp8QuantizationType

        kwargs["fp8_quantization_type"] = Fp8QuantizationType(fp8_quantization_type)

    if activation_type is not None:
        from flashinfer.fused_moe.core import ActivationType

        kwargs["activation_type"] = ActivationType(activation_type)

    trtllm_fp8_block_scale_routed_moe(**kwargs)


def _fake_fp8_per_tensor_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: Optional[int] = None,
) -> torch.Tensor:
    return torch.empty(
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
    )


@register_custom_op(fake_impl=_fake_fp8_per_tensor_scale_moe)
def trtllm_fp8_per_tensor_scale_moe_wrapper(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: Optional[int] = None,
) -> torch.Tensor:
    # lazy import
    try:
        from flashinfer.fused_moe import trtllm_fp8_per_tensor_scale_moe
    except ImportError as e:
        raise ImportError(
            "Can't import trtllm_fp8_per_tensor_scale_moe from flashinfer. "
            "Please check flashinfer version."
        ) from e

    kwargs = {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "gemm1_weights": gemm1_weights,
        "output1_scales_scalar": output1_scales_scalar,
        "output1_scales_gate_scalar": output1_scales_gate_scalar,
        "gemm2_weights": gemm2_weights,
        "output2_scales_scalar": output2_scales_scalar,
        "num_experts": num_experts,
        "top_k": top_k,
        "n_group": n_group,
        "topk_group": topk_group,
        "intermediate_size": intermediate_size,
        "local_expert_offset": local_expert_offset,
        "local_num_experts": local_num_experts,
        "routed_scaling_factor": routed_scaling_factor,
        "use_routing_scales_on_input": use_routing_scales_on_input,
        "routing_method_type": routing_method_type,
        "enable_pdl": enable_pdl,
        "tune_max_num_tokens": tune_max_num_tokens,
    }

    if activation_type is not None:
        from flashinfer.fused_moe.core import ActivationType

        kwargs["activation_type"] = ActivationType(activation_type)

    return trtllm_fp8_per_tensor_scale_moe(**kwargs)


def _fake_fp8_per_channel_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_per_channel_weight_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_per_channel_weight_scale: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: Optional[int] = None,
    norm_topk_prob: bool = True,
) -> torch.Tensor:
    return torch.empty(
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
    )


def _fake_fp8_per_channel_scale_routed_moe(
    topk_ids: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_per_channel_weight_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_per_channel_weight_scale: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: Optional[int] = None,
) -> torch.Tensor:
    return torch.empty(
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
    )


def _fp8_per_channel_kwargs(
    *,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_per_channel_weight_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_per_channel_weight_scale: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int,
    enable_pdl: Optional[bool],
    tune_max_num_tokens: int,
    activation_type: Optional[int],
) -> dict:
    kwargs = {
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_per_channel_weight_scale": gemm1_per_channel_weight_scale,
        "output1_scale_scalar": output1_scale_scalar,
        "output1_scale_gate_scalar": output1_scale_gate_scalar,
        "gemm2_weights": gemm2_weights,
        "gemm2_per_channel_weight_scale": gemm2_per_channel_weight_scale,
        "output2_scale_scalar": output2_scale_scalar,
        "num_experts": num_experts,
        "top_k": top_k,
        "n_group": n_group,
        "topk_group": topk_group,
        "intermediate_size": intermediate_size,
        "local_expert_offset": local_expert_offset,
        "local_num_experts": local_num_experts,
        "routed_scaling_factor": routed_scaling_factor,
        "use_routing_scales_on_input": use_routing_scales_on_input,
        "routing_method_type": routing_method_type,
        "enable_pdl": enable_pdl,
        "tune_max_num_tokens": tune_max_num_tokens,
    }
    if activation_type is not None:
        from flashinfer.fused_moe.core import ActivationType

        kwargs["activation_type"] = ActivationType(activation_type)
    return kwargs


@register_custom_op(fake_impl=_fake_fp8_per_channel_scale_moe)
def trtllm_fp8_per_channel_scale_moe_wrapper(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_per_channel_weight_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_per_channel_weight_scale: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: Optional[int] = None,
    norm_topk_prob: bool = True,
) -> torch.Tensor:
    try:
        from flashinfer.fused_moe import trtllm_fp8_per_channel_scale_moe
    except ImportError as e:
        raise ImportError(
            "Can't import trtllm_fp8_per_channel_scale_moe from flashinfer. "
            "Please check flashinfer version."
        ) from e

    kwargs = _fp8_per_channel_kwargs(
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_per_channel_weight_scale=gemm1_per_channel_weight_scale,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        gemm2_weights=gemm2_weights,
        gemm2_per_channel_weight_scale=gemm2_per_channel_weight_scale,
        output2_scale_scalar=output2_scale_scalar,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        use_routing_scales_on_input=use_routing_scales_on_input,
        routing_method_type=routing_method_type,
        enable_pdl=enable_pdl,
        tune_max_num_tokens=tune_max_num_tokens,
        activation_type=activation_type,
    )
    return trtllm_fp8_per_channel_scale_moe(
        routing_logits=routing_logits,
        norm_topk_prob=norm_topk_prob,
        **kwargs,
    )


@register_custom_op(fake_impl=_fake_fp8_per_channel_scale_routed_moe)
def trtllm_fp8_per_channel_scale_routed_moe_wrapper(
    topk_ids: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_per_channel_weight_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_per_channel_weight_scale: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: Optional[int] = None,
) -> torch.Tensor:
    try:
        from flashinfer.fused_moe import trtllm_fp8_per_channel_scale_routed_moe
    except ImportError as e:
        raise ImportError(
            "Can't import trtllm_fp8_per_channel_scale_routed_moe from flashinfer. "
            "Please check flashinfer version."
        ) from e

    if topk_weights is not None:
        # FlashInfer's per-channel routed entry point currently accepts only
        # PackedScoreIdx routing, while SGLang's shared routed abstraction now
        # preserves separate ids/weights when available.
        from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import (
            fused_pack_topk,
        )

        topk_ids = fused_pack_topk(topk_ids, topk_weights)

    kwargs = _fp8_per_channel_kwargs(
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_per_channel_weight_scale=gemm1_per_channel_weight_scale,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        gemm2_weights=gemm2_weights,
        gemm2_per_channel_weight_scale=gemm2_per_channel_weight_scale,
        output2_scale_scalar=output2_scale_scalar,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        use_routing_scales_on_input=use_routing_scales_on_input,
        routing_method_type=routing_method_type,
        enable_pdl=enable_pdl,
        tune_max_num_tokens=tune_max_num_tokens,
        activation_type=activation_type,
    )
    return trtllm_fp8_per_channel_scale_routed_moe(topk_ids=topk_ids, **kwargs)
