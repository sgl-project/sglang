from typing import Optional

import torch

from sglang.srt.utils import direct_register_custom_op


def trtllm_fp8_block_scale_moe_wrapper(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
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
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    # lazy import
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
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
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

    return trtllm_fp8_block_scale_moe(**kwargs)


def fake_trtllm_fp8_block_scale_moe_wrapper(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
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
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    return torch.empty(
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
    )


direct_register_custom_op(
    op_name="trtllm_fp8_block_scale_moe_wrapper",
    op_func=trtllm_fp8_block_scale_moe_wrapper,
    mutates_args=[],
    fake_impl=fake_trtllm_fp8_block_scale_moe_wrapper,
)


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

    return trtllm_fp8_per_tensor_scale_moe(**kwargs)


def fake_trtllm_fp8_per_tensor_scale_moe_wrapper(
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
) -> torch.Tensor:
    return torch.empty(
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
    )


direct_register_custom_op(
    op_name="trtllm_fp8_per_tensor_scale_moe",
    op_func=trtllm_fp8_per_tensor_scale_moe_wrapper,
    mutates_args=[],
    fake_impl=fake_trtllm_fp8_per_tensor_scale_moe_wrapper,
)
