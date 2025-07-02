# Adapted from https://github.com/vllm-project/vllm/pull/18595/files#diff-f426a6de78c82ffec568eff6811bfbf0043dab5f87f1a8c0cffdbdcb8a81e035
from typing import Optional

import torch
from sgl_kernel import gelu_and_mul, silu_and_mul
from triton_kernels.matmul_ogs import matmul_ogs
from triton_kernels.routing import GatherIndx, RoutingData, ScatterIndx, routing

from sglang.srt.utils import direct_register_custom_op


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    if not renormalize:
        gating_output = torch.softmax(gating_output, dim=-1)
    routing_data, gather_idx, scatter_idx = routing(gating_output, topk, renormalize)

    return triton_kernel_fused_experts(
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        inplace=inplace,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data: RoutingData,
    gather_indx: GatherIndx,
    scatter_indx: ScatterIndx,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:

    assert use_fp8_w8a8 == False, "use_fp8_w8a8 is not supported"
    assert per_channel_quant == False, "per_channel_quant is not supported"
    assert expert_map == None, "expert_map is not supported"
    assert w1_scale == None, "w1_scale is not supported"
    assert w2_scale == None, "w2_scale is not supported"
    assert a1_scale == None, "a1_scale is not supported"
    assert a2_scale == None, "a2_scale is not supported"
    assert block_shape == None, "block_shape is not supported"

    # type check
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    assert w1.dtype == torch.bfloat16, "w1 must be bfloat16"
    assert w2.dtype == torch.bfloat16, "w2 must be bfloat16"

    # Shape check
    assert hidden_states.ndim == 2, "hidden_states must be 2D"
    assert (
        hidden_states.shape[-1] == w1.shape[-2]
    ), f"hidden_states shape[-1] {hidden_states.shape} must be equal to w1 shape[-2] {w1.shape}"
    assert (
        w2.shape[-1] == w1.shape[1]
    ), f"w2 shape[-1] {w2.shape[-1]} must be equal to w1 shape[1] {w1.shape[1]}"

    # feature check
    assert inplace == False, "Inplace is not supported in new triton MoE kernel"

    M, K = hidden_states.shape
    E, _, N = w1.shape
    n_expts_act = routing_data.n_expts_act
    dtype = hidden_states.dtype

    if global_num_experts == -1:
        global_num_experts = E

    # consistent with default implementation
    intermediate_cache2 = torch.empty(
        (M * n_expts_act, N // 2), device="cuda", dtype=dtype
    )

    intermediate_cache1 = matmul_ogs(
        hidden_states,
        w1,
        None,
        routing_data,
        gather_indx=gather_indx,
        gammas=routing_data.gate_scal if apply_router_weight_on_input else None,
    )

    if activation == "silu":
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
    elif activation == "gelu":
        gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    intermediate_cache3 = matmul_ogs(
        intermediate_cache2,
        w2,
        None,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else routing_data.gate_scal,
    )

    return intermediate_cache3


def triton_kernel_moe_forward_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="forward_cuda_triton",
    op_func=triton_kernel_moe_forward,
    mutates_args=[],
    fake_impl=triton_kernel_moe_forward_fake,
)
