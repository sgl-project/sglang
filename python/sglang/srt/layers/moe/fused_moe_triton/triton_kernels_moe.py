# Adapted from https://github.com/vllm-project/vllm/pull/18595/files#diff-f426a6de78c82ffec568eff6811bfbf0043dab5f87f1a8c0cffdbdcb8a81e035

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel import gelu_and_mul, silu_and_mul
from triton_kernels.matmul_ogs import (
    FlexCtx,
    FnSpecs,
    FusedActivation,
    PrecisionConfig,
    matmul_ogs,
)
from triton_kernels.numerics import InFlexData
from triton_kernels.routing import GatherIndx, RoutingData, ScatterIndx
from triton_kernels.swiglu import swiglu_fn

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import TopKOutput


def quantize(w, dtype, dev, **opt):
    if dtype == "bf16":
        return w.to(torch.bfloat16), InFlexData()


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: TopKOutput,
    moe_runner_config: MoeRunnerConfig,
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

    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert TopKOutputChecker.format_is_triton_kernel(topk_output)

    routing_data, gather_idx, scatter_idx = topk_output

    return triton_kernel_fused_experts(
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        inplace=False,  # triton kernel doesn't support inplace
        activation=moe_runner_config.activation,
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

    assert use_fp8_w8a8 is False, "use_fp8_w8a8 is not supported"
    assert per_channel_quant is False, "per_channel_quant is not supported"
    assert expert_map is None, "expert_map is not supported"
    assert w1_scale is None, "w1_scale is not supported"
    assert w2_scale is None, "w2_scale is not supported"
    assert a1_scale is None, "a1_scale is not supported"
    assert a2_scale is None, "a2_scale is not supported"
    assert block_shape is None, "block_shape is not supported"

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
    assert inplace is False, "Inplace is not supported in new triton MoE kernel"

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


def triton_kernel_moe_with_bias_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_pcg,
    b1: torch.Tensor,
    w2: torch.Tensor,
    w2_pcg,
    b2: torch.Tensor,
    topk_output: TopKOutput,
    moe_runner_config: MoeRunnerConfig,
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
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert TopKOutputChecker.format_is_triton_kernel(topk_output)

    routing_data, gather_idx, scatter_idx = topk_output

    return triton_kernel_fused_experts_with_bias(
        hidden_states,
        w1=w1,
        w1_pcg=w1_pcg,
        b1=b1,
        w2=w2,
        w2_pcg=w2_pcg,
        b2=b2,
        routing_data=routing_data,
        gather_indx=gather_idx,
        scatter_indx=scatter_idx,
        inplace=False,  # triton kernel doesn't support inplace
        activation=moe_runner_config.activation,
        use_fp8_w8a8=use_fp8_w8a8,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        gemm1_alpha=moe_runner_config.gemm1_alpha,
        gemm1_clamp_limit=moe_runner_config.gemm1_clamp_limit,
    )


def triton_kernel_fused_experts_with_bias(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_pcg,
    b1: torch.Tensor,
    w2: torch.Tensor,
    w2_pcg,
    b2: torch.Tensor,
    routing_data: RoutingData,
    gather_indx: GatherIndx,
    scatter_indx: ScatterIndx,
    inplace: bool = False,
    activation: str = "silu",
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
) -> torch.Tensor:
    assert use_fp8_w8a8 is False, "use_fp8_w8a8 is not supported"
    assert per_channel_quant is False, "per_channel_quant is not supported"
    assert expert_map is None, "expert_map is not supported"
    assert w1_scale is None, "w1_scale is not supported"
    assert w2_scale is None, "w2_scale is not supported"
    assert a1_scale is None, "a1_scale is not supported"
    assert a2_scale is None, "a2_scale is not supported"
    assert block_shape is None, "block_shape is not supported"

    # type check
    assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
    for w in (w1, w2):
        # TODO assert bf16 or mxfp4
        # assert (w.dtype == torch.bfloat16) or check-is-mxfp4, f"w must be bfloat16 or mxfp4 {w1.dtype=}"
        pass

    # Shape check
    assert hidden_states.ndim == 2, "hidden_states must be 2D"
    assert (
        hidden_states.shape[-1] == w1.shape[-2]
    ), f"hidden_states shape[-1] {hidden_states.shape} must be equal to w1 shape[-2] {w1.shape}"
    assert (
        w2.shape[-1] == w1.shape[1]
    ), f"w2 shape[-1] {w2.shape[-1]} must be equal to w1 shape[1] {w1.shape[1]}"

    # feature check
    assert inplace is False, "Inplace is not supported in new triton MoE kernel"

    E, _, _ = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    # TODO maybe completely remove this branch
    if w1.dtype == torch.bfloat16:
        device = "cuda"
        optg = dict()
        w1, w1_flex = quantize(w1, "bf16", device, **optg)
        w1_pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w1_flex))

        w2, w2_flex = quantize(w2, "bf16", device, **optg)
        w2_pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex))

    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")),
        (gemm1_alpha, gemm1_clamp_limit),
        2,
    )

    intermediate_cache = matmul_ogs(
        hidden_states,
        w1,
        b1,
        routing_data,
        gather_indx=gather_indx,
        precision_config=w1_pcg,
        gammas=None,
        fused_activation=act,
    )

    return matmul_ogs(
        intermediate_cache,
        w2,
        b2,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=w2_pcg,
        gammas=routing_data.gate_scal,
    )
