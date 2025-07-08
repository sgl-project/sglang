from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from sgl_kernel import sgl_per_tensor_quant_fp8

IS_TRITON_KERNELS_AVAILABLE = False
try:
    import os
    import sys
    llm_root = os.getenv('SGL_ROOT')
    if llm_root:
        # On CI, we use SGL_ROOT to locate the 3rdparty directory.
        triton_path = os.path.join(llm_root, '3rdparty', 'triton', 'python',
                                   'triton_kernels')
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        triton_path = os.path.join(current_dir, '..', '..', '..', '..', '..', '..',
                                   '3rdparty', 'triton', 'python',
                                   'triton_kernels')
    triton_path = os.path.abspath(triton_path)
    if os.path.exists(triton_path) and triton_path not in sys.path:
        sys.path.insert(0, triton_path)
    import triton
    import triton_kernels
    import triton_kernels.swiglu
    from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig, matmul_ogs, MicroscalingCtx, SwizzlingType, InFlexData
    from triton_kernels.routing import routing
    from triton_kernels.numerics_details.mxfp import (SWIZZLE_ALIGN_INNER,
                                                      SWIZZLE_SIZE_INNER,
                                                      SWIZZLE_SIZE_OUTER,
                                                      SwizzlingType)
    IS_TRITON_KERNELS_AVAILABLE = True
except ImportError:
    print("triton_kernels not available")

def fused_experts_oai(
    hidden_states: torch.Tensor,  # (num_tokens, hidden_dim)
    w13: torch.Tensor,  # (num_experts, hidden_dim, intermediate_dim * 2)
    w2: torch.Tensor,  # (num_experts, intermediate_dim, hidden_dim)
    expert_logits: torch.Tensor,  # (num_tokens, num_experts)
    top_k: int,
    inplace: bool,
    activation: str,  # "swiglu"
    apply_router_weight_on_input: bool,
    no_combine: bool,
    routed_scaling_factor: Optional[float],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    swiglu_alpha: torch.Tensor,
    swiglu_beta: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:

    gemm1_weights = w13
    gemm2_weights = w2

    num_experts = expert_logits.shape[1]
    if num_experts > 1:
        rdata, gather_indx, scatter_indx = routing(expert_logits, top_k)
    else:
        rdata, gather_indx, scatter_indx = None, None, None

    pc1 = PrecisionConfig(flex_ctx=FlexCtx(),
                        allow_tf32=False,
                        out_dtype=dtype)
    gemm1_output = matmul_ogs(
        hidden_states,
        gemm1_weights,
        w1_bias,
        rdata,
        gather_indx=gather_indx,
        precision_config=pc1)

    pcs = triton_kernels.swiglu.PrecisionConfig(limit=None)

    act_out = triton_kernels.swiglu.swiglu(
        gemm1_output,
        alpha=swiglu_alpha,
        beta=swiglu_beta,
        precision_config=pcs,
        routing_data=rdata)


    pc2 = PrecisionConfig(flex_ctx=FlexCtx(),
                            allow_tf32=False,
                            out_dtype=dtype)

    gemm2_output = matmul_ogs(
        act_out,
        gemm2_weights,
        w2_bias,
        rdata,
        scatter_indx=scatter_indx,
        precision_config=pc2,
        gammas=rdata.gate_scal if rdata else None)
    return gemm2_output

def quantize_fp8_per_tensor(
                input_q: torch.Tensor, 
                scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_type_ = torch.float8_e4m3fn
    output_q = torch.empty_like(input_q, dtype=fp8_type_, device=input_q.device)
    is_static = True
    if scale is None:
        scale = torch.tensor(1.0, dtype=torch.float32, device=input_q.device)
        is_static = False
    sgl_per_tensor_quant_fp8(input_q, output_q, scale, is_static)
    return output_q, scale

def swizzle(x: torch.Tensor):
    assert len(x.shape) == 3
    x = x.transpose(1, 2)  # From kernel order to swizzle work order
    original_shape = x.shape
    x = x.reshape(x.shape[0], x.shape[1] // SWIZZLE_SIZE_OUTER,
                  SWIZZLE_SIZE_OUTER // 32, 32,
                  x.shape[2] // SWIZZLE_SIZE_INNER, SWIZZLE_SIZE_INNER)
    x = x.transpose(
        -2, -4).contiguous()  # Swap the swizzle inner and outer dimensions
    x = x.reshape(original_shape)
    x = x.transpose(
        1, 2
    )  # Back to kernel order. Don't use .contiguous() here, it will break the kernel's assumption
    return x

def check_and_swizzle(weight_tensor: torch.Tensor,
                      scale_tensor: torch.Tensor) -> torch.Tensor:
    num_experts, dim1, dim2 = weight_tensor.shape
    dim1 *= 2  # We pack two mxfp4 values into one uint8, so dim1 being even is always required and can't be checked here
    assert dim1 % 32 == 0  # Quant group requirements. Should work without this but let's be safe
    assert (num_experts, dim1 // 32,
            dim2) == scale_tensor.shape, "scales shape mismatch"
    assert dim1 // 32 % SWIZZLE_ALIGN_INNER == 0, "Swizzle requirement not met"
    assert dim2 % SWIZZLE_SIZE_OUTER == 0, "Swizzle requirement not met"
    return swizzle(scale_tensor)

def fused_experts_mxfp4_oai(
    hidden_states: torch.Tensor,  # (num_tokens, hidden_dim)
    w13: torch.Tensor,  # (num_experts, hidden_dim, intermediate_dim * 2)
    w2: torch.Tensor,  # (num_experts, intermediate_dim, hidden_dim)
    expert_logits: torch.Tensor,  # (num_tokens, num_experts)
    top_k: int,
    fc31_input_dequant: torch.Tensor,
    fc2_input_dequant: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    inplace: bool,
    activation: str,  # "swiglu"
    apply_router_weight_on_input: bool,
    no_combine: bool,
    routed_scaling_factor: Optional[float],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    swiglu_alpha: torch.Tensor,
    swiglu_beta: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    activation_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    if activation_dtype == torch.float8_e4m3fn:
        hidden_states, hidden_states_scale = quantize_fp8_per_tensor(hidden_states, fc31_input_dequant)
    else:
        hidden_states = hidden_states
    gemm1_weights = w13
    gemm1_scales = w13_scale
    gemm2_weights = w2
    gemm2_scales = w2_scale
    top_k = top_k

    num_experts = expert_logits.shape[1]
    if num_experts > 1:
        rdata, gather_indx, scatter_indx = routing(expert_logits, top_k)
    else:
        rdata, gather_indx, scatter_indx = None, None, None
    
    mx_ctx_1 = MicroscalingCtx(
            weight_scale=gemm1_scales,
            swizzle_value=None,
            swizzle_scale=SwizzlingType.
            BLACKWELL,  #TODO: Integrate other types recently added to Triton TOT
            actual_weight_scale_shape=gemm1_scales.shape)
    if activation_dtype == torch.float8_e4m3fn:
        flex_ctx_1 = FlexCtx(
            lhs_data=InFlexData(scale=hidden_states_scale), )
    else:
        flex_ctx_1 = FlexCtx()
    pc1 = PrecisionConfig(mx_ctx=mx_ctx_1,
                          flex_ctx=flex_ctx_1,
                          allow_tf32=False,
                          out_dtype=dtype)

    # Call the Triton kernel, which also does permutation
    gemm1_output = matmul_ogs(
            hidden_states,
            gemm1_weights,
            w1_bias,  # Bias
            rdata,
            gather_indx=gather_indx,
            precision_config=pc1)
    pcs = triton_kernels.swiglu.PrecisionConfig(limit=None)
    # Call the Triton activation kernel
    act_out = triton_kernels.swiglu.swiglu(
        gemm1_output,
        alpha=swiglu_alpha,
        beta=swiglu_beta,
        precision_config=pcs,
        routing_data=rdata)
    if activation_dtype == torch.float8_e4m3fn:
        # Quantize the activation output manually since the Triton activation kernel doesn't support bf16 in fp8 out
        act_out, act_scale = quantize_fp8_per_tensor(act_out, fc2_input_dequant)
    mx_ctx_2 = MicroscalingCtx(
            weight_scale=gemm2_scales,
            swizzle_value=None,
            swizzle_scale=SwizzlingType.
            BLACKWELL,  #TODO: Integrate other types recently added to Triton TOT
            actual_weight_scale_shape=gemm2_scales.shape)
    if activation_dtype == torch.float8_e4m3fn:
        flex_ctx_2 = FlexCtx(lhs_data=InFlexData(scale=act_scale), )
    else:
        flex_ctx_2 = FlexCtx()
    pc2 = PrecisionConfig(mx_ctx=mx_ctx_2,
                            flex_ctx=flex_ctx_2,
                            allow_tf32=False,
                            out_dtype=dtype)

    # Call the Triton kernel, which also does finalization
    gemm2_output = matmul_ogs(
        act_out,
        gemm2_weights,
        w2_bias,  # Bias
        rdata,
        scatter_indx=scatter_indx,
        precision_config=pc2,
        gammas=rdata.gate_scal if rdata else None)
    return gemm2_output
