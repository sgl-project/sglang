from typing import Optional, Tuple

import torch
import torch.nn.functional as F
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
                                                      SwizzlingType,
                                                      perm_tensor_from_contig, perm_tuple_from_contig,
                                                      swizzle_mx_scale_bw, swizzle_mxfp4_scale_hopper,
                                                      swizzle_mxfp4_value_hopper)
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch, upcast_from_mxfp_torch
    IS_TRITON_KERNELS_AVAILABLE = True
except ImportError:
    print("triton_kernels not available")

def shuffle_for_activation_kernel(weight: torch.Tensor) -> torch.Tensor:
    temp_weight = weight.clone()
    last_dim = weight.shape[-1]
    if weight.dim() == 3:
        weight[:, :, 1::2] = temp_weight[:, :, last_dim // 2:]
        weight[:, :, 0::2] = temp_weight[:, :, 0:last_dim // 2]
    elif weight.dim() == 2:
        weight[:, 1::2] = temp_weight[:, last_dim // 2:]
        weight[:, 0::2] = temp_weight[:, 0:last_dim // 2]
    return weight

def quantize_to_mxfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = tensor.transpose(1, 2).contiguous()
    tensor_fp4, tensor_scales = downcast_to_mxfp_torch(tensor, torch.uint8, axis=1)
    tensor_fp4 = tensor_fp4.transpose(1, 2).contiguous()
    tensor_scales = tensor_scales.transpose(1, 2).contiguous()
    return tensor_fp4, tensor_scales

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

def get_swizzle_type(activation_type):
    assert activation_type in [torch.float8_e4m3fn, torch.bfloat16]
    assert torch.cuda.get_device_capability()[0] >= 9
    if torch.cuda.get_device_capability()[0] < 10:
        if activation_type == torch.float8_e4m3fn:
            swizzle_mx_value = None
            swizzle_mx_scale = SwizzlingType.BLACKWELL
        else:
            swizzle_mx_value = SwizzlingType.HOPPER
            swizzle_mx_scale = SwizzlingType.HOPPER
    else:
        swizzle_mx_value = None
        swizzle_mx_scale = SwizzlingType.BLACKWELL
    return swizzle_mx_value, swizzle_mx_scale

def swizzle_weight_and_scale(weight_tensor: torch.Tensor,
                             scale_tensor: torch.Tensor,
                             swizzle_value: SwizzlingType,
                             swizzle_scale: SwizzlingType) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    # switch to swizzle shape
    quant_tensor = weight_tensor.transpose(1, 2).contiguous()
    scale = scale_tensor.transpose(1, 2).contiguous()
    # Swizzling
    if swizzle_value == SwizzlingType.HOPPER:
        quant_tensor = swizzle_mxfp4_value_hopper(quant_tensor,
                                                  op_idx=0,
                                                  mma_version=3)
    assert quant_tensor.is_contiguous()
    axis = 1
    swizzle_axis = 2 if swizzle_scale else None
    quant_tensor = perm_tensor_from_contig(quant_tensor, axis, swizzle_axis)
    orig_scale_shape = scale.shape
    if swizzle_scale == SwizzlingType.BLACKWELL:
        scale = swizzle_mx_scale_bw(scale, allow_pad=True)
    elif swizzle_scale == SwizzlingType.HOPPER:
        scale = swizzle_mxfp4_scale_hopper(scale, num_warps=8)
    assert scale.is_contiguous()
    scale = perm_tensor_from_contig(scale, axis, swizzle_axis)
    actual_scale_shape = perm_tuple_from_contig(orig_scale_shape, axis,
                                                swizzle_axis)
    return quant_tensor, scale, actual_scale_shape

def pad_weight_and_scale_on_hopper(weight: torch.Tensor, 
                                   scale: Optional[torch.Tensor], 
                                   swizzle_scale: SwizzlingType) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if swizzle_scale == SwizzlingType.HOPPER:
        assert weight.dim() in [2,
                                3], "Weight should be 2D or 3D tensor"
        out_dim = weight.shape[-1]
        assert scale is None or scale.shape[
            -1] == out_dim, "Out dim of weight and scale should match"
        pad_size = (256 - out_dim % 256) % 256
        weight = F.pad(
            weight,
            (0, pad_size
             )).contiguous()  # Pad the last dimension on right side
        if scale is not None:
            scale = F.pad(scale, (0, pad_size)).contiguous()
    return (weight, scale) if scale is not None else weight

def maybe_remove_padding(gemm_output: torch.Tensor, 
                         expected_size: int,
                         swizzle_scale: SwizzlingType):
    assert gemm_output.dim() == 2
    if gemm_output.shape[-1] != expected_size:
        assert swizzle_scale == SwizzlingType.HOPPER, "Only Hopper style swizzle can have padding"
        assert gemm_output.shape[
            -1] % 256 == 0, "The padding is not done correctly"
        gemm_output = gemm_output[:, :expected_size]
    return gemm_output

def fused_experts_oai(
    hidden_states: torch.Tensor,  # (num_tokens, hidden_dim)
    w13: torch.Tensor,  # (num_experts, hidden_dim, intermediate_dim * 2)
    w2: torch.Tensor,  # (num_experts, intermediate_dim, hidden_dim)
    expert_logits: torch.Tensor,  # (num_tokens, num_experts)
    top_k: int,
    activation: str,  # "swiglu"
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    swiglu_alpha: torch.Tensor,
    swiglu_beta: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    clamp_limit: Optional[float] = None) -> torch.Tensor:

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

    pcs = triton_kernels.swiglu.PrecisionConfig(limit=clamp_limit)

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

def fused_experts_mxfp4_oai(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    expert_logits: torch.Tensor,
    top_k: int,
    fc31_input_dequant: torch.Tensor,
    fc2_input_dequant: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    activation: str,  # "swiglu"
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    swiglu_alpha: torch.Tensor,
    swiglu_beta: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    activation_dtype: torch.dtype = torch.float8_e4m3fn,
    swizzle_value: Optional[SwizzlingType] = None,
    swizzle_scale: Optional[SwizzlingType] = None,
    actual_w13_scale_shape: Optional[torch.Size] = None,
    actual_w2_scale_shape: Optional[torch.Size] = None,
    intermediate_size: int = 0,
    hidden_size: int = 0,
    clamp_limit: Optional[float] = None,
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
            swizzle_value=swizzle_value,
            swizzle_scale=swizzle_scale,
            actual_weight_scale_shape=actual_w13_scale_shape)
    if activation_dtype == torch.float8_e4m3fn:
        flex_ctx_1 = FlexCtx(
            lhs_data=InFlexData(scale=hidden_states_scale), )
    else:
        flex_ctx_1 = FlexCtx()
    pc1 = PrecisionConfig(mx_ctx=mx_ctx_1,
                          flex_ctx=flex_ctx_1,
                          allow_tf32=False,
                          out_dtype=dtype)

    gemm1_output = matmul_ogs(
            hidden_states,
            gemm1_weights,
            w1_bias,
            rdata,
            gather_indx=gather_indx,
            precision_config=pc1)

    gemm1_output = maybe_remove_padding(gemm1_output,
                                        intermediate_size * 2,
                                        swizzle_scale).contiguous()

    pcs = triton_kernels.swiglu.PrecisionConfig(limit=clamp_limit)
    act_out = triton_kernels.swiglu.swiglu(
        gemm1_output,
        alpha=swiglu_alpha,
        beta=swiglu_beta,
        precision_config=pcs,
        routing_data=rdata)
    if activation_dtype == torch.float8_e4m3fn:
        act_out, act_scale = quantize_fp8_per_tensor(act_out, fc2_input_dequant)
    mx_ctx_2 = MicroscalingCtx(
            weight_scale=gemm2_scales,
            swizzle_value=swizzle_value,
            swizzle_scale=swizzle_scale,
            actual_weight_scale_shape=actual_w2_scale_shape)
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
    gemm2_output = maybe_remove_padding(gemm2_output, 
                                        hidden_size,
                                        swizzle_scale)
    return gemm2_output
