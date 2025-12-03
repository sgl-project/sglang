from typing import Optional, Tuple

import torch
from sgl_kernel.scalar_type import ScalarType
from sgl_kernel.utils import _get_cache_buf


def awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel.awq_dequantize.default(qweight, scales, qzeros)


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.int8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype):
    return torch.ops.sgl_kernel.fp8_blockwise_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
    )


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.fp8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def _bmm_fp8_internal(
    workspace_buffer: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
) -> None:
    cublas_handle = torch.cuda.current_blas_handle()
    torch.ops.sgl_kernel.bmm_fp8.default(
        A,
        B,
        D,
        A_scale,
        B_scale,
        workspace_buffer,
        cublas_handle,
    )


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )
    workspace_buffer = _get_cache_buf("bmm_fp8_workspace", 32 * 1024 * 1024, A.device)
    _bmm_fp8_internal(workspace_buffer, A, B, out, A_scale, B_scale)
    return out


def dsv3_fused_a_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if output is None:
        output = torch.empty(
            (mat_a.shape[0], mat_b.shape[1]),
            device=mat_a.device,
            dtype=mat_a.dtype,
        )
    torch.ops.sgl_kernel.dsv3_fused_a_gemm.default(output, mat_a, mat_b)
    return output


def sgl_per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
) -> None:
    if enable_v2 is None:
        from sglang.srt.utils import get_bool_env_var

        enable_v2 = get_bool_env_var("SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2")

    if enable_v2:
        return torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit_v2.default(
            input,
            output_q,
            output_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            fuse_silu_and_mul,
            masked_m,
        )

    assert not fuse_silu_and_mul, "only v2 support fuse_silu_and_mul"
    assert masked_m is None, "only v2 support masked_m"
    torch.ops.sgl_kernel.sgl_per_token_group_quant_8bit.default(
        input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
    )


# For legacy usage
sgl_per_token_group_quant_fp8 = sgl_per_token_group_quant_8bit
sgl_per_token_group_quant_int8 = sgl_per_token_group_quant_8bit


def sgl_per_tensor_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    is_static: bool,
) -> None:
    torch.ops.sgl_kernel.sgl_per_tensor_quant_fp8.default(
        input, output_q, output_s, is_static
    )


def sgl_per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.sgl_per_token_quant_fp8.default(input, output_q, output_s)


def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops.sgl_kernel.cutlass_scaled_fp4_mm.default(
        out, a, b, block_scale_a, block_scale_b, alpha
    )
    return out


def scaled_fp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in a sizzled layout.
    """
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Then, the scaling
    # factors in float8_e4m3fn are packed into an int32 for every 4 values.
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // block_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    # padded part should be zeroed out
    if rounded_n > scale_n:
        output_scale = torch.zeros(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )
    else:
        output_scale = torch.empty(
            (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
        )

    torch.ops.sgl_kernel.scaled_fp4_quant.default(
        output, input, output_scale, input_global_scale
    )
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


def qserve_w4a8_per_chn_gemm(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    wscales: torch.Tensor,
    ascales: torch.Tensor,
    w_szs: torch.Tensor,
    a_ssums: torch.Tensor,
    out_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out_feats is None:
        # NOTE(HandH1998): qserve_w4a8_per_chn_gemm only supports out dtype=torch.float16 now
        out_feats = torch.empty(
            (in_feats.shape[0], kernel.shape[0]),
            device=in_feats.device,
            dtype=torch.float16,
        )
    torch.ops.sgl_kernel.qserve_w4a8_per_chn_gemm.default(
        in_feats, kernel, wscales, ascales, w_szs, a_ssums, out_feats
    )
    return out_feats


def qserve_w4a8_per_group_gemm(
    in_feats: torch.Tensor,
    kernel: torch.Tensor,
    zeros: torch.Tensor,
    scales_i8: torch.Tensor,
    wscales: torch.Tensor,
    ascales: torch.Tensor,
    out_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out_feats is None:
        # NOTE(HandH1998): qserve_w4a8_per_group_gemm only supports out dtype=torch.float16 now
        out_feats = torch.empty(
            (in_feats.shape[0], kernel.shape[0]),
            device=in_feats.device,
            dtype=torch.float16,
        )
    torch.ops.sgl_kernel.qserve_w4a8_per_group_gemm.default(
        in_feats, kernel, zeros, scales_i8, wscales, ascales, out_feats
    )
    return out_feats


def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    output = torch.empty(
        hidden_states.shape[0],
        router_weights.shape[0],
        device=hidden_states.device,
        dtype=out_dtype,
    )
    torch.ops.sgl_kernel.dsv3_router_gemm(
        output,
        hidden_states,
        router_weights,
    )
    return output


def shuffle_rows(input_tensor, dst2src_map, output_tensor_shape):
    output_tensor = torch.empty(
        output_tensor_shape,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    torch.ops.sgl_kernel.shuffle_rows.default(input_tensor, dst2src_map, output_tensor)
    return output_tensor


def scaled_fp4_grouped_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Quantize input tensor to FP4 and return quantized tensor and scale, for
    grouped gemm inputs (e.g., grouped_gemm_nt_masked for flashinfer).
    Args:
        input: The input tensor to be quantized to FP4, with shape (l, m, k)
            l is number of groups, m is number of tokens per group, k is number of features.
        input_global_scale: A scalar scaling factor for the entire tensor, with
            shape (l,).
    Outputs:
        output: The quantized tensor in FP4, with shape (m, k // 2, l) but the physical
            layout is (l, m, k // 2). `// 2` is because two fp4 values are packed into
            an uint8.
        output_scales: The blockscale tensor in FP8-E4M3, with shape (32, 4, rm, 4, rk, l)
            but the physical layout is (l, rm, rk, 32, 4, 4).
    Note:
        For the shape of output_scales, `32 * 4 * rm` is a padded m to nearest multiple of 128.
        `4 * rk` is a padded `k // 16` to nearest multiple of 4. These layout constants are
        required by the NVIDIA Blackwell MMA operations.
    """
    device = input_tensor.device
    l, m, k = input_tensor.shape
    sf_vec_size = 16
    assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

    scale_k = k // sf_vec_size
    padded_k = (scale_k + (4 - 1)) // 4 * 4
    padded_k_int32 = padded_k // 4
    padded_m = (m + (128 - 1)) // 128 * 128
    output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
    output_scales = torch.empty(
        l, padded_m, padded_k_int32, device=device, dtype=torch.int32
    )

    torch.ops.sgl_kernel.silu_and_mul_scaled_fp4_experts_quant.default(
        output.view(l * m, k // 2),
        output_scales.view(l * padded_m, padded_k_int32),
        input_tensor.view(l * m, k),
        input_global_scale,
        mask,
        use_silu_and_mul=False,
    )
    # The physical layout of the output is (l, m, k // 2), but we want to return a
    # logical layout (m, k // 2, l) required by the flashinfer masked group gemm.
    output = output.permute(1, 2, 0)
    # The physical layout of the output scales is already swizzled as (l, rm, rk, 32, 4, 4), a
    # requirement for the flashinfer masked group gemm, where rm=m/128 and rk=k/4. The logic
    # layout is (32, 4, rm, 4, rk, l).
    output_scales = output_scales.view(torch.float8_e4m3fn).view(
        l, padded_m // 128, padded_k // 4, 32, 4, 4
    )
    output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
    return output, output_scales


def silu_and_mul_scaled_fp4_grouped_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Quantize input tensor to FP4 and return quantized tensor and scale, for
    grouped gemm inputs (e.g., grouped_gemm_nt_masked for flashinfer).
    Args:
        input: The input tensor to be quantized to FP4, with shape (l, m, k * 2)
            l is number of groups, m is number of tokens per group, k is number of features.
        input_global_scale: A scalar scaling factor for the entire tensor, with
            shape (l,).
        mask: The mask tensor, with shape (l,)
    Outputs:
        output: The quantized tensor in FP4, with shape (m, k // 2, l) but the physical
            layout is (l, m, k // 2). `// 2` is because two fp4 values are packed into
            an uint8.
        output_scales: The blockscale tensor in FP8-E4M3, with shape (32, 4, rm, 4, rk, l)
            but the physical layout is (l, rm, rk, 32, 4, 4).
    Note:
        For the shape of output_scales, `32 * 4 * rm` is a padded m to nearest multiple of 128.
        `4 * rk` is a padded `k // 16` to nearest multiple of 4. These layout constants are
        required by the NVIDIA Blackwell MMA operations.
    """
    device = input_tensor.device
    l, m, k_by_2 = input_tensor.shape
    k = k_by_2 // 2
    sf_vec_size = 16
    assert k % sf_vec_size == 0, f"k must be multiple of 16, but got {k}."

    scale_k = k // sf_vec_size
    padded_k = (scale_k + (4 - 1)) // 4 * 4
    padded_k_int32 = padded_k // 4
    padded_m = (m + (128 - 1)) // 128 * 128
    output = torch.empty(l, m, k // 2, device=device, dtype=torch.uint8)
    output_scales = torch.empty(
        l, padded_m, padded_k_int32, device=device, dtype=torch.int32
    )

    torch.ops.sgl_kernel.silu_and_mul_scaled_fp4_experts_quant.default(
        output.view(l * m, k // 2),
        output_scales.view(l * padded_m, padded_k_int32),
        input_tensor.view(l * m, k_by_2),
        input_global_scale,
        mask,
        use_silu_and_mul=True,
    )
    # The physical layout of the output is (l, m, k // 2), but we want to return a
    # logical layout (m, k // 2, l) required by the flashinfer masked group gemm.
    output = output.permute(1, 2, 0)
    # The physical layout of the output scales is already swizzled as (l, rm, rk, 32, 4, 4), a
    # requirement for the flashinfer masked group gemm, where rm=m/128 and rk=k/4. The logic
    # layout is (32, 4, rm, 4, rk, l).
    output_scales = output_scales.view(torch.float8_e4m3fn).view(
        l, padded_m // 128, padded_k // 4, 32, 4, 4
    )
    output_scales = output_scales.permute(3, 4, 1, 5, 2, 0)
    return output, output_scales


def scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int,
    expert_map: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale, for
    packed MoE Inputs.
    Args:
        input: The input tensor to be quantized to FP4
        expert_map: The expert map tensor
        input_global_scale: A scalar scaling factor for the entire tensor.
        expert_offsets: The expert offsets tensor
        blockscale_offsets: The blockscale offsets tensor
    Outputs:
        output: The quantized tensor in FP4
        output_scales: The blockscale tensor in FP8-E4M3
    """
    assert (
        input_tensor.ndim == 2
    ), f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    if expert_map is not None:
        (m, k) = input_tensor.shape
        output_tensor_shape = (m * topk, k)
        input_tensor = shuffle_rows(input_tensor, expert_map, output_tensor_shape)
    m_numtopk, k = input_tensor.shape
    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE Expert Quantization. This is used to prevent the kernel
    # from running out of memory. This value can also be increased to support
    # larger models.
    import os

    MAX_TOKENS_PER_EXPERT = int(os.environ.get("MODELOPT_MAX_TOKENS_PER_EXPERT", 65536))
    assert m_numtopk <= MAX_TOKENS_PER_EXPERT * topk, (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT("
        f"{MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        f" MODELOPT_MAX_TOKENS_PER_EXPERT to set this value."
    )
    scales_k = k // 16
    padded_k = (scales_k + (4 - 1)) // 4

    # output is uint8 and packed fp4 values
    output = torch.empty(
        m_numtopk, k // 2, device=input_tensor.device, dtype=torch.uint8
    )
    # padded part should be zeroed out
    if padded_k > scales_k:
        output_scales = torch.zeros(
            MAX_TOKENS_PER_EXPERT * topk,
            padded_k,
            dtype=torch.int32,
            device=input_tensor.device,
        )
    else:
        output_scales = torch.empty(
            MAX_TOKENS_PER_EXPERT * topk,
            padded_k,
            dtype=torch.int32,
            device=input_tensor.device,
        )
    torch.ops.sgl_kernel.scaled_fp4_experts_quant.default(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    output_scales = output_scales.view(torch.float8_e4m3fn)
    return output, output_scales


# GPTQ kernels
def gptq_marlin_gemm(
    a: torch.Tensor,
    c: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.gptq_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_shuffle: bool,
    bit: int,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.gptq_gemm(
        a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit
    )


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None:
    torch.torch.ops.sgl_kernel.gptq_shuffle(q_weight, q_perm, bit)
