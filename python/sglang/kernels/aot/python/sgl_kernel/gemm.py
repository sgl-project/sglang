from typing import Optional, Tuple

import torch


def awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel.awq_dequantize.default(qweight, scales, qzeros)


def convrot_int8_supported_sm_versions() -> list[int]:
    """Compute capabilities (major * 10 + minor) the convrot_int8_* ops carry code for."""
    return list(torch.ops.sgl_kernel.convrot_int8_supported_sm_versions.default())


def convrot_rotate_quantize_activation(
    x: torch.Tensor, group_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group Hadamard rotation + per-row INT8 quant of BF16 [M, K].

    Returns (x_q int8 [M, K], x_scale float32 [M]). Also the offline transform
    for a [N, K] weight, yielding (weight_q, weight_scale).
    """
    return torch.ops.sgl_kernel.convrot_rotate_quantize_activation.default(
        x, group_size
    )


def convrot_int8_fused_linear(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 256,
) -> torch.Tensor:
    """BF16 [M, K] x int8 [N, K] -> BF16 [M, N]; x is rotated and quantized in-kernel."""
    return torch.ops.sgl_kernel.convrot_int8_fused_linear.default(
        x, weight_q, weight_scale, bias, group_size
    )


def convrot_int8_fused_linear_gelu_input(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 256,
) -> torch.Tensor:
    """convrot_int8_fused_linear(F.gelu(x, approximate="tanh"), ...), bitwise, in one op."""
    return torch.ops.sgl_kernel.convrot_int8_fused_linear_gelu_input.default(
        x, weight_q, weight_scale, bias, group_size
    )


def convrot_int8_fused_linear_out(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    group_size: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """convrot_int8_fused_linear into a contiguous BF16 [M, N] `out`; returns `out`."""
    return torch.ops.sgl_kernel.convrot_int8_fused_linear_out.default(
        x, weight_q, weight_scale, bias, group_size, out
    )


def convrot_int8_linear_prequant(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 256,
) -> torch.Tensor:
    """GEMM + dequant on (xq int8 [M, K], xs float32 [M]) from
    convrot_rotate_quantize_activation; bitwise equal to the fused op."""
    return torch.ops.sgl_kernel.convrot_int8_linear_prequant.default(
        xq, xs, weight_q, weight_scale, bias, group_size
    )


def convrot_int8_linear_prequant_out(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    group_size: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """convrot_int8_linear_prequant into a contiguous BF16 [M, N] `out`; returns `out`."""
    return torch.ops.sgl_kernel.convrot_int8_linear_prequant_out.default(
        xq, xs, weight_q, weight_scale, bias, group_size, out
    )


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.int8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
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
    _V2_KERNEL_SUPPORTED_GROUP_SIZES = [16, 32, 64, 128]
    if enable_v2 is None:
        enable_v2 = group_size in _V2_KERNEL_SUPPORTED_GROUP_SIZES

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


def sgl_per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.sgl_per_token_quant_fp8.default(input, output_q, output_s)


def shuffle_rows(input_tensor, dst2src_map, output_tensor_shape):
    output_tensor = torch.empty(
        output_tensor_shape,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    torch.ops.sgl_kernel.shuffle_rows.default(input_tensor, dst2src_map, output_tensor)
    return output_tensor


# GPTQ kernels
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
