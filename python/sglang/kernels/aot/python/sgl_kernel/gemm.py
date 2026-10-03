from typing import Optional

import torch


def gptq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
) -> torch.Tensor:
    bits = 32 * qzeros.size(1) // qweight.size(1)
    if bits not in [2, 4, 8]:
        raise NotImplementedError("Only 2,4,8 bits are supported.")
    group_size = 32 * qweight.size(0) // qzeros.size(0) // bits

    wf = (
        torch.tensor(list(range(0, 32, bits)), dtype=torch.int32)
        .unsqueeze(0)
        .to(qweight.device)
    )

    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
    ).to(torch.int8)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)
    zeros = zeros + 1
    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    scales = scales.reshape(-1, 1, scales.shape[-1])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
    ).to(torch.int8)
    weight = torch.bitwise_and(weight, (2**bits) - 1)
    weight = weight.reshape(-1, group_size, weight.shape[2])
    weight = scales * (weight - zeros)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    return weight


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
