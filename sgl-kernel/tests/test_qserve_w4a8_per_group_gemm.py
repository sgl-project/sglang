import pytest
import torch
from sgl_kernel import qserve_w4a8_per_group_gemm


# Adapted from https://github.com/mit-han-lab/omniserve/blob/main/omniserve/modeling/layers/quantized_linear/w4a8_linear.py
def convert_to_qserve_format(qweight, chn_scale, scale_i8, zero_i8, group_size):
    assert qweight.min() >= 0 and qweight.max() <= 15, "Quantized weight out of range"
    in_features = qweight.shape[1]
    out_features = qweight.shape[0]
    assert in_features % 32 == 0, "Input features must be divisible by 32"
    assert out_features % 32 == 0, "Output features must be divisible by 32"
    assert group_size == 128, "Group size must be 128"
    assert (
        in_features % group_size == 0
    ), "Input features must be divisible by group_size"

    # ---- Repack the weight ---- #
    # pack to M // 32, K // 32, (8, 4), ([2], 2, 2, 4)
    qweight_unpack_reorder = (
        qweight.reshape(
            out_features // 32,
            2,
            2,
            8,
            in_features // 32,
            2,
            4,
            4,
        )
        .permute(0, 4, 3, 6, 1, 5, 2, 7)
        .contiguous()
    )
    qweight_unpack_reorder = (
        qweight_unpack_reorder.permute(0, 1, 2, 3, 5, 6, 7, 4)
        .contiguous()
        .to(torch.int8)
    )
    # B_fp16_reorder = B_fp16_reorder[:, :, :, :, :, :, [3, 2, 1, 0]].contiguous()
    # [16, 0, 17, 1, ...]
    qweigth_unpack_repacked = (
        qweight_unpack_reorder[..., 1] << 4
    ) + qweight_unpack_reorder[..., 0]
    qweigth_unpack_repacked = qweigth_unpack_repacked.reshape(
        out_features // 32, in_features // 32, 32, 16
    )
    qweigth_unpack_repacked = qweigth_unpack_repacked.reshape(
        out_features, in_features // 2
    )

    # ---- Pack the scales ---- #
    chn_scale = chn_scale.reshape(out_features)

    scale_i8 = (
        scale_i8.reshape(out_features, in_features // group_size)
        .transpose(0, 1)
        .contiguous()
    )
    scale_i8 = scale_i8.reshape(in_features // group_size, out_features // 32, 32)
    scale_i8 = (
        scale_i8.reshape(in_features // group_size, out_features // 32, 4, 8)
        .transpose(-2, -1)
        .contiguous()
    )
    scale_i8 = scale_i8.reshape(in_features // group_size, out_features).contiguous()

    # ---- Pack the zeros ---- #
    zero_i8 = -zero_i8
    # zero_i8 = zero_i8.int()  # convert to 2-complement

    zero_i8 = (
        zero_i8.reshape(out_features, in_features // group_size)
        .transpose(0, 1)
        .contiguous()
    )
    zero_i8 = zero_i8.reshape(in_features // group_size, out_features // 32, 32)
    # for the last dimension, organize as 0, 8, 16, 24, 1, 9, 17, 25, ... following the requirement of tensor core gemm
    zero_i8 = (
        zero_i8.reshape(in_features // group_size, out_features // 32, 4, 8)
        .transpose(-2, -1)
        .contiguous()
    )
    zero_i8 = (
        zero_i8.reshape(in_features // group_size, out_features).contiguous() * scale_i8
    )

    return qweigth_unpack_repacked, chn_scale, scale_i8, zero_i8


# Progressive Group INT4 Quantization
def progressive_group_quantize_tensor(tensor, group_size):
    assert group_size == 128, "Group size must be 128"
    assert (
        tensor.shape[-1] % group_size == 0
    ), "Input features must be divisible by group_size"
    # Channel scale
    # NOTE(HandH1998): use protective quantization range
    chn_scale = tensor.abs().max(dim=-1, keepdim=True)[0] / 119
    tensor_i8 = torch.clamp(torch.round(tensor / chn_scale), -119, 119)

    # Group scale
    tensor_i8 = tensor_i8.reshape(-1, group_size)
    tensor_i8_min = tensor_i8.min(dim=-1, keepdim=True)[0]
    tensor_i8_max = tensor_i8.max(dim=-1, keepdim=True)[0]
    q_min = 0
    q_max = 15
    scale_i8 = torch.round((tensor_i8_max - tensor_i8_min) / (q_max - q_min))
    zero_i8 = q_min - torch.round(tensor_i8_min / scale_i8)
    tensor_q = (
        torch.clamp(torch.round(tensor_i8 / scale_i8) + zero_i8, q_min, q_max)
        .reshape(tensor.shape[0], -1)
        .to(torch.int8)
    )
    return (
        tensor_q,
        chn_scale.to(torch.float16),
        scale_i8.reshape(tensor.shape[0], -1).to(torch.int8),
        zero_i8.reshape(tensor.shape[0], -1).to(torch.int8),
    )


# INT8 Quantization
def sym_quantize_tensor(tensor):
    tensor_scale = tensor.abs().max(dim=-1, keepdim=True)[0] / 127
    tensor_q = torch.clamp(torch.round(tensor / tensor_scale), -128, 127).to(torch.int8)
    return tensor_q, tensor_scale.to(torch.float16)


def torch_w4a8_per_group_gemm(
    a, b, a_scale, b_chn_scale, b_scale_i8, b_zero_i8, group_size, out_dtype
):
    assert group_size == 128, "Group size must be 128"
    b_dq = (
        b.reshape(-1, group_size).to(torch.float32)
        - b_zero_i8.reshape(-1, 1).to(torch.float32)
    ) * b_scale_i8.reshape(-1, 1).to(torch.float32)
    b_dq = b_dq.reshape(b.shape[0], b.shape[1])
    o = torch.matmul(a.to(torch.float32), b_dq.t())
    o = o * a_scale.view(-1, 1) * b_chn_scale.view(1, -1)
    return o.to(out_dtype)


def _test_accuracy_once(M, N, K, group_size, out_dtype, device):
    # to avoid overflow, multiply 0.01
    a = torch.randn((M, K), device=device, dtype=torch.float32) * 0.01
    b = torch.randn((N, K), device=device, dtype=torch.float32) * 0.01

    # symmetric quantize a
    a_q, a_scale = sym_quantize_tensor(a)
    # asymmetric quantize b
    b_q, b_chn_scale, b_scale_i8, b_zero_i8 = progressive_group_quantize_tensor(
        b, group_size
    )
    # convert to qserve format
    b_q_format, b_chn_scale_format, b_scale_i8_format, b_zero_i8_format = (
        convert_to_qserve_format(b_q, b_chn_scale, b_scale_i8, b_zero_i8, group_size)
    )

    out = qserve_w4a8_per_group_gemm(
        a_q,
        b_q_format,
        b_zero_i8_format,
        b_scale_i8_format,
        b_chn_scale_format,
        a_scale,
    )
    ref_out = torch_w4a8_per_group_gemm(
        a_q, b_q, a_scale, b_chn_scale, b_scale_i8, b_zero_i8, group_size, out_dtype
    )
    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("M", [1, 16, 32, 64, 128, 512, 1024, 4096, 8192])
@pytest.mark.parametrize("N", [128, 512, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("out_dtype", [torch.float16])
def test_accuracy(M, N, K, group_size, out_dtype):
    _test_accuracy_once(M, N, K, group_size, out_dtype, "cuda")


if __name__ == "__main__":
    pytest.main([__file__])
