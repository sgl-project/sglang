import pytest
import torch
from sgl_kernel import qserve_w4a8_per_chn_gemm


# Adapted from https://github.com/mit-han-lab/omniserve/blob/main/omniserve/modeling/layers/quantized_linear/w4a8_linear.py
def convert_to_qserve_format(qweight, scale, zero):
    assert qweight.min() >= 0 and qweight.max() <= 15, "Quantized weight out of range"
    in_features = qweight.shape[1]
    out_features = qweight.shape[0]
    assert in_features % 32 == 0, "Input features must be divisible by 32"
    assert out_features % 32 == 0, "Output features must be divisible by 32"

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
    qweight_unpack_repacked = (
        qweight_unpack_reorder[..., 1] << 4
    ) + qweight_unpack_reorder[..., 0]
    qweight_unpack_repacked = qweight_unpack_repacked.reshape(
        out_features // 32, in_features // 32, 32, 16
    )
    qweight_unpack_repacked = qweight_unpack_repacked.reshape(
        out_features, in_features // 2
    ).contiguous()

    # ---- Pack the scales ---- #
    scale = scale.reshape(out_features).to(torch.float16).contiguous()
    szero = zero.reshape(out_features).to(torch.float16).contiguous() * scale

    return qweight_unpack_repacked, scale, szero


# INT4 Quantization
def asym_quantize_tensor(tensor):
    tensor_min = tensor.min(dim=-1, keepdim=True)[0]
    tensor_max = tensor.max(dim=-1, keepdim=True)[0]
    q_min = 0
    q_max = 15
    tensor_scale = (tensor_max - tensor_min) / (q_max - q_min)
    tensor_zero = q_min - torch.round(tensor_min / tensor_scale)
    tensor_q = torch.clamp(
        torch.round(tensor / tensor_scale) + tensor_zero, q_min, q_max
    ).to(torch.int8)
    return tensor_q, tensor_scale.to(torch.float16), tensor_zero.to(torch.int8)


# INT8 Quantization
def sym_quantize_tensor(tensor):
    tensor_scale = tensor.abs().max(dim=-1, keepdim=True)[0] / 127
    tensor_q = torch.clamp(torch.round(tensor / tensor_scale), -128, 127).to(torch.int8)
    return tensor_q, tensor_scale.to(torch.float16)


def torch_w4a8_per_chn_gemm(a, b, a_scale, b_scale, b_zero, out_dtype):
    print(a.shape)
    print(b.shape)
    print(b_zero.shape)
    o = torch.matmul(
        a.to(torch.float16), (b.to(torch.float16) - b_zero.to(torch.float16)).t()
    )
    o = o * a_scale.view(-1, 1) * b_scale.view(1, -1)
    return o.to(out_dtype)


def _test_accuracy_once(M, N, K, out_dtype, device):
    # to avoid overflow, multiply 0.01
    a = torch.randn((M, K), device=device, dtype=torch.float32) * 0.01
    b = torch.randn((N, K), device=device, dtype=torch.float32) * 0.01

    # symmetric quantize a
    a_q, a_scale = sym_quantize_tensor(a)
    # asymmetric quantize b
    b_q, b_scale, b_zero = asym_quantize_tensor(b)
    # convert to qserve format
    b_q_format, b_scale_format, b_szero_format = convert_to_qserve_format(
        b_q, b_scale, b_zero
    )

    # cal sum of every row of a
    a_sum = a.sum(dim=-1, keepdim=True).to(torch.float16)
    out = qserve_w4a8_per_chn_gemm(
        a_q, b_q_format, b_scale_format, a_scale, b_szero_format, a_sum
    )
    ref_out = torch_w4a8_per_chn_gemm(a_q, b_q, a_scale, b_scale, b_zero, out_dtype)
    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize("M", [1, 16, 32, 64, 128, 512, 1024, 4096, 8192])
@pytest.mark.parametrize("N", [128, 512, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("K", [512, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("out_dtype", [torch.float16])
def test_accuracy(M, N, K, out_dtype):
    _test_accuracy_once(M, N, K, out_dtype, "cuda")


if __name__ == "__main__":
    pytest.main([__file__])
