import pytest
import torch
from sgl_kernel import gptq_gemm

from sglang.srt.layers.quantization.utils import pack_cols, pack_rows


def torch_dequantize(q_weight, q_zeros, scales, g_idx, use_shuffle, bit, K, N):
    assert bit == 4, "Reference dequantization only supports 4-bit"
    group_size = K // scales.shape[0]
    pack_factor = 32 // bit

    # unpack q_weight: (K//pack_factor, N) -> (K, N)
    unpacked_q_weight = torch.empty(
        q_weight.shape[0] * pack_factor,
        q_weight.shape[1],
        dtype=torch.uint8,
        device=q_weight.device,
    )
    for i in range(pack_factor):
        unpacked_q_weight[i::pack_factor, :] = (q_weight >> (i * 4)) & 0x0F

    # unpack q_zeros: (num_groups, N//pack_factor) -> (num_groups, N)
    unpacked_q_zeros = torch.empty(
        q_zeros.shape[0],
        q_zeros.shape[1] * pack_factor,
        dtype=torch.uint8,
        device=q_zeros.device,
    )
    for i in range(pack_factor):
        unpacked_q_zeros[:, i::pack_factor] = (q_zeros >> (i * 4)) & 0x0F

    unpacked_q_zeros += 1
    unpacked_q_zeros = unpacked_q_zeros.to(scales.dtype)

    scale_zeros = unpacked_q_zeros * scales  # (num_groups, N)

    current_g_idx = torch.tensor(
        [i // group_size for i in range(K)], dtype=torch.int32, device=q_weight.device
    )

    scale_mat = scales[current_g_idx]  # (K, N)
    scale_zeros_mat = scale_zeros[current_g_idx]  # (K, N)

    # dequant: weight * scale - scale_zeros
    dequantized_b = unpacked_q_weight.to(scales.dtype) * scale_mat - scale_zeros_mat

    return dequantized_b.reshape(K, N)


def torch_gptq_gemm(
    a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit
):
    K, N = a.shape[1], b_q_weight.shape[1]

    b_dequant = torch_dequantize(
        b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit, K, N
    )
    c = torch.matmul(a, b_dequant)
    return c


def _test_gptq_gemm_once(M, N, K, bit, group_size, use_shuffle, dtype, device="cuda"):

    b_fp = torch.randn(K, N, dtype=dtype, device=device)

    assert K % group_size == 0, "K must be divisible by group_size"
    num_groups = K // group_size

    if use_shuffle:
        return
    else:
        g_idx = torch.tensor(
            [i // group_size for i in range(K)], dtype=torch.int32, device=device
        )
        b_shuffled = b_fp[g_idx]

    b_grouped = b_shuffled.reshape(num_groups, group_size, N)

    b_max = torch.max(b_grouped, dim=1, keepdim=True)[0]
    b_min = torch.min(b_grouped, dim=1, keepdim=True)[0]

    scales = (b_max - b_min) / (2**bit - 1)
    scales = scales.clamp(min=1e-6)

    zeros_float = (-b_min / scales).round()

    q_b = (
        (b_grouped / scales + zeros_float).round().clamp(0, 2**bit - 1).to(torch.uint8)
    )

    q_zeros_unpacked = zeros_float.to(torch.uint8) - 1

    b_q_weight = pack_rows(q_b.reshape(K, N), bit, K, N)

    q_zeros_unpacked = q_zeros_unpacked.reshape(num_groups, N)
    b_gptq_qzeros = pack_cols(q_zeros_unpacked, bit, num_groups, N)
    b_gptq_scales = scales.squeeze(1)

    a = torch.randn(M, K, dtype=dtype, device=device)

    c_ref = torch_gptq_gemm(
        a, b_q_weight, b_gptq_qzeros, b_gptq_scales, g_idx, use_shuffle, bit
    )
    c_out = gptq_gemm(
        a, b_q_weight, b_gptq_qzeros, b_gptq_scales, g_idx, use_shuffle, bit
    )

    rtol = 4e-2
    atol = 4e-2
    torch.testing.assert_close(c_ref, c_out, rtol=rtol, atol=atol)
    print(
        f"✅ Test passed: M={M}, N={N}, K={K}, bit={bit}, group_size={group_size}, use_shuffle={use_shuffle}, dtype={dtype}"
    )


@pytest.mark.parametrize("M", [1, 8, 128])
@pytest.mark.parametrize("N", [2048, 4096])
@pytest.mark.parametrize("K", [2048, 4096])
@pytest.mark.parametrize("bit", [4])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("use_shuffle", [False])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gptq_gemm(M, N, K, bit, group_size, use_shuffle, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _test_gptq_gemm_once(M, N, K, bit, group_size, use_shuffle, dtype, "cuda")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
