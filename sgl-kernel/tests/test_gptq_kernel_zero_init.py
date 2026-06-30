"""Test for the gptq_gemm uninitialized-output accumulation race.

The quantized gptq_gemm path accumulates K-slice partials into the output via
atomicAdd across gridDim.z blocks, so the output must be zero-initialized. This
test fills the allocator with non-zero bytes before each call and uses
K >> BLOCK_KN_SIZE (gridDim.z > 1) to surface a missing zero-init.
"""

import sys

import pytest
import torch
from sgl_kernel import gptq_gemm

from sglang.srt.layers.quantization.utils import pack_cols, pack_rows


def _poison_allocator(device, dtype, numel):
    # Free non-zero blocks so a later torch::empty of the same size reuses them
    for _ in range(4):
        junk = torch.full((numel,), 12345.0, dtype=dtype, device=device)
        del junk
    torch.cuda.synchronize()


def _build_4bit_gptq_weight(K, N, group_size, dtype, device):
    assert K % group_size == 0
    num_groups = K // group_size

    b_fp = torch.randn(K, N, dtype=dtype, device=device)
    g_idx = torch.tensor(
        [i // group_size for i in range(K)], dtype=torch.int32, device=device
    )
    b_grouped = b_fp.reshape(num_groups, group_size, N)

    b_max = torch.max(b_grouped, dim=1, keepdim=True)[0]
    b_min = torch.min(b_grouped, dim=1, keepdim=True)[0]
    scales = ((b_max - b_min) / (2**4 - 1)).clamp(min=1e-6)
    zeros_float = (-b_min / scales).round()

    q_b = (b_grouped / scales + zeros_float).round().clamp(0, 2**4 - 1).to(torch.uint8)
    q_zeros_unpacked = (zeros_float.to(torch.uint8) - 1).reshape(num_groups, N)

    b_q_weight = pack_rows(q_b.reshape(K, N), 4, K, N)
    b_gptq_qzeros = pack_cols(q_zeros_unpacked, 4, num_groups, N)
    b_gptq_scales = scales.squeeze(1)
    return b_q_weight, b_gptq_qzeros, b_gptq_scales, g_idx


def _dequantize_4bit(b_q_weight, b_gptq_qzeros, b_gptq_scales, K, N, group_size):
    pack_factor = 32 // 4
    unpacked_w = torch.empty(
        b_q_weight.shape[0] * pack_factor,
        b_q_weight.shape[1],
        dtype=torch.uint8,
        device=b_q_weight.device,
    )
    for i in range(pack_factor):
        unpacked_w[i::pack_factor, :] = (b_q_weight >> (i * 4)) & 0x0F

    unpacked_z = torch.empty(
        b_gptq_qzeros.shape[0],
        b_gptq_qzeros.shape[1] * pack_factor,
        dtype=torch.uint8,
        device=b_gptq_qzeros.device,
    )
    for i in range(pack_factor):
        unpacked_z[:, i::pack_factor] = (b_gptq_qzeros >> (i * 4)) & 0x0F
    unpacked_z = (unpacked_z + 1).to(b_gptq_scales.dtype)

    scale_zeros = unpacked_z * b_gptq_scales
    current_g_idx = torch.tensor(
        [i // group_size for i in range(K)],
        dtype=torch.int32,
        device=b_q_weight.device,
    )
    scale_mat = b_gptq_scales[current_g_idx]
    scale_zeros_mat = scale_zeros[current_g_idx]
    return (unpacked_w.to(b_gptq_scales.dtype) * scale_mat - scale_zeros_mat).reshape(
        K, N
    )


@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("N", [2048, 4096, 11008])  # 11008 is not a multiple of 512
@pytest.mark.parametrize("K", [2048, 4096])
def test_gptq_gemm_output_zero_init(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.float16
    group_size = 128
    bit = 4
    use_shuffle = False

    b_q_weight, b_gptq_qzeros, b_gptq_scales, g_idx = _build_4bit_gptq_weight(
        K, N, group_size, dtype, device
    )
    b_dequant = _dequantize_4bit(
        b_q_weight, b_gptq_qzeros, b_gptq_scales, K, N, group_size
    )

    # fp32 ground truth: matmul over the (fp16-representable) dequantized weights.
    # Compared with an fp16-appropriate tolerance -- outputs here are O(100) and
    # accumulate over K up to 4096, so a tight fp16-noise-floor tolerance (e.g.
    # 4e-2) false-fails on a correct kernel. A missing zero-init, on the other
    # hand, leaks the poisoned allocator bytes into the output and blows the
    # error up by orders of magnitude (well past any sane tolerance), so this
    # still catches the bug it is meant to guard.
    rtol, atol = 2e-2, 1.0

    # Repeat: the cross-block atomicAdd ordering is nondeterministic, so several
    # trials make a missing zero-init reliable to surface
    for _ in range(20):
        a = torch.randn(M, K, dtype=dtype, device=device)
        c_ref = torch.matmul(a.float(), b_dequant.float())

        _poison_allocator(device, dtype, M * N)
        c_out = gptq_gemm(
            a, b_q_weight, b_gptq_qzeros, b_gptq_scales, g_idx, use_shuffle, bit
        )

        assert not torch.isnan(c_out).any(), "gptq_gemm produced NaNs"
        torch.testing.assert_close(c_out.float(), c_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
