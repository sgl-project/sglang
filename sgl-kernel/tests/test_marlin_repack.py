import numpy as np
import pytest
import torch
from sgl_kernel import awq_marlin_repack, gptq_marlin_repack
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.quantization.utils import (
    gptq_quantize_weights,
    pack_cols,
    pack_rows,
    quantize_weights,
    sort_weights,
)
from sglang.test.test_marlin_utils import get_weight_perm, marlin_weights

GPTQ_MARLIN_TILE = 16
MARLIN_K_CHUNKS = [128]
MARLIN_N_CHUNKS = [64, 256]

MNK_FACTORS = [
    (1, 1, 1),
    (1, 4, 8),
    (1, 7, 5),
    (13, 17, 67),
    (26, 37, 13),
    (67, 13, 11),
    (257, 13, 11),
    (658, 13, 11),
]


def awq_pack(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    # Interleave column dim (for the dequantize code) and pack it to int32
    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_w = q_w.reshape((-1, len(interleave)))[:, interleave].ravel()
    q_w = q_w.reshape((-1, size_n)).contiguous()

    return pack_cols(q_w, num_bits, size_k, size_n)


@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("k_tiles,n_tiles", [(1, 1), (2, 2)])
@pytest.mark.parametrize("group_size", [16, 32])
def test_awq_marlin_repack_correct(num_bits, k_tiles, n_tiles, group_size):
    tile_k, tile_n = 16, 64
    size_k = k_tiles * tile_k
    size_n = n_tiles * tile_n
    pack_factor = 32 // num_bits

    b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")

    w_ref, q_w, s, zp = quantize_weights(
        b_weight, scalar_types.uint4, group_size, zero_points=True
    )

    q_w_awq = awq_pack(q_w, num_bits, size_k, size_n)

    weight_perm = get_weight_perm(num_bits)
    q_w_marlin = marlin_weights(q_w, size_k, size_n, num_bits, weight_perm)

    out_gpu = awq_marlin_repack(q_w_awq, size_k, size_n, num_bits)
    assert out_gpu.is_cuda and out_gpu.dtype == torch.int32

    expected_cols = size_n * tile_k // pack_factor
    assert list(out_gpu.shape) == [size_k // tile_k, expected_cols]

    torch.cuda.synchronize()

    torch.testing.assert_close(out_gpu, q_w_marlin)


@pytest.mark.parametrize("k_chunk", MARLIN_K_CHUNKS)
@pytest.mark.parametrize("n_chunk", MARLIN_N_CHUNKS)
@pytest.mark.parametrize("quant_type", [scalar_types.uint4b8])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_gptq_marlin_repack(
    k_chunk, n_chunk, quant_type, group_size, act_order, mnk_factors
):
    m_factor, n_factor, k_factor = mnk_factors

    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    if size_k % group_size != 0:
        pytest.skip("size_k must be divisible by group_size")

    # Create input
    b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        b_weight, quant_type, group_size, act_order
    )

    q_w_gptq = pack_rows(q_w, quant_type.size_bits, size_k, size_n)

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=b_weight.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    marlin_layout_perm = get_weight_perm(quant_type.size_bits)
    q_w_marlin_ref = marlin_weights(
        q_w, size_k, size_n, quant_type.size_bits, marlin_layout_perm
    )

    # Run Marlin repack GPU kernel
    q_w_marlin = gptq_marlin_repack(
        q_w_gptq, sort_indices, size_k, size_n, quant_type.size_bits
    )

    torch.cuda.synchronize()

    torch.testing.assert_close(q_w_marlin, q_w_marlin_ref)


if __name__ == "__main__":
    import subprocess

    subprocess.call(["pytest", "--tb=short", str(__file__)])
