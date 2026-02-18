import numpy as np
import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.awq_marlin_repack import (
    awq_marlin_repack as jit_awq_marlin_repack,
)
from sglang.srt.layers.quantization.utils import pack_cols, quantize_weights
from sglang.test.test_marlin_utils import get_weight_perm, marlin_weights

try:
    from sgl_kernel import awq_marlin_repack as aot_awq_marlin_repack

    AOT_AVAILABLE = True
except ImportError:
    AOT_AVAILABLE = False


def awq_pack(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

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
@pytest.mark.parametrize("k_tiles,n_tiles", [(1, 1), (2, 2), (4, 4)])
@pytest.mark.parametrize("group_size", [16, 32])
def test_awq_marlin_repack_jit_vs_aot(num_bits, k_tiles, n_tiles, group_size):
    if not AOT_AVAILABLE:
        pytest.skip("sgl_kernel AOT not available")

    tile_k, tile_n = 16, 64
    size_k = k_tiles * tile_k
    size_n = n_tiles * tile_n

    b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")

    w_ref, q_w, s, zp = quantize_weights(
        b_weight, scalar_types.uint4, group_size, zero_points=True
    )

    q_w_awq = awq_pack(q_w, num_bits, size_k, size_n)

    out_jit = jit_awq_marlin_repack(q_w_awq, size_k, size_n, num_bits)
    out_aot = aot_awq_marlin_repack(q_w_awq, size_k, size_n, num_bits)

    torch.cuda.synchronize()

    # Bitwise equality
    torch.testing.assert_close(out_jit, out_aot, rtol=0, atol=0)


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

    out_gpu = jit_awq_marlin_repack(q_w_awq, size_k, size_n, num_bits)
    assert out_gpu.is_cuda and out_gpu.dtype == torch.int32

    expected_cols = size_n * tile_k // pack_factor
    assert list(out_gpu.shape) == [size_k // tile_k, expected_cols]

    torch.cuda.synchronize()

    torch.testing.assert_close(out_gpu, q_w_marlin)


if __name__ == "__main__":
    import subprocess

    subprocess.call(["pytest", "--tb=short", str(__file__)])
