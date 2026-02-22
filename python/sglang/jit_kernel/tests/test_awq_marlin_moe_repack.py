import numpy as np
import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.awq_marlin_repack import (
    awq_marlin_moe_repack as jit_awq_marlin_moe_repack,
)
from sglang.srt.layers.quantization.utils import pack_cols, quantize_weights

try:
    from sgl_kernel import awq_marlin_moe_repack as aot_awq_marlin_moe_repack

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


@pytest.mark.parametrize("num_bits", [4])
@pytest.mark.parametrize("num_experts", [2, 4, 8])
@pytest.mark.parametrize("k_tiles,n_tiles", [(1, 1), (2, 2), (4, 4)])
@pytest.mark.parametrize("group_size", [16, 32])
def test_awq_marlin_moe_repack_jit_vs_aot(
    num_bits, num_experts, k_tiles, n_tiles, group_size
):
    if not AOT_AVAILABLE:
        pytest.skip("sgl_kernel AOT not available")

    tile_k, tile_n = 16, 64
    size_k = k_tiles * tile_k
    size_n = n_tiles * tile_n
    pack_factor = 32 // num_bits

    # Create per-expert AWQ-packed weights
    b_q_weight = torch.empty(
        (num_experts, size_k, size_n // pack_factor),
        dtype=torch.int32,
        device="cuda",
    )
    for e in range(num_experts):
        b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")
        w_ref, q_w, s, zp = quantize_weights(
            b_weight, scalar_types.uint4, group_size, zero_points=True
        )
        b_q_weight[e] = awq_pack(q_w, num_bits, size_k, size_n)

    perm = torch.empty((num_experts, 0), dtype=torch.int32, device="cuda")

    out_jit = jit_awq_marlin_moe_repack(b_q_weight, perm, size_k, size_n, num_bits)
    out_aot = aot_awq_marlin_moe_repack(b_q_weight, perm, size_k, size_n, num_bits)

    torch.cuda.synchronize()

    # Bitwise equality
    torch.testing.assert_close(out_jit, out_aot, rtol=0, atol=0)


@pytest.mark.parametrize("num_bits", [4])
@pytest.mark.parametrize("num_experts", [2, 4])
@pytest.mark.parametrize("k_tiles,n_tiles", [(1, 1), (2, 2)])
@pytest.mark.parametrize("group_size", [16, 32])
def test_awq_marlin_moe_repack_shape(
    num_bits, num_experts, k_tiles, n_tiles, group_size
):
    tile_k, tile_n = 16, 64
    size_k = k_tiles * tile_k
    size_n = n_tiles * tile_n
    pack_factor = 32 // num_bits

    # Create per-expert AWQ-packed weights
    b_q_weight = torch.empty(
        (num_experts, size_k, size_n // pack_factor),
        dtype=torch.int32,
        device="cuda",
    )
    for e in range(num_experts):
        b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")
        w_ref, q_w, s, zp = quantize_weights(
            b_weight, scalar_types.uint4, group_size, zero_points=True
        )
        b_q_weight[e] = awq_pack(q_w, num_bits, size_k, size_n)

    perm = torch.empty((num_experts, 0), dtype=torch.int32, device="cuda")

    out = jit_awq_marlin_moe_repack(b_q_weight, perm, size_k, size_n, num_bits)
    torch.cuda.synchronize()

    assert out.is_cuda and out.dtype == torch.int32
    expected_shape = (num_experts, size_k // 16, size_n * (num_bits // 2))
    assert list(out.shape) == list(expected_shape)


if __name__ == "__main__":
    import subprocess

    subprocess.call(["pytest", "--tb=short", str(__file__)])
