import numpy as np
import pytest
import torch
from sgl_kernel import awq_marlin_repack, gptq_marlin_repack
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.quantization.utils import (
    get_pack_factor,
    pack_cols,
    quantize_weights,
)

GPTQ_MARLIN_TILE = 16


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


def marlin_permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


def marlin_weights(q_w, size_k, size_n, num_bits, perm):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_packed


def get_weight_perm(num_bits: int):
    perm_list: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


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


@pytest.mark.parametrize("num_bits", [4])
@pytest.mark.parametrize("k_tiles,n_tiles", [(1, 1), (2, 2)])
@pytest.mark.parametrize("group_size", [-1, 32])
@pytest.mark.parametrize("has_perm", [True, False])
def test_gptq_marlin_repack(num_bits, k_tiles, n_tiles, group_size, has_perm):
    tile_k, tile_n = 16, 64
    size_k = k_tiles * tile_k
    size_n = n_tiles * tile_n
    pack_factor = 32 // num_bits

    if group_size == -1:
        group_size = size_k

    if size_k % group_size != 0:
        pytest.skip("size_k must be divisible by group_size")

    b_weight = torch.randn((size_k, size_n), dtype=torch.float16, device="cuda")

    _, q_w, _, _ = quantize_weights(
        b_weight, scalar_types.uint4, group_size, zero_points=False
    )

    q_w_gptq = pack_cols(q_w, num_bits, size_k, size_n)

    q_w_permuted = q_w
    if has_perm:
        act_order_perm = torch.randperm(size_k, device="cuda").to(torch.int32)
        q_w_permuted = q_w[act_order_perm]
    else:
        act_order_perm = torch.empty((0,), dtype=torch.int32, device="cuda")

    marlin_layout_perm = get_weight_perm(num_bits).cuda()
    q_w_marlin_ref = marlin_weights(
        q_w_permuted, size_k, size_n, num_bits, marlin_layout_perm
    )

    out_gpu = gptq_marlin_repack(q_w_gptq, act_order_perm, size_k, size_n, num_bits)

    assert out_gpu.is_cuda and out_gpu.dtype == torch.int32

    expected_cols = size_n * tile_k // pack_factor
    assert list(out_gpu.shape) == [size_k // tile_k, expected_cols]

    torch.cuda.synchronize()

    torch.testing.assert_close(out_gpu, q_w_marlin_ref)


if __name__ == "__main__":
    import subprocess

    subprocess.call(["pytest", "--tb=short", str(__file__)])
