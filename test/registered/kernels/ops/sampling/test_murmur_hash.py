import sys

import numpy as np
import pytest
import torch

from sglang.kernels.ops.sampling.murmur_hash import (
    _murmur_hash32_jit,
    _murmur_hash32_triton,
    murmur_hash32,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# (positions_dtype, col_indices_dtype) used by production call sites plus a
# few extra integer dtypes the kernels accept.
INDEX_DTYPE_COMBOS = [
    (torch.int64, torch.int64),  # sampler path
    (torch.uint64, torch.int64),  # eagle verify-side path
    (torch.int32, torch.int32),
    (torch.uint32, torch.uint32),
    (torch.int64, torch.uint32),
]

SIZES = [1, 2, 33, 100, 1023, 1024, 1025, 4096]


def _seed_positions(n, pos_dtype, device):
    seed = torch.randint(
        0, torch.iinfo(torch.int64).max, (n,), dtype=torch.uint64, device=device
    )
    positions = torch.randint(0, 1 << 30, (n,), device=device).to(pos_dtype)
    return seed, positions


def _murmur_hash32_reference(seed, positions, col_indices):
    """Bit-exact PyTorch CPU reference implementation for a CPU oracle."""

    seed_np = seed.cpu().numpy().astype(np.uint64).reshape(-1, 1)
    positions_np = (
        positions.cpu().numpy().astype(np.uint32).astype(np.uint64).reshape(-1, 1)
    )
    col_indices_np = (
        col_indices.cpu().numpy().astype(np.uint32).astype(np.uint64).reshape(1, -1)
    )

    M32 = np.uint64(0xFFFFFFFF)
    C1 = np.uint64(0xCC9E2D51)
    C2 = np.uint64(0x1B873593)
    N5 = np.uint64(5)
    NN = np.uint64(0xE6546B64)

    def mix(h, k):
        k = (k * C1) & M32
        k = ((k << np.uint64(15)) | (k >> np.uint64(17))) & M32
        k = (k * C2) & M32
        h = (h ^ k) & M32
        h = ((h << np.uint64(13)) | (h >> np.uint64(19))) & M32
        h = (h * N5 + NN) & M32
        return h

    def fmix(h):
        h = (h ^ (h >> np.uint64(16))) & M32
        h = (h * np.uint64(0x85EBCA6B)) & M32
        h = (h ^ (h >> np.uint64(13))) & M32
        h = (h * np.uint64(0xC2B2AE35)) & M32
        h = (h ^ (h >> np.uint64(16))) & M32
        return h

    h = mix(np.zeros_like(seed_np), seed_np & M32)
    h = mix(h, (seed_np >> np.uint64(32)) & M32)
    h = mix(h, positions_np)
    h = mix(h, col_indices_np)
    h = h ^ np.uint64(16)
    res_np = fmix(h).astype(np.uint32).reshape(-1)
    return torch.from_numpy(res_np)


@pytest.mark.parametrize("n", [1, 2, 33])
@pytest.mark.parametrize("m", [1, 2, 100, 1023, 1024, 1025])
@pytest.mark.parametrize("pos_dtype, col_dtype", INDEX_DTYPE_COMBOS)
def test_murmur_hash_jit_matches_reference(n, m, pos_dtype, col_dtype):
    seed, positions = _seed_positions(n, pos_dtype, "cuda")
    col_indices = torch.arange(m, device="cuda", dtype=torch.int64).to(col_dtype)
    out = _murmur_hash32_jit(seed, positions, col_indices)
    expected = _murmur_hash32_reference(seed, positions, col_indices)
    assert torch.equal(out.reshape(-1), expected.to("cuda"))


@pytest.mark.parametrize("n", [1, 2, 33])
@pytest.mark.parametrize("m", [1, 2, 100, 1023, 1024, 1025, 4096])
@pytest.mark.parametrize("pos_dtype, col_dtype", INDEX_DTYPE_COMBOS)
def test_murmur_hash_jit_matches_triton(n, m, pos_dtype, col_dtype):
    seed, positions = _seed_positions(n, pos_dtype, "cuda")
    col_indices = torch.arange(m, device="cuda", dtype=torch.int64).to(col_dtype)
    jit_out = _murmur_hash32_jit(seed, positions, col_indices)
    triton_out = _murmur_hash32_triton(seed, positions, col_indices)
    assert torch.equal(jit_out, triton_out)


def test_murmur_hash_large_grid():
    # m spans multiple 256-thread blocks; n larger than one grid.y row.
    n, m = 500, 10000
    seed, positions = _seed_positions(n, torch.int64, "cuda")
    col_indices = torch.arange(m, device="cuda", dtype=torch.int64)
    out = _murmur_hash32_jit(seed, positions, col_indices)
    assert out.shape == (n, m)
    assert torch.equal(
        out.reshape(-1),
        _murmur_hash32_reference(seed, positions, col_indices).to("cuda"),
    )


def test_murmur_hash_empty_batch():
    seed = torch.empty(0, dtype=torch.uint64, device="cuda")
    positions = torch.empty(0, dtype=torch.int64, device="cuda")
    col_indices = torch.arange(16, device="cuda", dtype=torch.int64)
    out = _murmur_hash32_jit(seed, positions, col_indices)
    assert out.shape == (0, 16)


def test_murmur_hash_entry_point_matches_jit():
    n, m = 4, 1024
    seed, positions = _seed_positions(n, torch.int64, "cuda")
    col_indices = torch.arange(m, device="cuda", dtype=torch.int64)
    assert torch.equal(
        murmur_hash32(seed, positions, col_indices),
        _murmur_hash32_jit(seed, positions, col_indices),
    )


def test_murmur_hash_production_shapes():
    # Sampler: batch x vocab. Eagle: batch x (draft_token_num + 1).
    seed = torch.randint(
        0, torch.iinfo(torch.int64).max, (8,), dtype=torch.uint64, device="cuda"
    )
    positions = torch.arange(8, dtype=torch.int64, device="cuda")
    col_indices = torch.arange(32768, device="cuda", dtype=torch.int64)
    out = murmur_hash32(seed, positions, col_indices)
    assert out.shape == (8, 32768)
    assert torch.equal(
        out.reshape(-1),
        _murmur_hash32_reference(seed, positions, col_indices).to("cuda"),
    )

    cols = torch.arange(5, device="cuda", dtype=torch.int64)
    eagle = murmur_hash32(seed, positions.to(torch.uint64), cols)
    assert eagle.shape == (8, 5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
