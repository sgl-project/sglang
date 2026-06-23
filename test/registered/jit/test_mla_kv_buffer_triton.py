import sys

import pytest
import torch

from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")

DEVICE = "cuda"
CACHE_SIZE = 8192

# Kimi K2.5 MLA get/set buffer shape.
NOPE_DIM = 512
ROPE_DIM = 64
TOTAL_DIM = NOPE_DIM + ROPE_DIM

# Covers get block-split (<256), get per-loc (>=256), and set wrapper branches.
N_LOC_CASES = [4, 128, 255, 256, 511, 512, 2048]


def _make_loc(n_loc: int, pattern: str) -> torch.Tensor:
    if pattern == "seq":
        return torch.arange(n_loc, device=DEVICE, dtype=torch.int64)
    return torch.randperm(CACHE_SIZE, device=DEVICE, dtype=torch.int64)[:n_loc]


def _set_reference(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
) -> torch.Tensor:
    ref = kv_buffer.clone()
    ref[loc.long(), :, :NOPE_DIM] = cache_k_nope
    ref[loc.long(), :, NOPE_DIM:TOTAL_DIM] = cache_k_rope
    return ref


def _get_reference(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    loc = loc.long()
    return (
        kv_buffer[loc, :, :NOPE_DIM].contiguous(),
        kv_buffer[loc, :, NOPE_DIM:TOTAL_DIM].contiguous(),
    )


@pytest.mark.parametrize("n_loc", N_LOC_CASES)
@pytest.mark.parametrize("pattern", ["seq", "rand"])
def test_set_mla_kv_buffer_triton_matches_torch(n_loc, pattern):
    torch.manual_seed(1000 + n_loc)
    dtype = torch.bfloat16
    loc = _make_loc(n_loc, pattern)
    cache_k_nope = torch.randn((n_loc, 1, NOPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((n_loc, 1, ROPE_DIM), dtype=dtype, device=DEVICE)
    kv_buffer = torch.randn((CACHE_SIZE, 1, TOTAL_DIM), dtype=dtype, device=DEVICE)
    ref = _set_reference(kv_buffer, loc, cache_k_nope, cache_k_rope)

    set_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
    torch.cuda.synchronize()

    assert torch.equal(kv_buffer, ref)


@pytest.mark.parametrize("n_loc", N_LOC_CASES)
@pytest.mark.parametrize("pattern", ["seq", "rand"])
def test_get_mla_kv_buffer_triton_matches_torch(n_loc, pattern):
    torch.manual_seed(2000 + n_loc)
    dtype = torch.bfloat16
    loc = _make_loc(n_loc, pattern)
    kv_buffer = torch.randn((CACHE_SIZE, 1, TOTAL_DIM), dtype=dtype, device=DEVICE)
    cache_k_nope = torch.empty((n_loc, 1, NOPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.empty((n_loc, 1, ROPE_DIM), dtype=dtype, device=DEVICE)
    nope_ref, rope_ref = _get_reference(kv_buffer, loc)

    get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
    torch.cuda.synchronize()

    assert torch.equal(cache_k_nope, nope_ref)
    assert torch.equal(cache_k_rope, rope_ref)


@pytest.mark.parametrize("n_loc", [128, 2048])
def test_mla_kv_buffer_triton_round_trip(n_loc):
    torch.manual_seed(3000 + n_loc)
    dtype = torch.bfloat16
    loc = _make_loc(n_loc, "rand")
    cache_k_nope_in = torch.randn((n_loc, 1, NOPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_rope_in = torch.randn((n_loc, 1, ROPE_DIM), dtype=dtype, device=DEVICE)
    kv_buffer = torch.empty((CACHE_SIZE, 1, TOTAL_DIM), dtype=dtype, device=DEVICE)

    set_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope_in, cache_k_rope_in)

    cache_k_nope_out = torch.empty_like(cache_k_nope_in)
    cache_k_rope_out = torch.empty_like(cache_k_rope_in)
    get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope_out, cache_k_rope_out)
    torch.cuda.synchronize()

    assert torch.equal(cache_k_nope_out, cache_k_nope_in)
    assert torch.equal(cache_k_rope_out, cache_k_rope_in)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
