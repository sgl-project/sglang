import sys

import pytest
import torch

from sglang.jit_kernel.set_mla_kv_buffer import (
    can_use_set_mla_kv_buffer,
    set_mla_kv_buffer,
)
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")

DEVICE = "cuda"
CACHE_SIZE = 8192

# (nope_dim, rope_dim) pairs: standard MLA, MLA scale buffer, FP8 nope-extended layout.
SHAPES = get_ci_test_range(
    [(512, 64), (512, 32), (256, 64), (128, 64), (528, 64)],
    [(512, 64), (528, 64)],
)
BATCH_SIZES = get_ci_test_range([1, 7, 64, 257, 1024], [1, 64, 1024])

# Kimi K2.5 MLA get/set buffer shape.
KIMI_NOPE_DIM = 512
KIMI_ROPE_DIM = 64
KIMI_TOTAL_DIM = KIMI_NOPE_DIM + KIMI_ROPE_DIM

# Covers get block-split (<256), get per-loc (>=256), and set wrapper branches.
N_LOC_CASES = [4, 128, 255, 256, 511, 512, 2048]


def _set_reference_inplace(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    n_loc = loc.shape[0]
    src_nope = cache_k_nope.reshape(n_loc, -1)
    src_rope = cache_k_rope.reshape(n_loc, -1)
    kv_view = kv_buffer.view(kv_buffer.shape[0], -1)
    kv_view[loc.long(), :nope_dim] = src_nope
    kv_view[loc.long(), nope_dim : nope_dim + src_rope.shape[-1]] = src_rope


def _make_loc(n_loc: int, pattern: str) -> torch.Tensor:
    if pattern == "seq":
        return torch.arange(n_loc, device=DEVICE, dtype=torch.int64)
    return torch.randperm(CACHE_SIZE, device=DEVICE, dtype=torch.int64)[:n_loc]


def _get_reference(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    loc = loc.long()
    return (
        kv_buffer[loc, :, :KIMI_NOPE_DIM].contiguous(),
        kv_buffer[loc, :, KIMI_NOPE_DIM:KIMI_TOTAL_DIM].contiguous(),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_set_mla_kv_buffer_correctness(dtype, shape, batch_size):
    nope_dim, rope_dim = shape
    total_dim = nope_dim + rope_dim

    cache_k_nope = torch.randn((batch_size, 1, nope_dim), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((batch_size, 1, rope_dim), dtype=dtype, device=DEVICE)
    kv_buffer = torch.randn((CACHE_SIZE, 1, total_dim), dtype=dtype, device=DEVICE)
    kv_ref = kv_buffer.clone()

    loc = torch.randperm(CACHE_SIZE, device=DEVICE)[:batch_size]

    set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)
    _set_reference_inplace(kv_ref, loc, cache_k_nope, cache_k_rope)

    assert torch.equal(kv_buffer, kv_ref)


@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_set_mla_kv_buffer_loc_dtypes(loc_dtype):
    nope_dim, rope_dim = 512, 64
    batch_size = 128
    dtype = torch.bfloat16

    cache_k_nope = torch.randn((batch_size, 1, nope_dim), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((batch_size, 1, rope_dim), dtype=dtype, device=DEVICE)
    kv_buffer = torch.randn(
        (CACHE_SIZE, 1, nope_dim + rope_dim), dtype=dtype, device=DEVICE
    )
    kv_ref = kv_buffer.clone()

    loc = torch.randperm(CACHE_SIZE, device=DEVICE)[:batch_size].to(loc_dtype)

    set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)
    _set_reference_inplace(kv_ref, loc, cache_k_nope, cache_k_rope)

    assert torch.equal(kv_buffer, kv_ref)


def test_set_mla_kv_buffer_uint8_byte_layout():
    """FP8 DSA byte-layout: cache_k_nope is uint8 with [fp8(512) | scales(16)] = 528,
    cache_k_rope is uint8 [128]; total payload = 656 bytes."""
    nope_bytes, rope_bytes = 528, 128
    batch_size = 64
    dtype = torch.uint8

    cache_k_nope = torch.randint(
        0, 256, (batch_size, 1, nope_bytes), dtype=dtype, device=DEVICE
    )
    cache_k_rope = torch.randint(
        0, 256, (batch_size, 1, rope_bytes), dtype=dtype, device=DEVICE
    )
    kv_buffer = torch.randint(
        0, 256, (CACHE_SIZE, 1, nope_bytes + rope_bytes), dtype=dtype, device=DEVICE
    )
    kv_ref = kv_buffer.clone()

    loc = torch.randperm(CACHE_SIZE, device=DEVICE)[:batch_size]

    set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)
    _set_reference_inplace(kv_ref, loc, cache_k_nope, cache_k_rope)

    assert torch.equal(kv_buffer, kv_ref)


def test_set_mla_kv_buffer_empty_loc():
    nope_dim, rope_dim = 512, 64
    dtype = torch.bfloat16
    cache_k_nope = torch.empty((0, 1, nope_dim), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.empty((0, 1, rope_dim), dtype=dtype, device=DEVICE)
    kv_buffer = torch.randn(
        (CACHE_SIZE, 1, nope_dim + rope_dim), dtype=dtype, device=DEVICE
    )
    kv_before = kv_buffer.clone()

    loc = torch.empty((0,), dtype=torch.int64, device=DEVICE)
    set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)

    assert torch.equal(kv_buffer, kv_before)


def test_can_use_set_mla_kv_buffer():
    assert can_use_set_mla_kv_buffer(1024, 128)  # bf16 (512,64)
    assert can_use_set_mla_kv_buffer(528, 128)  # fp8 byte layout
    assert not can_use_set_mla_kv_buffer(13, 8)  # not multiple of 4


@pytest.mark.parametrize("n_loc", N_LOC_CASES)
@pytest.mark.parametrize("pattern", ["seq", "rand"])
def test_set_mla_kv_buffer_wrapper_matches_torch(n_loc, pattern):
    torch.manual_seed(1000 + n_loc)
    dtype = torch.bfloat16
    loc = _make_loc(n_loc, pattern)
    cache_k_nope = torch.randn((n_loc, 1, KIMI_NOPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((n_loc, 1, KIMI_ROPE_DIM), dtype=dtype, device=DEVICE)
    kv_buffer = torch.randn((CACHE_SIZE, 1, KIMI_TOTAL_DIM), dtype=dtype, device=DEVICE)
    ref = kv_buffer.clone()
    _set_reference_inplace(ref, loc, cache_k_nope, cache_k_rope)

    set_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
    torch.cuda.synchronize()

    assert torch.equal(kv_buffer, ref)


@pytest.mark.parametrize("n_loc", N_LOC_CASES)
@pytest.mark.parametrize("pattern", ["seq", "rand"])
def test_get_mla_kv_buffer_wrapper_matches_torch(n_loc, pattern):
    torch.manual_seed(2000 + n_loc)
    dtype = torch.bfloat16
    loc = _make_loc(n_loc, pattern)
    kv_buffer = torch.randn((CACHE_SIZE, 1, KIMI_TOTAL_DIM), dtype=dtype, device=DEVICE)
    cache_k_nope = torch.empty((n_loc, 1, KIMI_NOPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.empty((n_loc, 1, KIMI_ROPE_DIM), dtype=dtype, device=DEVICE)
    nope_ref, rope_ref = _get_reference(kv_buffer, loc)

    get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
    torch.cuda.synchronize()

    assert torch.equal(cache_k_nope, nope_ref)
    assert torch.equal(cache_k_rope, rope_ref)


@pytest.mark.parametrize("n_loc", [128, 2048])
def test_mla_kv_buffer_wrapper_round_trip(n_loc):
    torch.manual_seed(3000 + n_loc)
    dtype = torch.bfloat16
    loc = _make_loc(n_loc, "rand")
    cache_k_nope_in = torch.randn((n_loc, 1, KIMI_NOPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_rope_in = torch.randn((n_loc, 1, KIMI_ROPE_DIM), dtype=dtype, device=DEVICE)
    kv_buffer = torch.empty((CACHE_SIZE, 1, KIMI_TOTAL_DIM), dtype=dtype, device=DEVICE)

    set_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope_in, cache_k_rope_in)

    cache_k_nope_out = torch.empty_like(cache_k_nope_in)
    cache_k_rope_out = torch.empty_like(cache_k_rope_in)
    get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope_out, cache_k_rope_out)
    torch.cuda.synchronize()

    assert torch.equal(cache_k_nope_out, cache_k_nope_in)
    assert torch.equal(cache_k_rope_out, cache_k_rope_in)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
