import sys

import pytest
import torch

from sglang.jit_kernel.set_mla_kv_buffer import (
    can_use_set_mla_kv_buffer,
    set_mla_kv_buffer,
)
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")

DEVICE = "cuda"
CACHE_SIZE = 4096

# (nope_dim, rope_dim) pairs: standard MLA, MLA scale buffer, FP8 nope-extended layout.
SHAPES = get_ci_test_range(
    [(512, 64), (512, 32), (256, 64), (128, 64), (528, 64)],
    [(512, 64), (528, 64)],
)
BATCH_SIZES = get_ci_test_range([1, 7, 64, 257, 1024], [1, 64, 1024])


def _ref(kv_buffer, loc, cache_k_nope, cache_k_rope):
    nope_dim = cache_k_nope.shape[-1]
    n_loc = loc.shape[0]
    src_nope = cache_k_nope.reshape(n_loc, -1)
    src_rope = cache_k_rope.reshape(n_loc, -1)
    kv_view = kv_buffer.view(kv_buffer.shape[0], -1)
    kv_view[loc.long(), :nope_dim] = src_nope
    kv_view[loc.long(), nope_dim : nope_dim + src_rope.shape[-1]] = src_rope


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
    _ref(kv_ref, loc, cache_k_nope, cache_k_rope)

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
    _ref(kv_ref, loc, cache_k_nope, cache_k_rope)

    assert torch.equal(kv_buffer, kv_ref)


def test_set_mla_kv_buffer_uint8_byte_layout():
    """FP8 NSA byte-layout: cache_k_nope is uint8 with [fp8(512) | scales(16)] = 528,
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
    _ref(kv_ref, loc, cache_k_nope, cache_k_rope)

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
