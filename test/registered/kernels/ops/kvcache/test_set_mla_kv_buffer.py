import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.kernels.ops.kvcache.set_mla_kv_buffer import (
    can_use_set_mla_kv_buffer,
    set_mla_kv_buffer,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
CACHE_SIZE = 4096
TRITON_NOPE_DIM = 128
TRITON_ROPE_DIM = 64
CUDA_TMA_ONLY = pytest.mark.skipif(
    torch.version.hip is not None,
    reason="The TMA bulk-store kernel requires CUDA SM90+",
)

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
@CUDA_TMA_ONLY
def test_set_mla_kv_buffer_correctness(dtype, shape, batch_size):
    nope_dim, rope_dim = shape
    total_dim = nope_dim + rope_dim

    cache_k_nope = torch.randn((batch_size, 1, nope_dim), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((batch_size, 1, rope_dim), dtype=dtype, device=DEVICE)
    kv_buffer = torch.randn((CACHE_SIZE, 1, total_dim), dtype=dtype, device=DEVICE)
    kv_ref = kv_buffer.clone()

    loc = torch.randperm(CACHE_SIZE - 1, device=DEVICE)[:batch_size] + 1

    set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)
    _ref(kv_ref, loc, cache_k_nope, cache_k_rope)

    assert torch.equal(kv_buffer, kv_ref)


@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
@CUDA_TMA_ONLY
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

    loc = (torch.randperm(CACHE_SIZE - 1, device=DEVICE)[:batch_size] + 1).to(loc_dtype)

    set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)
    _ref(kv_ref, loc, cache_k_nope, cache_k_rope)

    assert torch.equal(kv_buffer, kv_ref)


@CUDA_TMA_ONLY
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

    loc = torch.randperm(CACHE_SIZE - 1, device=DEVICE)[:batch_size] + 1

    set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)
    _ref(kv_ref, loc, cache_k_nope, cache_k_rope)

    assert torch.equal(kv_buffer, kv_ref)


@CUDA_TMA_ONLY
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


@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
@CUDA_TMA_ONLY
def test_set_mla_kv_buffer_reserved_skip_index(loc_dtype):
    nope_dim, rope_dim = 512, 64
    dtype = torch.bfloat16
    cache_k_nope = torch.randn((4, 1, nope_dim), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((4, 1, rope_dim), dtype=dtype, device=DEVICE)
    cache_k_nope[[0, 2]] = torch.nan
    cache_k_rope[[0, 2]] = torch.nan
    kv_buffer = torch.randn(
        (CACHE_SIZE, 1, nope_dim + rope_dim), dtype=dtype, device=DEVICE
    )
    reserved_before = kv_buffer[0].clone()
    loc = torch.tensor([0, 7, 0, 9], dtype=loc_dtype, device=DEVICE)

    set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope)

    torch.testing.assert_close(kv_buffer[0], reserved_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        kv_buffer[7, 0, :nope_dim], cache_k_nope[1, 0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        kv_buffer[7, 0, nope_dim:], cache_k_rope[1, 0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        kv_buffer[9, 0, :nope_dim], cache_k_nope[3, 0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        kv_buffer[9, 0, nope_dim:], cache_k_rope[3, 0], rtol=0.0, atol=0.0
    )


@CUDA_TMA_ONLY
def test_set_mla_kv_buffer_zero_index_can_be_written_when_skip_disabled():
    nope_dim, rope_dim = 512, 64
    dtype = torch.bfloat16
    cache_k_nope = torch.randn((1, 1, nope_dim), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((1, 1, rope_dim), dtype=dtype, device=DEVICE)
    kv_buffer = torch.randn(
        (CACHE_SIZE, 1, nope_dim + rope_dim), dtype=dtype, device=DEVICE
    )
    loc = torch.zeros(1, dtype=torch.int64, device=DEVICE)

    set_mla_kv_buffer(
        kv_buffer,
        loc,
        cache_k_nope,
        cache_k_rope,
        reserved_skip_index=-1,
    )

    torch.testing.assert_close(
        kv_buffer[0, 0, :nope_dim], cache_k_nope[0, 0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        kv_buffer[0, 0, nope_dim:], cache_k_rope[0, 0], rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_set_mla_kv_buffer_triton_reserved_skip_index(loc_dtype):
    dtype = torch.bfloat16
    cache_k_nope = torch.randn((4, 1, TRITON_NOPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((4, 1, TRITON_ROPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_nope[[0, 2]] = torch.nan
    cache_k_rope[[0, 2]] = torch.nan
    kv_buffer = torch.randn(
        (CACHE_SIZE, 1, TRITON_NOPE_DIM + TRITON_ROPE_DIM),
        dtype=dtype,
        device=DEVICE,
    )
    reserved_before = kv_buffer[0].clone()
    loc = torch.tensor([0, 7, 0, 9], dtype=loc_dtype, device=DEVICE)

    set_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)

    torch.testing.assert_close(kv_buffer[0], reserved_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        kv_buffer[7, 0],
        torch.cat((cache_k_nope[1, 0], cache_k_rope[1, 0])),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        kv_buffer[9, 0],
        torch.cat((cache_k_nope[3, 0], cache_k_rope[3, 0])),
        rtol=0.0,
        atol=0.0,
    )


def test_set_mla_kv_buffer_triton_zero_index_can_be_written_when_skip_disabled():
    dtype = torch.bfloat16
    cache_k_nope = torch.randn((1, 1, TRITON_NOPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((1, 1, TRITON_ROPE_DIM), dtype=dtype, device=DEVICE)
    kv_buffer = torch.randn(
        (CACHE_SIZE, 1, TRITON_NOPE_DIM + TRITON_ROPE_DIM),
        dtype=dtype,
        device=DEVICE,
    )
    loc = torch.zeros(1, dtype=torch.int64, device=DEVICE)

    set_mla_kv_buffer_triton(
        kv_buffer,
        loc,
        cache_k_nope,
        cache_k_rope,
        reserved_skip_index=-1,
    )

    torch.testing.assert_close(
        kv_buffer[0, 0],
        torch.cat((cache_k_nope[0, 0], cache_k_rope[0, 0])),
        rtol=0.0,
        atol=0.0,
    )


def test_set_mla_kv_buffer_triton_fp8_quant_reserved_skip_index():
    fp8_dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
    cache_k_nope = torch.randn(
        (4, 1, TRITON_NOPE_DIM), dtype=torch.bfloat16, device=DEVICE
    )
    cache_k_rope = torch.randn(
        (4, 1, TRITON_ROPE_DIM), dtype=torch.bfloat16, device=DEVICE
    )
    cache_k_nope[[0, 2]] = torch.nan
    cache_k_rope[[0, 2]] = torch.nan
    kv_buffer = torch.randint(
        0,
        256,
        (CACHE_SIZE, 1, TRITON_NOPE_DIM + TRITON_ROPE_DIM),
        dtype=torch.uint8,
        device=DEVICE,
    )
    reserved_before = kv_buffer[0].clone()
    loc = torch.tensor([0, 7, 0, 9], dtype=torch.int64, device=DEVICE)

    set_mla_kv_buffer_triton_fp8_quant(
        kv_buffer,
        loc,
        cache_k_nope,
        cache_k_rope,
        fp8_dtype,
    )

    torch.testing.assert_close(kv_buffer[0], reserved_before, rtol=0.0, atol=0.0)
    expected = torch.cat((cache_k_nope[1, 0], cache_k_rope[1, 0])).to(fp8_dtype)
    torch.testing.assert_close(
        kv_buffer[7, 0], expected.view(torch.uint8), rtol=0.0, atol=0.0
    )


def test_set_mla_kv_scale_buffer_triton_reserved_skip_index():
    cache_k_nope = torch.randn((4, 1, 16), dtype=torch.float32, device=DEVICE)
    cache_k_rope = torch.randn((4, 1, 4), dtype=torch.float32, device=DEVICE)
    cache_k_nope[[0, 2]] = torch.nan
    cache_k_rope[[0, 2]] = torch.nan
    kv_buffer = torch.randn((CACHE_SIZE, 1, 20), dtype=torch.float32, device=DEVICE)
    reserved_before = kv_buffer[0].clone()
    loc = torch.tensor([0, 7, 0, 9], dtype=torch.int64, device=DEVICE)

    set_mla_kv_scale_buffer_triton(
        kv_buffer,
        loc,
        cache_k_nope,
        cache_k_rope,
    )

    torch.testing.assert_close(kv_buffer[0], reserved_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        kv_buffer[7, 0],
        torch.cat((cache_k_nope[1, 0], cache_k_rope[1, 0])),
        rtol=0.0,
        atol=0.0,
    )


@CUDA_TMA_ONLY
def test_can_use_set_mla_kv_buffer():
    assert can_use_set_mla_kv_buffer(1024, 128)  # bf16 (512,64)
    assert can_use_set_mla_kv_buffer(528, 128)  # fp8 byte layout
    assert not can_use_set_mla_kv_buffer(13, 8)  # not multiple of 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
