import pytest
import torch

from sglang.kernels.ops.kvcache.mla_buffer import (
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
CACHE_SIZE = 32
NOPE_DIM = 128
ROPE_DIM = 64


@pytest.mark.parametrize("loc_dtype", [torch.int32, torch.int64])
def test_set_mla_kv_buffer_triton_reserved_skip_index(loc_dtype):
    dtype = torch.bfloat16
    cache_k_nope = torch.randn((4, 1, NOPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((4, 1, ROPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_nope[[0, 2]] = torch.nan
    cache_k_rope[[0, 2]] = torch.nan
    kv_buffer = torch.randn(
        (CACHE_SIZE, 1, NOPE_DIM + ROPE_DIM), dtype=dtype, device=DEVICE
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
    cache_k_nope = torch.randn((1, 1, NOPE_DIM), dtype=dtype, device=DEVICE)
    cache_k_rope = torch.randn((1, 1, ROPE_DIM), dtype=dtype, device=DEVICE)
    kv_buffer = torch.randn(
        (CACHE_SIZE, 1, NOPE_DIM + ROPE_DIM), dtype=dtype, device=DEVICE
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


def test_set_mla_kv_buffer_fp8_quant_reserved_skip_index():
    fp8_dtype = (
        torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn
    )
    cache_k_nope = torch.randn(
        (4, 1, NOPE_DIM), dtype=torch.bfloat16, device=DEVICE
    )
    cache_k_rope = torch.randn(
        (4, 1, ROPE_DIM), dtype=torch.bfloat16, device=DEVICE
    )
    cache_k_nope[[0, 2]] = torch.nan
    cache_k_rope[[0, 2]] = torch.nan
    kv_buffer = torch.randint(
        0,
        256,
        (CACHE_SIZE, 1, NOPE_DIM + ROPE_DIM),
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


def test_set_mla_kv_scale_buffer_reserved_skip_index():
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
