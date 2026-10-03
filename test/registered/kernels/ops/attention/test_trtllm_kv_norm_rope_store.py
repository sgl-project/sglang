"""Byte-level parity with the existing norm/RoPE -> cast -> index_put path."""

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import fused_norm_rope_inplace_triton
from sglang.kernels.ops.attention.dsv4.kv_norm_rope_store import (
    fused_k_norm_rope_uniform_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def inputs(n, dtype=torch.bfloat16, strided=False, index_dtype=torch.int64):
    torch.manual_seed(42)
    width = 1792 if strided else 512
    x = torch.randn(n, width, device="cuda", dtype=dtype)[:, -512:]
    weight = torch.randn(512, device="cuda", dtype=dtype) * 0.5 + 1
    angles = torch.randn(1024, 32, device="cuda", dtype=torch.float32)
    freq = torch.polar(torch.ones_like(angles), angles)
    pos = torch.randint(0, 1024, (n,), device="cuda", dtype=index_dtype)
    loc = torch.randperm(n + 17, device="cuda")[:n].to(index_dtype).contiguous()
    cache = torch.zeros(n + 17, 512, device="cuda", dtype=torch.float8_e4m3fn)
    return x, weight, freq, pos, loc, cache


def reference(x, weight, freq, pos, loc, cache, eps=1e-6):
    y = x.clone()
    fused_norm_rope_inplace_triton(y, weight, eps, freq, positions=pos)
    cache.view(torch.uint8)[loc.long()] = y.to(torch.float8_e4m3fn).view(torch.uint8)


def check(x, weight, freq, pos, loc, cache, eps=1e-6):
    original = x.clone()
    expected = cache.clone()
    reference(x, weight, freq, pos, loc, expected, eps)
    fused_k_norm_rope_uniform_fp8(x, weight, eps, freq, pos, loc, cache)
    torch.cuda.synchronize()
    assert torch.equal(cache.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(x, original)


@pytest.mark.parametrize("n", [0, 1, 5, 6, 63, 64, 65, 315, 378, 384, 1024, 16384])
@pytest.mark.parametrize(
    "strided,index_dtype", [(False, torch.int64), (True, torch.int32)]
)
def test_parity(n, strided, index_dtype):
    check(*inputs(n, strided=strided, index_dtype=index_dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_other_input_dtypes(dtype):
    check(*inputs(65, dtype=dtype, strided=True))


def test_zero_and_negative_slots():
    x, weight, freq, pos, loc, cache = inputs(65)
    x.zero_()
    loc[:10] -= cache.shape[0]
    check(x, weight, freq, pos, loc, cache)


def test_fp8_overflow():
    x, weight, freq, pos, loc, cache = inputs(65)
    weight.mul_(400)
    check(x, weight, freq, pos, loc, cache)


def test_graph_replay_updates():
    x, weight, freq, pos, loc, cache = inputs(65, strided=True, index_dtype=torch.int32)
    fused_k_norm_rope_uniform_fp8(x, weight, 1e-6, freq, pos, loc, cache)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fused_k_norm_rope_uniform_fp8(x, weight, 1e-6, freq, pos, loc, cache)
    for seed in [2, 3]:
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x))
        pos.copy_(torch.randint(0, 1024, pos.shape, device="cuda", dtype=pos.dtype))
        loc.copy_(torch.randperm(cache.shape[0], device="cuda")[: x.shape[0]])
        cache.zero_()
        expected = cache.clone()
        reference(x, weight, freq, pos, loc, expected)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(cache.view(torch.uint8), expected.view(torch.uint8))


def test_large_cache_offset():
    x, weight, freq, pos, _, _ = inputs(2)
    # Sparse writes cross the signed 32-bit byte-offset boundary.
    rows = (1 << 22) + 2
    cache = torch.empty(rows, 512, device="cuda", dtype=torch.float8_e4m3fn)
    loc = torch.tensor([0, rows - 1], device="cuda", dtype=torch.int64)
    expected = torch.zeros(2, 512, device="cuda", dtype=torch.float8_e4m3fn)
    reference(x, weight, freq, pos, torch.arange(2, device="cuda"), expected)
    fused_k_norm_rope_uniform_fp8(x, weight, 1e-6, freq, pos, loc, cache)
    torch.cuda.synchronize()
    assert torch.equal(cache.view(torch.uint8)[loc], expected.view(torch.uint8))
