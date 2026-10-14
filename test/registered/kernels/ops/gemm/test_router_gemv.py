"""Tests for the router-logits GEMV and the BF16x3 router GEMM with FP32 gates."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.router_gemv import (
    router_gemm_bf16x3_cublas,
    router_gemv,
    router_gemv_supported,
    split_fp32_to_bf16x3,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Beam router shape: 256 experts over a 5120 hidden size.
N, K = 256, 5120

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA required",
)


def _inputs(m: int):
    generator = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, K, device="cuda", generator=generator).bfloat16()
    w = 0.02 * torch.randn(N, K, device="cuda", generator=generator)
    return x, w


def _max_error(out: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> float:
    return (out.double() - x.double() @ w.double().t()).abs().max().item()


@requires_cuda
def test_bf16x3_split_reconstructs_fp32_exactly():
    _, w = _inputs(1)
    w = torch.cat([w, w * 1e30, w * 1e-20, -w])
    w3 = split_fp32_to_bf16x3(w)
    n = w.shape[0]
    assert w3.dtype == torch.bfloat16 and w3.shape == (3 * n, K)
    hi, mid, lo = (part.float() for part in w3.split(n))
    assert torch.equal(hi + mid + lo, w)
    # The CUDA kernel must round like the torch reference path.
    assert torch.equal(w3.cpu(), split_fp32_to_bf16x3(w.cpu()))


@requires_cuda
@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_fp32_gate_gemv_matches_fp64(m):
    x, w = _inputs(m)
    assert router_gemv_supported(x, w)
    out = router_gemv(x, w)
    assert out.dtype == torch.float32
    # Measured about 5e-7 on H200 and B300; a BF16 or TF32 weight read is ~1e-3.
    assert _max_error(out, x, w) < 2e-6


@requires_cuda
def test_fp32_gate_gemv_stops_at_its_row_path():
    x, w = _inputs(9)
    assert not router_gemv_supported(x, w)
    assert not router_gemv_supported(x[:1], w.bfloat16())


@requires_cuda
@pytest.mark.parametrize("m", [9, 64, 1024, 16384])
def test_bf16x3_router_gemm_matches_fp64(m):
    x, w = _inputs(m)
    out = router_gemm_bf16x3_cublas(x, w)
    assert out.dtype == torch.float32
    # Measured up to about 5e-5 (tensor-core accumulation); an fp32 SGEMM is about 2e-5.
    assert _max_error(out, x, w) < 1e-4
    assert _max_error(F.linear(x.float(), w.bfloat16().float()), x, w) > 1e-4


@requires_cuda
def test_bf16x3_router_gemm_reads_in_place_weight_updates():
    x, w = _inputs(64)
    router_gemm_bf16x3_cublas(x, w)
    # Online weight updates copy into the gate in place, without a reload hook.
    w.copy_(-2 * w)
    assert _max_error(router_gemm_bf16x3_cublas(x, w), x, w) < 1e-4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
