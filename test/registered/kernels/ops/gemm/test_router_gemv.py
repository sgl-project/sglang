"""Tests for the router-logits GEMV and the BF16x3 router GEMM with FP32 gates."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.router_gemv import (
    router_gemm_bf16x3_cublas,
    router_gemv,
    router_gemv_supported,
)
from sglang.kernels.ops.quantization.bf16x3_split import split_fp32_to_bf16x3
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA required",
)

# Beam router shape: 256 experts over a 5120 hidden size.
N, K = 256, 5120


def _inputs(m: int):
    generator = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, K, device="cuda", generator=generator).bfloat16()
    w = 0.02 * torch.randn(N, K, device="cuda", generator=generator)
    return x, w


def _max_error(out: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> float:
    return (out.double() - x.double() @ w.double().t()).abs().max().item()


def test_bf16x3_split_reconstructs_fp32_exactly():
    _, w = _inputs(1)
    # Ends of the exact range 2**-110 <= |w| < 2**128 - 2**119, full mantissas.
    edges = torch.tensor(
        [
            2.0**-110 * (1 + 2.0**-23),
            2.0**-110 * (2 - 2.0**-23),
            2.0**128 - 2.0**119 - 2.0**104,
        ],
        device="cuda",
    ).repeat(K)[:K]
    w = torch.cat([w, w * 1e30, w * 1e-20, -w, edges[None], -edges[None]])
    w3 = split_fp32_to_bf16x3(w)
    n = w.shape[0]
    assert w3.dtype == torch.bfloat16 and w3.shape == (3 * n, K)
    hi, mid, lo = (part.float() for part in w3.split(n))
    assert torch.equal(hi + mid + lo, w)


@pytest.mark.parametrize("m", [1, 8])
def test_fp32_gate_gemv_matches_fp64(m):
    x, w = _inputs(m)
    assert router_gemv_supported(x, w)
    assert _max_error(router_gemv(x, w), x, w) < 2e-6


@pytest.mark.parametrize("m", [9, 64])
def test_bf16x3_router_gemm_matches_fp64(m):
    x, w = _inputs(m)
    assert _max_error(router_gemm_bf16x3_cublas(x, w), x, w) < 1e-4
    # A BF16 read of the FP32 gate misses the same bound.
    assert _max_error(F.linear(x.float(), w.bfloat16().float()), x, w) > 1e-4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
