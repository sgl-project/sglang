"""Tests for the shared BF16 x FP32 GEMM helper."""

import sys

import pytest
import torch

import sglang.jit_kernel.dsv4.gemm as dsv4_gemm
from sglang.jit_kernel.dsv4.gemm import (
    _linear_bf16_fp32_hpc_ops,
    linear_bf16_fp32,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-unit-1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m", [1, 8, 64])
def test_linear_bf16_fp32_matches_fp32_reference(weight_dtype, m):
    torch.manual_seed(42)
    x = torch.randn((m, 4096), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((256, 4096), dtype=torch.float32, device="cuda").to(weight_dtype)

    out = linear_bf16_fp32(x, w)
    ref = x.float() @ w.float().t()

    assert out.shape == (m, 256)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=8e-2)


def test_linear_bf16_fp32_explicit_hpc_ops_dispatch(monkeypatch):
    x = torch.randn((8, 16), dtype=torch.bfloat16)
    w = torch.randn((64, 16), dtype=torch.float32)
    expected = torch.randn((8, 64), dtype=torch.float32)
    seen_min_m = []

    def fake_hpc_ops(x_arg, w_arg, *, min_m=8, **_kwargs):
        assert x_arg is x
        assert w_arg is w
        seen_min_m.append(min_m)
        return expected

    monkeypatch.setattr(dsv4_gemm, "_linear_bf16_fp32_hpc_ops", fake_hpc_ops)

    out = linear_bf16_fp32(x, w, hpc_ops_min_m=128)

    assert out is expected
    assert seen_min_m == [128]


def test_linear_bf16_fp32_explicit_hpc_ops_fallback(monkeypatch):
    x = torch.randn((8, 16), dtype=torch.bfloat16)
    w = torch.randn((64, 16), dtype=torch.float32)

    monkeypatch.setattr(
        dsv4_gemm,
        "_linear_bf16_fp32_hpc_ops",
        lambda *_args, **_kwargs: None,
    )

    out = linear_bf16_fp32(x, w, hpc_ops_min_m=128)
    ref = x.float() @ w.t()

    assert out.shape == (8, 64)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_linear_bf16_fp32_hpc_ops_optional_path():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("HPC-Ops public wheel is currently SM90-only")
    try:
        import hpc  # noqa: F401
    except Exception:
        pytest.skip("HPC-Ops is not installed")

    torch.manual_seed(42)
    x = torch.randn((16, 4096), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((256, 4096), dtype=torch.float32, device="cuda")

    out = _linear_bf16_fp32_hpc_ops(x, w)
    ref = x.float() @ w.t()

    assert out is not None
    assert out.shape == (16, 256)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=8e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
