"""Tests for the quality-gated fused linear + tanh-GELU (cublasLt epilogue).

Covers the numerical bound of the fused op vs the tanh-GELU reference, the
fp32 ground-truth equidistance of the fused bf16 path, the static/runtime
guards, the all-or-nothing mount semantics, and the bit-exactness of the
unmounted (lossless default) path.
"""

import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.fused_linear_gelu import (
    can_fuse_linear_gelu,
    can_fuse_linear_gelu_static,
    fused_linear_gelu_tanh,
    mark_fused_gelu_site,
    mount_fused_linear_gelu,
    unmount_fused_linear_gelu,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


class _GeluSite(nn.Module):
    """Minimal stand-in for a model tanh-GELU up-projection site."""

    def __init__(self, dim=64, inner=256, dtype=torch.bfloat16, bias=True):
        super().__init__()
        self.proj = nn.Linear(dim, inner, bias=bias, device=DEVICE, dtype=dtype)
        mark_fused_gelu_site(self, "proj")

    def forward(self, x):
        if self._sgl_fused_gelu_enabled and can_fuse_linear_gelu(self.proj, x):
            return fused_linear_gelu_tanh(x, self.proj.weight, self.proj.bias)
        return F.gelu(self.proj(x), approximate="tanh")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_matches_tanh_gelu_reference(dtype):
    """Fused epilogue == proj + tanh-GELU within half-precision tolerance."""
    x = torch.randn(4096, 3072, device=DEVICE, dtype=dtype)
    linear = nn.Linear(3072, 12288, device=DEVICE, dtype=dtype)
    ref = F.gelu(linear(x), approximate="tanh")
    fused = fused_linear_gelu_tanh(x, linear.weight, linear.bias)
    atol = 2e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(fused, ref, atol=atol, rtol=2e-2)


def test_fused_bf16_equidistant_to_fp32_truth():
    """Fused-bf16 must not sit farther from the fp32 truth than baseline-bf16."""
    x32 = torch.randn(2048, 1024, device=DEVICE)
    linear32 = nn.Linear(1024, 4096, device=DEVICE)
    truth = F.gelu(linear32(x32), approximate="tanh")
    linear16 = linear32.to(torch.bfloat16)
    x16 = x32.to(torch.bfloat16)
    base_err = (F.gelu(linear16(x16), approximate="tanh").float() - truth).abs().mean()
    fused_err = (
        (fused_linear_gelu_tanh(x16, linear16.weight, linear16.bias).float() - truth)
        .abs()
        .mean()
    )
    # Same distance to the fp32 ground truth within 10% -- the fused path is a
    # rounding-order change, not a numerics change.
    assert fused_err <= base_err * 1.1, (fused_err, base_err)


def test_gate_off_is_bit_exact_and_unmount_restores():
    """Lossless default (and post-unmount) output is bit-identical."""
    site = _GeluSite()
    x = torch.randn(128, 64, device=DEVICE, dtype=torch.bfloat16)
    baseline = F.gelu(site.proj(x), approximate="tanh")
    assert not site._sgl_fused_gelu_enabled  # default: unmounted
    assert torch.equal(site(x), baseline)
    assert mount_fused_linear_gelu(site)
    assert site._sgl_fused_gelu_enabled
    unmount_fused_linear_gelu(site)
    assert not site._sgl_fused_gelu_enabled
    assert torch.equal(site(x), baseline)


def test_mount_is_all_or_nothing():
    """One ineligible site keeps every site of the model on the reference path."""
    model = nn.ModuleDict(
        {
            "good": _GeluSite(),
            "bad": _GeluSite(dtype=torch.float32),  # fp32 weights: ineligible
        }
    )
    assert not mount_fused_linear_gelu(model)
    assert not model["good"]._sgl_fused_gelu_enabled
    assert not model["bad"]._sgl_fused_gelu_enabled
    # Without the ineligible site, mounting succeeds.
    assert mount_fused_linear_gelu(model["good"])
    # A model with no marked sites mounts nothing.
    assert not mount_fused_linear_gelu(nn.Linear(8, 8, device=DEVICE))


def test_static_guards_reject_ineligible_linears():
    bad_dtype = nn.Linear(8, 8, device=DEVICE, dtype=torch.float32)
    assert not can_fuse_linear_gelu_static(bad_dtype)
    no_bias = nn.Linear(8, 8, bias=False, device=DEVICE, dtype=torch.bfloat16)
    assert not can_fuse_linear_gelu_static(no_bias)
    ok = nn.Linear(8, 8, device=DEVICE, dtype=torch.bfloat16)
    assert can_fuse_linear_gelu_static(ok)
    ok.skip_bias_add = True  # bias returned separately: epilogue would double-add
    assert not can_fuse_linear_gelu_static(ok)
    ok.skip_bias_add = False
    ok.quant_config = object()  # quantized checkpoints stay on the reference path
    assert not can_fuse_linear_gelu_static(ok)


def test_runtime_guard_falls_back_to_reference():
    """A site cast to fp32 after mounting runs the reference path bit-exactly."""
    site = _GeluSite(dtype=torch.bfloat16)
    assert mount_fused_linear_gelu(site)
    site.float()  # e.g. precision policy change after the mount decision
    x32 = torch.randn(16, 64, device=DEVICE, dtype=torch.float32)
    assert not can_fuse_linear_gelu(site.proj, x32)
    ref = F.gelu(site.proj(x32), approximate="tanh")
    assert torch.equal(site(x32), ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
