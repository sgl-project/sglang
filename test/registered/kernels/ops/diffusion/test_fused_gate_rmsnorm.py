"""Core checks for the quality-gated fused gate-RMSNorm path."""

import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import fused_gate_rmsnorm as fgn
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DIM, EPS = 4608, 1e-5  # Ideogram 4 hidden size / norm_eps


class _Site(nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.norm = nn.RMSNorm(DIM, eps=EPS, device="cuda", dtype=dtype)
        fgn.mark_fused_gate_rmsnorm_site(self, ("norm",))


def test_fused_matches_ideogram_reference():
    torch.manual_seed(0)
    site = _Site()
    w = site.norm.weight.data
    x = torch.randn(1, 64, DIM, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    # adaln-style strided chunks, as produced by Ideogram's modulation .chunk()
    mods = torch.randn(1, 1, 2 * DIM, device="cuda", dtype=torch.bfloat16)
    scale, gate = mods.chunk(2, dim=-1)
    assert fgn.mount_fused_gate_rmsnorm(site)
    got_scale = fgn.fused_rmsnorm_scale(x, w, 1.0 + scale, EPS)
    got_gate = fgn.fused_rmsnorm_tanh_residual(x, gate, residual, w, EPS)
    ref_scale = F.rms_norm(x, (DIM,), w, EPS) * (1.0 + scale)
    ref_gate = residual + torch.tanh(gate) * F.rms_norm(x, (DIM,), w, EPS)
    # fused path uses bf16-native norm statistics: close, not bit-exact
    torch.testing.assert_close(got_scale, ref_scale, atol=8e-2, rtol=4e-2)
    torch.testing.assert_close(got_gate, ref_gate, atol=8e-2, rtol=4e-2)


def test_mount_guards_all_or_nothing():
    good, bad = _Site(), _Site(torch.float32)
    assert not fgn.mount_fused_gate_rmsnorm(nn.ModuleList([good, bad]))
    assert not fgn.fused_gate_rmsnorm_active(good)
    assert fgn.mount_fused_gate_rmsnorm(good)
    fgn.unmount_fused_gate_rmsnorm(good)
    assert not fgn.fused_gate_rmsnorm_active(good)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
