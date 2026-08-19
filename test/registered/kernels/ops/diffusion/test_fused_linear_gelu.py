"""Core checks for the quality-gated linear + tanh-GELU fusion."""

import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import fused_linear_gelu as gelu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Site(nn.Module):
    def __init__(self, dtype=torch.bfloat16, bias=True):
        super().__init__()
        self.proj = nn.Linear(64, 256, bias=bias, device="cuda", dtype=dtype)
        gelu.mark_fused_gelu_site(self, "proj")

    def forward(self, x):
        if gelu.fused_gelu_active(self) and gelu.can_fuse_linear_gelu(self.proj, x):
            return gelu.fused_linear_gelu_tanh(x, self.proj.weight, self.proj.bias)
        return F.gelu(self.proj(x), approximate="tanh")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_matches_reference(dtype):
    torch.manual_seed(0)
    site = _Site(dtype)
    x = torch.randn(512, 64, device="cuda", dtype=dtype)
    ref = site(x)
    assert gelu.mount_fused_linear_gelu(site)
    atol = 2e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(site(x), ref, atol=atol, rtol=2e-2)


def test_flux_gelu_proj_site():
    """FLUX.1 shared-FF site: gate off is bit-exact, gate on is close."""
    from sglang.multimodal_gen.runtime.models.dits.flux import FluxFusedGELUProj

    torch.manual_seed(0)
    proj = nn.Linear(3072, 12288, device="cuda", dtype=torch.bfloat16)
    site = FluxFusedGELUProj(proj)
    x = torch.randn(1, 512, 3072, device="cuda", dtype=torch.bfloat16)
    ref = F.gelu(proj(x), approximate="tanh")

    assert torch.equal(site(x), ref)  # unmounted default: bit-exact reference
    assert gelu.mount_fused_linear_gelu(site)
    torch.testing.assert_close(site(x), ref, atol=2e-2, rtol=2e-2)
    gelu.unmount_fused_linear_gelu(site)
    assert torch.equal(site(x), ref)


def test_mount_guards_and_lossless_path():
    torch.manual_seed(0)
    good, bad = _Site(), _Site(torch.float32)
    model = nn.ModuleList([good, bad])
    assert not gelu.mount_fused_linear_gelu(model)
    assert not gelu.fused_gelu_active(good)

    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    ref = good(x)
    assert gelu.mount_fused_linear_gelu(good)
    gelu.unmount_fused_linear_gelu(good)
    assert torch.equal(good(x), ref)

    no_bias = nn.Linear(8, 8, bias=False, device="cuda", dtype=torch.bfloat16)
    assert not gelu.can_fuse_linear_gelu_static(no_bias)
    assert not gelu.can_fuse_linear_gelu(good.proj, x.float())


@torch.no_grad()
def test_mounted_site_torch_compile_fullgraph():
    site = _Site()
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    assert gelu.mount_fused_linear_gelu(site)
    expected = site(x)
    actual = torch.compile(site, fullgraph=True)(x)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
