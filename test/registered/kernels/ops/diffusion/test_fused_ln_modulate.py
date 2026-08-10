import pytest
import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion.fused_ln_modulate import (
    can_fuse_ln_modulate,
    fused_ln_modulate,
    fused_ln_modulate_active,
    mark_fused_ln_modulate_site,
    mount_fused_ln_modulate,
    unmount_fused_ln_modulate,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


@pytest.mark.parametrize("seq_len", [4096, 512])
def test_fused_ln_modulate_matches_reference(seq_len):
    x = torch.randn((1, seq_len, 3072), device="cuda", dtype=torch.bfloat16)
    scale = torch.randn((1, 3072), device="cuda", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    assert can_fuse_ln_modulate(x, scale, shift)
    out = fused_ln_modulate(x, scale, shift, eps=1e-6)
    norm = nn.LayerNorm(3072, eps=1e-6, elementwise_affine=False).cuda()
    ref = norm(x) * (1 + scale[:, None]) + shift[:, None]
    # Contract: bf16 rounding-order-level difference only, not bit-exact.
    torch.testing.assert_close(out, ref, atol=0.0625, rtol=0.05)


def test_fused_ln_modulate_guards_and_mount_protocol():
    x = torch.randn((2, 64, 3072), device="cuda", dtype=torch.bfloat16)
    row = torch.randn((2, 3072), device="cuda", dtype=torch.bfloat16)
    assert not can_fuse_ln_modulate(x, row, row)  # folded affine needs B == 1
    root = nn.Module()
    root.child = nn.Module()
    mark_fused_ln_modulate_site(root.child)
    assert not fused_ln_modulate_active(root.child)
    assert mount_fused_ln_modulate(root)
    assert fused_ln_modulate_active(root.child)
    unmount_fused_ln_modulate(root)
    assert not fused_ln_modulate_active(root.child)
    assert not mount_fused_ln_modulate(nn.Module())  # no marked sites


@torch.no_grad()
def test_mounted_ln_modulate_site_torch_compile_fullgraph():
    class Site(nn.Module):
        def __init__(self):
            super().__init__()
            mark_fused_ln_modulate_site(self)

        def forward(self, x, scale, shift):
            if fused_ln_modulate_active(self) and can_fuse_ln_modulate(x, scale, shift):
                return fused_ln_modulate(x, scale, shift, eps=1e-6)
            return (
                nn.functional.layer_norm(x, (x.shape[-1],), eps=1e-6)
                * (1 + scale[:, None])
                + shift[:, None]
            )

    site = Site()
    assert mount_fused_ln_modulate(site)
    x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    expected = site(x, scale, shift)
    actual = torch.compile(site, fullgraph=True)(x, scale, shift)
    torch.testing.assert_close(actual, expected, atol=0.0625, rtol=0.05)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
