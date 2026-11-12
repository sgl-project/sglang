"""Tracked RMSNorm FP32 regression; excluded from the blocking PR gate."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_ppu_ci

register_ppu_ci(est_time=90, suite="known-issues-1-ppu", nightly=True)


def test_rmsnorm_fp32():
    from sgl_kernel import rmsnorm

    dtype = torch.float32
    torch.manual_seed(42)
    x = torch.randn(4, 256).to(dtype)
    w = torch.randn(256).to(dtype)
    ref = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * w.float()
    )
    actual = rmsnorm(x.cuda(), w.cuda(), eps=1e-6, enable_pdl=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu().float(), ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
