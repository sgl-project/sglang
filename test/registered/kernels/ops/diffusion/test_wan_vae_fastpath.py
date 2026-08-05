"""Wan VAE decoder fast path: fused-kernel numerics and gate dispatch
(the lossless off-path must stay bit-exact)."""

import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.triton.wan_rmsnorm_silu import wan_rmsnorm_silu
from sglang.multimodal_gen.runtime.models.vaes.wan_vae_cuda_opt import (
    FusedWanRMSNormSiLU,
    VaeFastPathGate,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanRMS_norm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _cl3d(shape, dtype):
    return torch.randn(shape, device="cuda", dtype=dtype).contiguous(
        memory_format=torch.channels_last_3d
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "x_dtype,affine_dtype,atol,rtol",
    [
        (torch.float32, torch.float32, 1e-5, 1e-5),  # FastWan2.2 fp32 decode
        (torch.bfloat16, torch.float32, 1.5e-1, 3e-2),  # Wan2.1 bf16 autocast
    ],
)
def test_kernel_numerics(x_dtype, affine_dtype, atol, rtol) -> None:
    torch.cuda.manual_seed(0)
    x = _cl3d((1, 96, 3, 10, 14), x_dtype)
    gamma = torch.randn((96, 1, 1, 1), device="cuda", dtype=affine_dtype)
    for bias in (None, torch.randn_like(gamma)):
        expected = F.silu(
            F.normalize(x, dim=1) * 96**0.5 * gamma + (0 if bias is None else bias)
        )
        actual = wan_rmsnorm_silu(x, gamma, bias)
        assert actual is not None and actual.dtype == expected.dtype
        assert actual.stride() == x.stride()
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@torch.no_grad()
def test_fused_module_gate_dispatch() -> None:
    # Gate off must stay bit-exact; gate on must route to the fused kernel.
    torch.cuda.manual_seed(0)
    norm = WanRMS_norm(96, images=False).to(device="cuda", dtype=torch.bfloat16)
    norm.gamma.add_(torch.randn_like(norm.gamma))
    gate = VaeFastPathGate()
    fused = FusedWanRMSNormSiLU(norm, gate)
    # Parameter names must not change (weight transfer matches by name).
    assert [n for n, _ in fused.named_parameters()] == ["gamma"]
    x = _cl3d((1, 96, 3, 10, 14), torch.bfloat16)
    assert torch.equal(fused(x), nn.SiLU()(norm(x)))
    gate.enabled = True
    expected = wan_rmsnorm_silu(x, norm.gamma, rms_scale=float(norm.scale))
    assert torch.equal(fused(x), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
