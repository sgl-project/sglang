"""Focused correctness checks for the FLUX.2 VAE CUDA fast path."""

import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.upsampling import Upsample2D

from sglang.kernels.ops.diffusion.triton import group_norm_silu_twopass as gn_kernel
from sglang.multimodal_gen.runtime.models.vaes import flux2_vae_cuda_opt as vae_opt
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.no_grad()
def test_flux2_vae_fastpath():
    torch.manual_seed(0)
    gate = vae_opt.VaeFastPathGate()
    gn = nn.GroupNorm(32, 128, eps=1e-6).to("cuda", torch.bfloat16)
    x = torch.randn(1, 128, 64, 64, device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last
    )
    ref = F.silu(gn(x))
    fused_gn = vae_opt.FusedGroupNormSiLU(gn, gate)
    assert set(fused_gn.state_dict()) == {"weight", "bias"}
    assert torch.equal(fused_gn(x), ref)
    assert (
        gn_kernel.group_norm_silu_4d(x.contiguous(), gn.weight, gn.bias, 32, 1e-6)
        is None
    )

    gate.enabled = True
    fast = fused_gn(x)
    assert fast.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(fast.float(), ref.float(), atol=0.06, rtol=0)

    gate.enabled = False
    up = Upsample2D(channels=32, use_conv=True).to("cuda", torch.bfloat16)
    fused_up = vae_opt.FusedUpsample2xConv2d(up, gate)
    assert set(fused_up.state_dict()) == {"conv.weight", "conv.bias"}
    x = torch.randn(2, 32, 33, 29, device="cuda", dtype=torch.bfloat16)
    ref = up(x)
    assert torch.equal(fused_up(x), ref)
    assert fused_up._fused_weight is None

    gate.enabled = True
    fast = fused_up(x)
    assert fused_up._fused_weight is not None
    ref_range = ref.float().max() - ref.float().min()
    relative_mse = F.mse_loss(fast.float(), ref.float()) / ref_range.square()
    assert relative_mse < 3.2e-5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
