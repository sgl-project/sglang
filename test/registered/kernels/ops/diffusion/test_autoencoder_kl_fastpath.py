"""Install-path checks for the generic AutoencoderKL CUDA fast path."""

import sys

import pytest
import torch

from sglang.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes import flux2_vae_cuda_opt as vae_opt
from sglang.multimodal_gen.runtime.models.vaes.autoencoder import AutoencoderKL
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _small_config():
    config = StableDiffusion3VAEConfig()
    config.arch_config.latent_channels = 2
    config.arch_config.block_out_channels = (4, 4)
    config.arch_config.down_block_types = ("DownEncoderBlock2D",) * 2
    config.arch_config.up_block_types = ("UpDecoderBlock2D",) * 2
    config.arch_config.layers_per_block = 1
    config.arch_config.norm_num_groups = 1
    config.arch_config.sample_size = 8
    return config


@torch.no_grad()
def test_autoencoder_kl_fastpath_install():
    torch.manual_seed(0)
    vae = AutoencoderKL(_small_config()).to("cuda", torch.bfloat16).eval()
    ref_names = {n for n, _ in vae.named_parameters()}
    ref_sd = {k: v.clone() for k, v in vae.state_dict().items()}
    z = torch.randn(1, 2, 8, 8, device="cuda", dtype=torch.bfloat16)
    ref = vae.decode(z)

    opt = vae_opt.maybe_optimize_autoencoder_kl(vae)
    gate = getattr(opt, vae_opt.GATE_ATTR, None)
    assert gate is not None and not gate.enabled
    # Wrappers must not change parameter FQNs; strict load must round-trip.
    assert {n for n, _ in opt.named_parameters()} == ref_names
    opt.load_state_dict(ref_sd, strict=True)
    # Gate off: bit-for-bit the original path.
    assert torch.equal(opt.decode(z), ref)
    # Gate on: fast path runs and stays close; gate off again restores exact.
    gate.enabled = True
    torch.testing.assert_close(opt.decode(z).float(), ref.float(), atol=0.1, rtol=0)
    gate.enabled = False
    assert torch.equal(opt.decode(z), ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
