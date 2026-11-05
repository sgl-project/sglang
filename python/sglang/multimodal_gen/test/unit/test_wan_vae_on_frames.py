# SPDX-License-Identifier: Apache-2.0
"""AutoencoderKLWan.decode hands out every frame once, in order, once it is final."""

import pytest
import torch

from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan

Z_DIM = 4


def _vae(*, wan22: bool) -> AutoencoderKLWan:
    """A tiny Wan 2.1 VAE, or with ``wan22`` the 2.2 residual blocks and 2x2 patches."""
    channels = 12 if wan22 else 3
    config = WanVAEConfig(
        arch_config=WanVAEArchConfig(
            base_dim=8,
            z_dim=Z_DIM,
            dim_mult=(1, 2, 2, 2),
            num_res_blocks=1,
            latents_mean=(0.0,) * Z_DIM,
            latents_std=(1.0,) * Z_DIM,
            is_residual=wan22,
            in_channels=channels,
            out_channels=channels,
            patch_size=2 if wan22 else None,
        ),
        load_encoder=False,
    )
    torch.manual_seed(0)
    return AutoencoderKLWan(config).eval()


@pytest.mark.parametrize("wan22", [False, True])
def test_causal_decode_hands_out_each_latent_frames_pixels(wan22):
    vae = _vae(wan22=wan22)
    latents = torch.randn(1, Z_DIM, 4, 2, 3)
    parts = []
    with torch.no_grad():
        video = vae.decode(latents, on_frames=parts.append)
        reference = vae.decode(latents)

    assert torch.equal(video, reference)
    assert [part.shape[2] for part in parts] == [1, 4, 4, 4]
    assert torch.equal(torch.cat(parts, dim=2), video)


def test_whole_clip_decode_hands_over_everything_at_the_end():
    vae = _vae(wan22=False)
    vae.use_feature_cache = False
    parts = []
    with torch.no_grad():
        video = vae.decode(torch.randn(1, Z_DIM, 3, 2, 3), on_frames=parts.append)

    assert len(parts) == 1
    assert parts[0] is video
