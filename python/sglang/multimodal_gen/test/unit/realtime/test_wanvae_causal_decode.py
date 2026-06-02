# SPDX-License-Identifier: Apache-2.0

import copy

import torch

from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    AutoencoderKLWan,
    feat_idx,
    first_chunk,
    forward_context,
    unpatchify,
)


def _tiny_cached_wan_vae() -> AutoencoderKLWan:
    config = WanVAEConfig(
        arch_config=WanVAEArchConfig(
            base_dim=4,
            z_dim=2,
            dim_mult=(1, 1),
            num_res_blocks=1,
            temperal_downsample=(False, True),
            attn_scales=(),
        ),
        load_encoder=False,
        load_decoder=True,
        use_feature_cache=True,
        use_parallel_decode=False,
    )
    return AutoencoderKLWan(config).eval()


def _slow_causal_decode(vae: AutoencoderKLWan, z: torch.Tensor) -> torch.Tensor:
    is_first_chunk = not vae._causal_decode_initialized
    if is_first_chunk:
        vae.clear_cache()

    x = vae.post_quant_conv(z)
    outs = []
    with forward_context(feat_cache_arg=vae._feat_map, feat_idx_arg=vae._conv_idx):
        for i in range(x.shape[2]):
            feat_idx.set(0)
            first_chunk.set(is_first_chunk and i == 0)
            outs.append(vae.decoder(x[:, :, i : i + 1, :, :]))

    out = torch.cat(outs, 2)
    if vae.config.patch_size is not None:
        out = unpatchify(out, patch_size=vae.config.patch_size)
    out = out.float().clamp(-1.0, 1.0)
    vae._causal_decode_initialized = True
    return out


def test_wan_vae_batched_causal_decode_matches_frame_loop():
    torch.manual_seed(0)
    slow = _tiny_cached_wan_vae()
    fast = _tiny_cached_wan_vae()
    fast.load_state_dict(copy.deepcopy(slow.state_dict()))

    first_latents = torch.randn(1, 2, 3, 4, 4)
    second_latents = torch.randn(1, 2, 3, 4, 4)

    with torch.no_grad():
        slow_first = _slow_causal_decode(slow, first_latents)
        fast_first = fast.causal_decode(first_latents)
        slow_second = _slow_causal_decode(slow, second_latents)
        fast_second = fast.causal_decode(second_latents)

    torch.testing.assert_close(fast_first, slow_first)
    torch.testing.assert_close(fast_second, slow_second)
