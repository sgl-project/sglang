# SPDX-License-Identifier: Apache-2.0
"""Shared MiniMax-H3 VAE fixtures for CPU and CUDA integration tests."""

from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)

_H3_ATTENTION_MODULE = (
    "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention"
)


def _h3_tiny_vae(**caps) -> AutoencoderKLLegacy:
    """Two-level 3D CNN encoder and one-block ViT decoder; 32 px tiles of 16.

    Same recipe as the CPU unit tests. ``current_platform.is_cuda`` is patched
    off during construction so the block attention keeps the plain SDPA path
    instead of building a USPAttention that needs server args."""
    torch.manual_seed(0)
    with patch(f"{_H3_ATTENTION_MODULE}.current_platform.is_cuda", return_value=False):
        vae = AutoencoderKLLegacy(
            in_channels=3,
            out_ch=3,
            ch=32,
            embed_dim=4,
            z_channels=4,
            use_3d_conv=True,
            num_res_blocks=1,
            ch_mult=[1, 1],
            space_down=[2, 1],
            space_up=[1, 2],
            time_down=[1, 1],
            padding_mode="reflect",
            use_t_isolated_gn=True,
            causal_encoder=True,
            causal_decoder=False,
            use_vit_decoder=True,
            vit_decoder_kwargs={
                "dim_head": 16,
                "heads": 2,
                "num_layers": 1,
                "norm_type": "rms_norm",
                "qk_norm_type": "rms_norm",
                "qk_norm_affine": False,
                "ffn_activation_fn": "silu",
                "ffn_use_gated": True,
                "rope_dim_ratio": 0.75,
                "rope_theta": 100.0,
                "num_register_tokens": 1,
            },
            clip_length=4,
            token_drop=1,
            encoder_tiling=True,
            decoder_tiling=True,
            tile_size=16,
            tile_overlap_min=4,
            **caps,
        )
    for parameter in vae.parameters():
        parameter.data.normal_(0, 0.2)
    return vae.eval()


def _h3_install_gate(vae) -> VaeFastPathGate:
    """What minimax_h3_vae_cuda_opt does for the batched decode."""
    gate = VaeFastPathGate()
    vae._sgl_gate = gate
    register_vae_fast_path_gate(vae, gate)
    return gate
