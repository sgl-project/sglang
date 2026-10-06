# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Kandinsky6ArchConfig's __post_init__ validation.

Pure dataclass/logic tests -- no GPU, no model weights. Ported from
FastVideo's ``fastvideo/tests/stages/test_kandinsky6_arch_config.py`` (the
reference implementation this native port was translated from) against the
sglang-diffusion config, which mirrors the diffusers
``Kandinsky6Transformer3DModel`` config 1:1.
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.configs.models.dits.kandinsky6 import (
    Kandinsky6ArchConfig,
    Kandinsky6VideoAudioConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_audio import (
    Kandinsky6AudioVAEConfig,
)

# transformer/config.json of kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers.
# Pro-sft-5s-Diffusers has out_visual_dim=16 and out_audio_dim=40 instead of the
# pi-Flow DX head widths below.
PRO_DISTILL_TRANSFORMER_CONFIG = {
    "audio_freqs_scaling": 0.144,
    "axes_dims": [32, 48, 48],
    "axes_dims_a": [32, 48, 48],
    "ca_rope": True,
    "cross_gates": True,
    "ff_dim": 16384,
    "ff_dim_a": 7168,
    "fix_modulation": True,
    "in_audio_dim": 40,
    "in_text_dim": 3584,
    "in_text_dim2": 768,
    "in_visual_dim": 16,
    "is_multimodal": True,
    "model_dim": 4096,
    "model_dim_a": 2048,
    "num_text_blocks": 4,
    "num_visual_blocks": 60,
    "out_audio_dim": 400,
    "out_visual_dim": 160,
    "patch_size": [1, 2, 2],
    "scale_factor": [1.0, 2.0, 2.0],
    "text_token_padding": True,
    "time_dim": 1024,
    "time_dim_a": 1024,
    "visual_cond": True,
    "visual_token_type_num_embeddings": 2,
}


def test_defaults_match_diffusers_reference():
    # Pins Kandinsky6ArchConfig to the actual Kandinsky6Transformer3DModel
    # defaults (verified against a real attention_engine="sdpa" checkpoint by
    # the FastVideo reference port) so a future edit can't silently drift.
    cfg = Kandinsky6ArchConfig()
    assert cfg.in_visual_dim == 16
    assert cfg.out_visual_dim == 16
    assert cfg.in_text_dim == 3584
    assert cfg.in_text_dim2 == 768
    assert cfg.time_dim == 1024
    assert cfg.model_dim == 4096
    assert cfg.ff_dim == 16384
    assert cfg.num_text_blocks == 4
    assert cfg.num_visual_blocks == 60
    assert cfg.axes_dims == (32, 48, 48)
    assert cfg.in_audio_dim == 20
    assert cfg.is_multimodal is True
    assert cfg.visual_cond is True


def test_audio_dims_default_to_matching_video_dims():
    cfg = Kandinsky6ArchConfig(
        model_dim=48, time_dim=32, ff_dim=128, axes_dims=(8, 8, 8)
    )
    assert cfg.model_dim_a == 48
    assert cfg.time_dim_a == 32
    assert cfg.ff_dim_a == 128
    assert cfg.axes_dims_a == (8, 8, 8)


def test_audio_dims_can_be_overridden_independently():
    cfg = Kandinsky6ArchConfig(
        model_dim=48,
        time_dim=32,
        ff_dim=128,
        axes_dims=(8, 8, 8),
        model_dim_a=48,
        time_dim_a=32,  # must match time_dim unless fix_modulation=True
        ff_dim_a=96,
        axes_dims_a=(4, 4, 4),
    )
    assert cfg.model_dim_a == 48
    assert cfg.ff_dim_a == 96
    assert cfg.axes_dims_a == (4, 4, 4)


def test_model_dim_a_not_divisible_by_head_dim_a_raises():
    with pytest.raises(ValueError, match="model_dim_a"):
        Kandinsky6ArchConfig(
            model_dim=48,
            time_dim=32,
            axes_dims=(8, 8, 8),
            model_dim_a=50,  # 50 % sum((4,4,4))=12 != 0
            time_dim_a=32,
            axes_dims_a=(4, 4, 4),
        )


def test_mismatched_time_dim_a_without_fix_modulation_raises():
    # Kandinsky6FusedTransformerDecoderBlock's cross-modal modulation is
    # driven by the *other* modality's time embedding by default
    # (fix_modulation=False) -- va_modulation is built from time_dim but
    # invoked with the audio time embedding, so time_dim_a must equal
    # time_dim or the fused block's modulation matmul shapes mismatch.
    with pytest.raises(ValueError, match="time_dim_a"):
        Kandinsky6ArchConfig(
            model_dim=48, time_dim=32, axes_dims=(8, 8, 8), time_dim_a=16
        )


def test_mismatched_time_dim_a_allowed_with_fix_modulation():
    cfg = Kandinsky6ArchConfig(
        model_dim=48,
        time_dim=32,
        axes_dims=(8, 8, 8),
        time_dim_a=16,
        fix_modulation=True,
    )
    assert cfg.time_dim_a == 16


def test_mismatched_time_dim_a_allowed_when_not_multimodal():
    # The cross-modal fused block doesn't exist at all when is_multimodal is
    # False, so the constraint shouldn't apply.
    cfg = Kandinsky6ArchConfig(
        model_dim=48,
        time_dim=32,
        axes_dims=(8, 8, 8),
        time_dim_a=16,
        is_multimodal=False,
    )
    assert cfg.time_dim_a == 16


def test_derived_arch_fields_match_dit_arch_config_contract():
    cfg = Kandinsky6ArchConfig(
        model_dim=48,
        time_dim=32,
        axes_dims=(8, 8, 8),
        in_visual_dim=6,
        out_visual_dim=6,
    )
    assert cfg.hidden_size == 48
    assert cfg.num_attention_heads == 48 // 24
    assert cfg.in_channels == 6
    assert cfg.out_channels == 6
    assert cfg.num_channels_latents == 6


def test_param_names_mapping_remaps_current_diffusers_names():
    import re

    cfg = Kandinsky6ArchConfig()
    mapping = cfg.param_names_mapping

    def _apply(key: str) -> str:
        for _ in range(len(mapping)):
            for pattern, repl in mapping.items():
                new_key, count = re.subn(pattern, repl, key)
                if count and new_key != key:
                    key = new_key
                    break
            else:
                return key
        return key

    assert (
        _apply("video_text_transformer_blocks.0.feed_forward.net.0.proj.weight")
        == "video_text_transformer_blocks.0.feed_forward.mlp.fc_in.weight"
    )
    assert (
        _apply("video_text_transformer_blocks.3.attn.to_q.weight")
        == "video_text_transformer_blocks.3.self_attention.to_q.weight"
    )
    assert (
        _apply("video_time_embeddings.timestep_embedder.linear_1.weight")
        == "video_time_embeddings.in_layer.weight"
    )
    assert (
        _apply("visual_transformer_blocks.0.video_dec_block.feed_forward.net.2.bias")
        == "visual_transformer_blocks.0.videoT.feed_forward.mlp.fc_out.bias"
    )


def test_pi_flow_distilled_head_widths_are_n_grid_times_the_latent_widths():
    # The distilled DiT predicts PiflowScheduler.n_grid (= 10) velocity grids per
    # token, so it outputs 10x the channels it takes in; the latents keep their width.
    config = Kandinsky6VideoAudioConfig()
    config.update_model_arch(dict(PRO_DISTILL_TRANSFORMER_CONFIG))
    arch = config.arch_config
    assert (arch.in_visual_dim, arch.out_visual_dim) == (16, 160)
    assert (arch.in_audio_dim, arch.out_audio_dim) == (40, 400)
    assert (arch.in_channels, arch.out_channels) == (16, 160)
    assert arch.num_channels_latents == 16


@pytest.mark.parametrize("scaling_factor", [0.5302, 0.417])
def test_audio_vae_scaling_factor_comes_from_the_checkpoint(scaling_factor):
    # audio_vae/config.json of Kandinsky-6.0-Pro-sft-5s-Diffusers (0.5302) and
    # Kandinsky-6.0-Pro-distill-5s-Diffusers (0.417): the two checkpoints normalize
    # their audio latents differently.
    config = Kandinsky6AudioVAEConfig()
    config.update_model_arch(
        {
            "mode": "44k",
            "sample_rate": 44100,
            "downsample_factor": 1024,
            "scaling_factor": scaling_factor,
        }
    )
    assert config.arch_config.scaling_factor == scaling_factor
