# SPDX-License-Identifier: Apache-2.0
"""Checkpoint architecture validation and native parameter-name mapping."""

import pytest

from sglang.multimodal_gen.configs.models.dits.kandinsky6 import (
    Kandinsky6ArchConfig,
    Kandinsky6VideoAudioConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_audio import (
    Kandinsky6AudioVAEConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, (48, 32, 128, (8, 8, 8))),
        ({"ff_dim_a": 96, "axes_dims_a": (4, 4, 4)}, (48, 32, 96, (4, 4, 4))),
        ({"time_dim_a": 16, "fix_modulation": True}, (48, 16, 128, (8, 8, 8))),
        ({"time_dim_a": 16, "is_multimodal": False}, (48, 16, 128, (8, 8, 8))),
    ],
)
def test_audio_dimension_inheritance_and_overrides(overrides, expected):
    arch = Kandinsky6ArchConfig(
        model_dim=48, time_dim=32, ff_dim=128, axes_dims=(8, 8, 8), **overrides
    )
    assert (
        arch.model_dim_a,
        arch.time_dim_a,
        arch.ff_dim_a,
        arch.axes_dims_a,
    ) == expected
    assert (arch.hidden_size, arch.num_attention_heads) == (48, 2)


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"model_dim_a": 50, "axes_dims_a": (4, 4, 4)}, "model_dim_a"),
        ({"time_dim_a": 16}, "time_dim_a"),
    ],
)
def test_incompatible_audio_dimensions_are_rejected(overrides, error):
    with pytest.raises(ValueError, match=error):
        Kandinsky6ArchConfig(
            model_dim=48, time_dim=32, axes_dims=(8, 8, 8), **overrides
        )


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "video_text_transformer_blocks.0.feed_forward.net.0.proj.weight",
            "video_text_transformer_blocks.0.feed_forward.mlp.fc_in.weight",
        ),
        (
            "video_text_transformer_blocks.3.attn.to_q.weight",
            "video_text_transformer_blocks.3.self_attention.to_q.weight",
        ),
        (
            "video_time_embeddings.timestep_embedder.linear_1.weight",
            "video_time_embeddings.in_layer.weight",
        ),
        (
            "visual_transformer_blocks.0.video_dec_block.feed_forward.net.2.bias",
            "visual_transformer_blocks.0.videoT.feed_forward.mlp.fc_out.bias",
        ),
    ],
)
def test_checkpoint_name_mapping(source, target):
    map_name = get_param_names_mapping(Kandinsky6ArchConfig().param_names_mapping)
    assert map_name(source)[0] == target


@pytest.mark.parametrize("n_grid", [1, 10], ids=["sft", "distilled"])
def test_checkpoint_head_width_does_not_change_latent_width(n_grid):
    config = Kandinsky6VideoAudioConfig()
    config.update_model_arch(
        {
            "in_visual_dim": 16,
            "out_visual_dim": 16 * n_grid,
            "in_audio_dim": 40,
            "out_audio_dim": 40 * n_grid,
            "model_dim_a": 2048,
            "ff_dim_a": 7168,
            "fix_modulation": True,
            "scale_factor": [1.0, 2.0, 2.0],
        }
    )
    arch = config.arch_config
    assert (arch.in_channels, arch.out_channels, arch.num_channels_latents) == (
        16,
        16 * n_grid,
        16,
    )
    assert (arch.in_audio_dim, arch.out_audio_dim) == (40, 40 * n_grid)
    assert tuple(arch.scale_factor) == (1.0, 2.0, 2.0)


@pytest.mark.parametrize("scaling_factor", [0.5302, 0.417])
def test_audio_vae_scaling_factor_comes_from_the_checkpoint(scaling_factor):
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
