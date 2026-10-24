# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 SR checkpoint configuration and CLI request validation."""

import argparse
import re

import pytest
from kandinsky6_sr_tiny_components import TINY_DIT

from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    Kandinsky6SRArchConfig,
    Kandinsky6SRDitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.kandinsky6 import (
    Kandinsky6TI2VASamplingParams,
)
from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping


def parse(**changes):
    config = Kandinsky6SRDitConfig()
    config.update_model_arch(dict(TINY_DIT, **changes))
    return config.arch_config


# --------------------------------------------------------------------------- #
# DiT arch config
# --------------------------------------------------------------------------- #
def test_arch_config_parses_an_official_config_and_maps_checkpoint_names():
    arch = parse()
    assert arch.out_visual_dim == 12 and arch.attribute_overrides == {}
    assert arch.patch_size == (1, 2, 2) and arch.axes_dims == (8, 4, 4)
    assert (arch.hidden_size, arch.num_attention_heads) == (32, 2)
    assert arch.sr_scale_factor == {"512": [1.0, 1.0, 1.0]}  # JSON keys stay strings
    assert (arch.sr_visual_size, arch.sr_scheduler_scale, arch.sr_fps) == (
        [512],
        5.0,
        24,
    )

    map_name = get_param_names_mapping(arch.param_names_mapping)
    assert map_name("pooled_bias")[0] == "pooled_bias"
    assert (
        map_name("visual_transformer_blocks.3.feed_forward.net.0.proj.weight")[0]
        == "visual_transformer_blocks.3.feed_forward.mlp.fc_in.weight"
    )
    assert (
        map_name("visual_transformer_blocks.0.feed_forward.net.2.weight")[0]
        == "visual_transformer_blocks.0.feed_forward.mlp.fc_out.weight"
    )
    assert (
        map_name("time_embeddings.timestep_embedder.linear_1.weight")[0]
        == "time_embeddings.in_layer.weight"
    )
    # the attention output projection is not a feed-forward layer and keeps its name
    assert (
        map_name("visual_transformer_blocks.0.self_attention.out_layer.weight")[0]
        == "visual_transformer_blocks.0.self_attention.out_layer.weight"
    )


def test_unknown_sr_params_keys_are_rejected():
    with pytest.raises(ValueError, match="sr_params has unsupported keys"):
        parse(sr_params={"fps": 24, "new_knob": 1})


@pytest.mark.parametrize(
    "changes,error,flag",
    [
        ({"use_text": True}, NotImplementedError, "use_text"),
        ({"use_adapter": True}, NotImplementedError, "use_adapter"),
        ({"use_lq_modulation": True}, NotImplementedError, "use_lq_modulation"),
        ({"surprise_key": 1}, ValueError, "surprise_key"),
        ({"attribute_overrides": {"n_grid": 2}}, ValueError, "n_grid"),
        ({"attribute_overrides": {"instruct_type": "bogus"}}, ValueError, "bogus"),
        ({"piflow_eps": None}, ValueError, "piflow_eps"),
        ({"n_grid": 3}, ValueError, "n_grid"),
        ({"patch_size": [2, 2, 2]}, ValueError, "patch_size"),
        ({"sr_lq_noise_type": "gauss"}, ValueError, "gauss"),
    ],
)
def test_unsupported_or_unknown_config_fails_loudly_naming_the_key(
    changes, error, flag
):
    """A config the port cannot honor must never build a silently different model:
    unsupported flags raise NotImplementedError naming the flag, unknown / inconsistent
    keys raise ValueError (``update_model_arch`` alone would park unknown keys in
    ``extra_attrs``)."""
    with pytest.raises(error, match=flag):
        parse(**changes)


def test_sparse_attention_request_is_detected_from_attention_params():
    assert parse().requested_sparse_attention() is None
    assert (
        parse(
            attention_params={"512": {"type": "nabla", "P": 0.9}}
        ).requested_sparse_attention()
        == "nabla"
    )
    overridden = parse(
        attention_params={"512": {"type": "nabla"}},
        attribute_overrides={"attention_params": {"512": {"type": "flash"}}},
    )
    assert overridden.requested_sparse_attention() is None


def test_default_arch_config_is_valid_and_text_free():
    """Registry / ``PipelineConfig.from_kwargs`` instantiate the default config."""
    arch = Kandinsky6SRArchConfig()
    assert arch.use_text is False


# --------------------------------------------------------------------------- #
# VAE config
# --------------------------------------------------------------------------- #
def test_vae_config_parses_bundle_config_and_rejects_bad_ones():
    config = Kandinsky6SRVAEConfig()
    config.update_model_arch(
        dict(
            vae_type="video-kvae",
            encoder_config={"ch": 8},
            decoder_config={"ch": 8},
            scaling_factor=0.53,
            spatial_factor=16,
            temporal_factor=4,
        )
    )
    arch = config.arch_config
    assert arch.scaling_factor == 0.53
    assert (arch.spatial_compression_ratio, arch.temporal_compression_ratio) == (16, 4)
    assert config.get_vae_scale_factor() == 16

    with pytest.raises(ValueError, match="video-kvae"):
        Kandinsky6SRVAEConfig().update_model_arch({"vae_type": "wan"})
    with pytest.raises(ValueError, match="surprise"):
        Kandinsky6SRVAEConfig().update_model_arch({"surprise": 1})


# --------------------------------------------------------------------------- #
# Pipeline config and sampling params
# --------------------------------------------------------------------------- #
def test_pipeline_config_is_text_free_monolithic_and_strict():
    config = Kandinsky6SRPipelineConfig()
    config.check_pipeline_config()  # the four per-encoder tuples must stay consistent
    assert config.text_encoder_configs == ()
    assert config.supports_disaggregation() is False
    # These components must load strictly and re-raise (no diffusers AutoModel fallback).
    assert {"transformer", "vae", "latent_upscaler"} <= set(
        config.native_only_components
    )


def test_sampling_params_validate_the_sr_knobs():
    params = Kandinsky6SRSamplingParams(video_path="clip.mp4")
    assert (params.sr_resolution_scale, params.num_inference_steps) == (2.25, 4)
    assert (params.sr_tiles_batch_size, params.sr_tile_min_overlap) == (1, 0.20)
    assert params.seed == 42 and params.guidance_scale == 1.0

    for bad in (
        {"sr_resolution_scale": 3},
        {"num_inference_steps": 0},
        {"sr_tiles_batch_size": 0},
        {"sr_tile_min_overlap": 1.0},
        {"sr_target_resize_mode": "crop"},
        {"sr_target_resolution": "big"},
    ):
        with pytest.raises(ValueError):
            Kandinsky6SRSamplingParams(video_path="clip.mp4", **bad)
    for good in ("hd", "fullhd", "2k", "1280x720", "none"):
        Kandinsky6SRSamplingParams(video_path="clip.mp4", sr_target_resolution=good)


def test_sr_flags_reach_the_sampling_params_through_the_generate_cli():
    """New SamplingParams fields are only CLI flags if the base ``add_cli_args`` declares
    them; ``get_cli_args`` then filters by the resolved subclass' dataclass fields."""
    parser = argparse.ArgumentParser()
    SamplingParams.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--video-path",
            "clip.mp4",
            "--sr-resolution-scale",
            "4",
            "--num-inference-steps",
            "3",
            "--sr-tiles-batch-size",
            "2",
            "--sr-tile-min-overlap",
            "0.3",
            "--sr-target-resolution",
            "fullhd",
            "--sr-target-resize-mode",
            "exact",
        ]
    )
    kwargs = Kandinsky6SRSamplingParams.get_cli_args(args)
    params = Kandinsky6SRSamplingParams(**kwargs)
    assert (
        params.video_path == ["clip.mp4"] and params.source_video_path() == "clip.mp4"
    )
    assert (params.sr_resolution_scale, params.num_inference_steps) == (4.0, 3)
    assert (params.sr_tiles_batch_size, params.sr_tile_min_overlap) == (2, 0.3)
    assert (params.sr_target_resolution, params.sr_target_resize_mode) == (
        "fullhd",
        "exact",
    )
    # the sr_* knobs are ignored by other models' params instead of choking on them; the
    # shared num_inference_steps flag, unlike those, is a core field every model gets.
    assert "sr_tiles_batch_size" not in Kandinsky6TI2VASamplingParams.get_cli_args(args)
    assert "num_inference_steps" in Kandinsky6TI2VASamplingParams.get_cli_args(args)


def test_request_needs_exactly_one_video_and_names_its_output_after_it():
    with pytest.raises(ValueError, match="--video-path"):
        Kandinsky6SRSamplingParams()._validate_with_pipeline_config(
            Kandinsky6SRPipelineConfig()
        )
    with pytest.raises(ValueError, match="exactly one video"):
        Kandinsky6SRSamplingParams(video_path=["a.mp4", "b.mp4"]).source_video_path()

    params = Kandinsky6SRSamplingParams(video_path="/data/my clip.mp4", prompt=" ")
    params._set_output_file_name()
    assert re.fullmatch(r"my_clip_sr2\.25x_\d{8}-\d{6}\.mp4", params.output_file_name)
