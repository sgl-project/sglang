# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 SR configs, request parameters and model routing (no weights, no GPU)."""

import argparse
import json
import re

import pytest

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    Kandinsky6SRArchConfig,
    Kandinsky6SRDitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
    _is_kandinsky6_t2va,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
    _is_kandinsky6_sr,
)
from sglang.multimodal_gen.configs.sample.kandinsky6 import (
    Kandinsky6TI2VASamplingParams,
)
from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines.kandinsky6_pipeline import (
    Kandinsky6TI2VAPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.kandinsky6_sr_pipeline import (
    Kandinsky6SRPipeline,
)

TINY_DIT = dict(
    in_visual_dim=4,
    in_text_dim=8,
    in_text_dim2=8,
    time_dim=16,
    out_visual_dim=4,
    patch_size=[1, 2, 2],
    model_dim=32,
    ff_dim=64,
    num_text_blocks=0,
    num_visual_blocks=2,
    axes_dims=[8, 4, 4],
    visual_cond=False,
    instruct_type="noise",
    attention_params={"512": {"type": "flash"}},
    use_text=False,
)
TINY_PIFLOW = dict(
    nfe=2,
    num_policy_substeps=8,
    final_step_size_scale=0.5,
    shift=5.0,
    n_grid=3,
    eps=1e-6,
)
SR_PARAMS = dict(
    scale_factor={"512": [1.0, 1.0, 1.0]},
    visual_size=[512],
    scheduler_scale=5.0,
    lq_noise_scale=0.7,
    lq_noise_type="ddpm",
    lq_channel_noise_scale=0.0,
    cap_noise_timestep=False,
    fps=24,
)


def flat_config(**changes):
    """A converted ``transformer/config.json`` (pi-Flow DX head) with ``changes`` applied."""
    flat = dict(TINY_DIT, _class_name="Kandinsky6SRTransformer3DModel", n_grid=3)
    flat.update(
        piflow_nfe=2,
        piflow_num_policy_substeps=8,
        piflow_final_step_size_scale=0.5,
        piflow_shift=5.0,
        piflow_eps=1e-6,
        attribute_overrides={"instruct_type": "noise"},
        sr_visual_size=[512],
        sr_scale_factor={"512": [1.0, 1.0, 1.0]},
        sr_scheduler_scale=5.0,
        sr_lq_noise_scale=0.7,
        sr_lq_noise_type="ddpm",
        sr_lq_channel_noise_scale=0.0,
        sr_cap_noise_timestep=False,
        sr_fps=24,
    )
    flat.update(changes)
    return flat


def official_config(**changes):
    """An official ``transformer/config.json``: constructor kwargs, ``sr_params``, the TOTAL DX head
    width in ``out_visual_dim`` (``base * n_grid``) and no ``n_grid`` / ``piflow_*`` keys.
    """
    official = {
        key: value for key, value in TINY_DIT.items() if key not in ("out_visual_dim",)
    }
    official.update(
        out_visual_dim=TINY_DIT["out_visual_dim"] * TINY_PIFLOW["n_grid"],
        attribute_overrides=None,
        sr_params=dict(SR_PARAMS),
    )
    official.update(changes)
    return official


def parse(flat):
    config = Kandinsky6SRDitConfig()
    config.update_model_arch(flat)
    return config.arch_config


# --------------------------------------------------------------------------- #
# DiT arch config
# --------------------------------------------------------------------------- #
def test_arch_config_parses_an_official_config_and_maps_checkpoint_names():
    arch = parse(official_config())
    assert arch.n_grid == 1 and not arch.is_piflow  # pi-Flow lives in the scheduler
    assert arch.out_visual_dim == 12 and arch.attribute_overrides == {}
    assert arch.patch_size == (1, 2, 2) and arch.axes_dims == (8, 4, 4)
    assert (arch.hidden_size, arch.num_attention_heads) == (32, 2)
    assert arch.sr_scale_factor == {"512": [1.0, 1.0, 1.0]}  # JSON keys stay strings
    assert (arch.sr_visual_size, arch.sr_scheduler_scale, arch.sr_fps) == (
        [512],
        5.0,
        24,
    )

    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

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
        parse(official_config(sr_params={"fps": 24, "new_knob": 1}))


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
        parse(flat_config(**changes))


def test_sparse_attention_request_is_detected_from_attention_params():
    assert parse(flat_config()).requested_sparse_attention() is None
    nabla = flat_config(attention_params={"512": {"type": "nabla", "P": 0.9}})
    assert parse(nabla).requested_sparse_attention() == "nabla"
    # an override can switch it off again
    overridden = flat_config(
        attention_params={"512": {"type": "nabla"}},
        attribute_overrides={"attention_params": {"512": {"type": "flash"}}},
    )
    assert parse(overridden).requested_sparse_attention() is None


def test_default_arch_config_is_valid_and_text_free():
    """Registry / ``PipelineConfig.from_kwargs`` instantiate the default config."""
    arch = Kandinsky6SRArchConfig()
    assert arch.use_text is False and not arch.is_piflow and arch.n_grid == 1


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


# --------------------------------------------------------------------------- #
# Registry routing
# --------------------------------------------------------------------------- #
SR_IDS = [
    "Kandinsky6SRPipeline",
    "/models/kandinsky6-sr-native",
    "/models/Kandinsky_6.0_SR_x2",
    "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers",
    "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers",
    "/models/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers",
]
T2VA_IDS = [
    "kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers",
    "/models/Kandinsky-6.0-Pro-sft-5s-Diffusers",
    "Kandinsky6TI2VAPipeline",
    "Kandinsky6T2VAPipeline",
]


def test_kandinsky6_detectors_route_sr_and_t2va_ids_apart():
    """Both families contain "kandinsky6": the SR detector is narrow and the T2VA one
    excludes it, so an SR path can never match (or double-match) the T2VA configs."""
    for model_id in SR_IDS:
        assert _is_kandinsky6_sr(model_id), model_id
        assert not _is_kandinsky6_t2va(model_id), model_id
    for model_id in T2VA_IDS:
        assert _is_kandinsky6_t2va(model_id), model_id
        assert not _is_kandinsky6_sr(model_id), model_id


def _write_model_index(directory, class_name):
    """Minimal bundle skeleton that passes ``verify_model_config_and_directory``."""
    for component in ("transformer", "vae"):
        (directory / component).mkdir(parents=True, exist_ok=True)
    (directory / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": class_name,
                "_diffusers_version": "0.37.0",
                "transformer": ["diffusers", "X"],
                "vae": ["diffusers", "Y"],
            }
        )
    )
    return str(directory)


def test_model_info_resolves_sr_bundles_to_sr_and_t2va_paths_to_k6(tmp_path):
    """The pipeline class comes from model_index ``_class_name``, the configs from the
    registered detectors: a bundle must get *both* halves from the same family."""
    sr_dir = _write_model_index(tmp_path / "vsr_bundle", "Kandinsky6SRPipeline")
    info = registry.get_model_info(sr_dir, backend="sglang")
    assert info.pipeline_cls is Kandinsky6SRPipeline
    assert info.pipeline_config_cls is Kandinsky6SRPipelineConfig
    assert info.sampling_param_cls is Kandinsky6SRSamplingParams

    k6_dir = _write_model_index(
        tmp_path / "Kandinsky-6.0-Pro-sft-5s-Diffusers-local", "Kandinsky6TI2VAPipeline"
    )
    info = registry.get_model_info(k6_dir, backend="sglang")
    assert info.pipeline_cls is Kandinsky6TI2VAPipeline
    assert info.pipeline_config_cls is Kandinsky6TI2VAPipelineConfig
    assert info.sampling_param_cls is Kandinsky6TI2VASamplingParams


# --------------------------------------------------------------------------- #
# Official config -> sampling spec (pi-Flow comes from the scheduler component)
# --------------------------------------------------------------------------- #
def _spec(arch, scheduler):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
        build_sampling_spec,
    )

    return build_sampling_spec(
        arch=arch,
        tiling_scale=2,
        seed=42,
        num_steps=5,
        tiles_batch_size=1,
        tile_min_overlap=0.2,
        scheduler=scheduler,
    )


def test_sampling_spec_takes_the_pi_flow_sampler_from_the_scheduler_component():
    from sglang.multimodal_gen.runtime.models.schedulers.kandinsky6_piflow import (
        PiflowScheduler,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
        PiflowParams,
    )

    arch = parse(official_config())
    scheduler = PiflowScheduler(
        nfe=2,
        n_grid=3,
        shift=5.0,
        eps=1e-6,
        final_step_size_scale=0.5,
        num_policy_substeps=8,
    )
    spec = _spec(arch, scheduler)
    assert spec.piflow == PiflowParams(
        nfe=2,
        num_policy_substeps=8,
        final_step_size_scale=0.5,
        shift=5.0,
        n_grid=3,
        eps=1e-6,
    )
    assert spec.scheduler_scale == 5.0 and spec.visual_size == 512

    # no pi-Flow section (nfe null) or no scheduler at all: flow-Euler, which a
    # DX-wide DiT head must never run
    for scheduler in (PiflowScheduler(n_grid=3, nfe=None), None):
        with pytest.raises(ValueError, match="flow-Euler"):
            _spec(arch, scheduler)
    assert _spec(parse(official_config(out_visual_dim=4)), None).piflow is None


def test_legacy_flat_config_still_carries_its_own_pi_flow_sampler():
    spec = _spec(parse(flat_config()), None)
    assert spec.piflow is not None and spec.piflow.n_grid == 3 and spec.piflow.nfe == 2


@pytest.mark.parametrize(
    "repo_id",
    [
        "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers",
        "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers",
    ],
)
def test_the_release_repo_ids_are_registered(repo_id, monkeypatch):
    """Both official VSR repos (flow-matching and 2-step distilled) get the SR
    classes; the Hub is not asked for the model index."""
    registry.get_model_info.cache_clear()
    monkeypatch.setattr(
        registry,
        "maybe_download_model_index",
        lambda model_path: {"_class_name": "Kandinsky6SRPipeline"},
    )
    info = registry.get_model_info(repo_id)
    assert info.pipeline_cls is Kandinsky6SRPipeline
    assert info.pipeline_config_cls is Kandinsky6SRPipelineConfig
    assert info.sampling_param_cls is Kandinsky6SRSamplingParams
