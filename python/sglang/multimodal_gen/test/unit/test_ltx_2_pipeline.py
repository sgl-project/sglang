from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("pybase64")

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages import denoising as denoising_mod
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import LTX2VideoTransformer3DModel
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    LTX2TwoStagePipeline,
)


def _make_server_args(ltx_variant: str):
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(ltx_variant=ltx_variant)
            )
        )
    )


def test_ltx23_two_stage_merges_stage2_distilled_lora():
    assert (
        LTX2TwoStagePipeline._should_merge_stage2_distilled_lora(
            _make_server_args("ltx_2_3")
        )
        is True
    )


def test_ltx2_two_stage_keeps_stage2_distilled_lora_unmerged():
    assert (
        LTX2TwoStagePipeline._should_merge_stage2_distilled_lora(
            _make_server_args("ltx_2")
        )
        is False
    )


def test_ltx23_av_ca_gate_timestep_factor_matches_official_scaling():
    model = object.__new__(LTX2VideoTransformer3DModel)
    model.config = SimpleNamespace(arch_config=SimpleNamespace(ltx_variant="ltx_2_3"))
    model.av_ca_timestep_scale_multiplier = 1000
    model.timestep_scale_multiplier = 1000
    assert model._get_av_ca_gate_timestep_factor() == 1.0


def test_ltx2_av_ca_gate_timestep_factor_preserves_legacy_scaling():
    model = object.__new__(LTX2VideoTransformer3DModel)
    model.config = SimpleNamespace(arch_config=SimpleNamespace(ltx_variant="ltx_2"))
    model.av_ca_timestep_scale_multiplier = 1
    model.timestep_scale_multiplier = 1000
    assert model._get_av_ca_gate_timestep_factor() == 1.0


def test_sp_sharding_image_latent_keeps_main_video_metadata(monkeypatch):
    monkeypatch.setattr(denoising_mod, "get_sp_world_size", lambda: 2)

    calls = []

    def shard_latents_for_sp(batch, latents):
        calls.append(int(latents.shape[1]))
        if int(latents.shape[1]) == 8192:
            batch.sp_video_latent_num_frames = 2
            batch.sp_video_start_frame = 8
            batch.sp_video_tokens_per_frame = 4096
        else:
            batch.sp_video_latent_num_frames = 1
            batch.sp_video_start_frame = 0
            batch.sp_video_tokens_per_frame = 4096
        return latents[:, : latents.shape[1] // 2, :], True

    stage = object.__new__(DenoisingStage)
    batch = SimpleNamespace(
        latents=torch.zeros((1, 8192, 4)),
        image_latent=torch.zeros((1, 4096, 4)),
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(shard_latents_for_sp=shard_latents_for_sp)
    )

    DenoisingStage._preprocess_sp_latents(stage, batch, server_args)

    assert calls == [8192, 4096]
    assert batch.did_sp_shard_latents is True
    assert batch.sp_video_latent_num_frames == 2
    assert batch.sp_video_start_frame == 8
    assert batch.sp_video_tokens_per_frame == 4096
