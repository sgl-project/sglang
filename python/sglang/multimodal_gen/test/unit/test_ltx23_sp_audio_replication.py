from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av import (
    LTX2AVDenoisingStage,
)


def test_ltx23_sp_audio_replication_stays_off_for_one_stage():
    batch = SimpleNamespace(
        image_latent=torch.randn(1, 8, 16),
        ltx2_num_image_tokens=8,
        did_sp_shard_latents=True,
        sp_video_start_frame=8,
    )
    server_args = SimpleNamespace(pipeline_class_name=None)

    assert LTX2AVDenoisingStage._should_apply_ltx2_ti2v(batch) is False

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av.get_sp_world_size",
        return_value=2,
    ):
        assert (
            LTX2AVDenoisingStage._should_replicate_ltx23_audio_for_sp(
                batch,
                server_args,
                is_ltx23_variant=True,
            )
            is False
        )


def test_ltx23_sp_audio_replication_stays_off_for_two_stage():
    batch = SimpleNamespace(
        image_latent=torch.randn(1, 8, 16),
        ltx2_num_image_tokens=8,
    )
    server_args = SimpleNamespace(pipeline_class_name="LTX2TwoStagePipeline")

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av.get_sp_world_size",
        return_value=2,
    ):
        assert (
            LTX2AVDenoisingStage._should_replicate_ltx23_audio_for_sp(
                batch,
                server_args,
                is_ltx23_variant=True,
            )
            is False
        )


def test_ltx23_forward_prepares_image_latent_before_sp_sharding():
    stage = object.__new__(LTX2AVDenoisingStage)
    batch = SimpleNamespace(
        image_path="dummy.png",
        image_latent=None,
        ltx2_num_image_tokens=0,
    )
    server_args = SimpleNamespace()

    class _StopAfterPrepare(Exception):
        pass

    def _fake_prepare_image_latent(batch, server_args):
        batch.image_latent = torch.ones(1, 8, 16)
        batch.ltx2_num_image_tokens = 8

    def _fake_prepare_denoising_loop(batch, server_args):
        assert batch.image_latent is not None
        assert batch.ltx2_num_image_tokens == 8
        raise _StopAfterPrepare

    stage._prepare_ltx2_image_latent = _fake_prepare_image_latent
    stage._prepare_denoising_loop = _fake_prepare_denoising_loop

    with pytest.raises(_StopAfterPrepare):
        LTX2AVDenoisingStage.forward(stage, batch, server_args)
