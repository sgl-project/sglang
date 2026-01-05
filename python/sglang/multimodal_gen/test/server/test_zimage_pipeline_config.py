from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs import base as pipeline_base
from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig


@pytest.mark.parametrize("rank, expected_value", [(0, 1.0), (1, 0.0)])
def test_zimage_shard_latents_for_sp_5d(monkeypatch, rank, expected_value):
    config = ZImagePipelineConfig()
    latents = torch.ones(1, 4, 1, 2, 2)

    # Override SP helpers in the base module to avoid distributed initialization.
    monkeypatch.setattr(pipeline_base, "get_sp_world_size", lambda: 2)
    monkeypatch.setattr(pipeline_base, "get_sp_parallel_rank", lambda: rank)

    sharded, did_shard = config.shard_latents_for_sp(None, latents)

    assert did_shard is True
    assert sharded.shape == (1, 4, 1, 2, 2)
    assert torch.all(sharded == expected_value)


def test_zimage_gather_latents_for_sp_5d_uses_time_dim(monkeypatch):
    config = ZImagePipelineConfig()
    latents = torch.zeros(1, 4, 2, 2, 2)
    called = {}

    def fake_all_gather(tensor, dim=-1):
        called["dim"] = dim
        return tensor

    monkeypatch.setattr(
        pipeline_base, "sequence_model_parallel_all_gather", fake_all_gather
    )
    out = config.gather_latents_for_sp(latents)

    assert out is latents
    assert called["dim"] == 2


def test_zimage_post_denoising_loop_unpads_frames():
    config = ZImagePipelineConfig()
    latents = torch.zeros(1, 4, 2, 2, 2)
    latents[:, :, 1, :, :] = 1.0
    batch = SimpleNamespace(raw_latent_shape=(1, 4, 1, 2, 2))

    out = config.post_denoising_loop(latents, batch)

    assert out.shape == (1, 4, 2, 2)
    assert torch.all(out == 0)
