from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs import base
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig


def test_wan_prompt_embed_accessors_return_transformer_tensor():
    prompt_embeds = torch.empty(1, 2, 3)
    negative_prompt_embeds = torch.empty(1, 2, 3)
    batch = SimpleNamespace(
        prompt_embeds=[prompt_embeds],
        negative_prompt_embeds=[negative_prompt_embeds],
    )

    config = WanT2V480PConfig()

    assert config.get_pos_prompt_embeds(batch) is prompt_embeds
    assert config.get_neg_prompt_embeds(batch) is negative_prompt_embeds


@pytest.mark.parametrize("enable_sequence_shard", [False, None])
def test_wan_nondivisible_frames_respect_sequence_shard_setting(
    monkeypatch, enable_sequence_shard
):
    """An explicit disable must not silently become token sharding."""
    monkeypatch.setattr(base, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(base, "get_sp_parallel_rank", lambda: 0)
    batch = SimpleNamespace(enable_sequence_shard=enable_sequence_shard)
    latents = torch.zeros(1, 1, 5, 2, 2)
    config = WanT2V480PConfig()

    if enable_sequence_shard is False:
        with pytest.raises(
            ValueError, match="sequence sharding is explicitly disabled"
        ):
            config.shard_latents_for_sp(batch, latents)
        assert batch.enable_sequence_shard is False
    else:
        result, did_shard = config.shard_latents_for_sp(batch, latents)
        assert result is latents
        assert not did_shard
        assert batch.enable_sequence_shard is True


def test_wan_divisible_frames_keep_frame_sharding(monkeypatch):
    monkeypatch.setattr(base, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(base, "get_sp_parallel_rank", lambda: 2)
    batch = SimpleNamespace(enable_sequence_shard=False)
    latents = torch.arange(8).reshape(1, 1, 8, 1, 1)

    result, did_shard = WanT2V480PConfig().shard_latents_for_sp(batch, latents)

    assert torch.equal(result, latents[:, :, 4:6])
    assert did_shard
    assert batch.enable_sequence_shard is False
