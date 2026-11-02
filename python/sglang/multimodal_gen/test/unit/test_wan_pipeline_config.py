from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    WanT2V480PConfig,
    t5_postprocess_text,
)


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


def test_wan_text_padding_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_DIFFUSION_WAN_TEXT_PADDING", raising=False)
    assert WanT2V480PConfig().text_encoder_extra_args == [{}]


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("max", [{}]),
        ("longest", [{"padding": "longest"}]),
        ("bucket", [{"padding": "longest", "pad_to_multiple_of": 64}]),
    ],
)
def test_wan_text_padding_modes(monkeypatch, mode, expected):
    monkeypatch.setenv("SGLANG_DIFFUSION_WAN_TEXT_PADDING", mode)
    assert WanT2V480PConfig().text_encoder_extra_args == expected


def test_wan_text_padding_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_WAN_TEXT_PADDING", "tiny")
    with pytest.raises(ValueError):
        WanT2V480PConfig()


def test_wan_postprocess_keeps_embeds_512_and_mask_matches():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    outputs = SimpleNamespace(last_hidden_state=hidden, attention_mask=mask)

    result = t5_postprocess_text(outputs, None)

    assert result.prompt_embeds.shape == (2, 512, 3)
    assert torch.equal(result.prompt_embeds[0, :3], hidden[0, :3])
    assert not result.prompt_embeds[1, 2:].any()
    assert result.prompt_embeds_mask.shape == (2, 512)
    assert result.prompt_embeds_mask.sum(dim=1).tolist() == [3, 2]
    assert result.prompt_seq_lens == [3, 2]
