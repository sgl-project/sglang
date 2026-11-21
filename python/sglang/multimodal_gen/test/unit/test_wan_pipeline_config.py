from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoConfig
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    FastWan2_1_T2V_480P_Config,
    TurboWanT2V480PConfig,
    Wan2_2_T2V_A14B_Config,
    WanT2V480PConfig,
    WanT2V720PConfig,
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


def test_wan_arch_text_len_follows_the_checkpoint_config():
    """A text_len in transformer/config.json must be the value the Wan model and
    tokenizer read, not the class default."""
    config = WanVideoConfig()
    config.update_model_arch({"text_len": 226})
    assert config.arch_config.text_len == 226


@pytest.mark.parametrize(
    "config_cls",
    [
        WanT2V480PConfig,
        WanT2V720PConfig,
        TurboWanT2V480PConfig,
        FastWan2_1_T2V_480P_Config,
        Wan2_2_T2V_A14B_Config,
    ],
)
def test_wan_t2v_configs_load_only_the_vae_decoder(config_cls):
    """Text-to-video never encodes with the VAE, so no T2V config may build the
    encoder; a subclass hook that replaces the parent hook brings it back."""
    vae_config = config_cls().vae_config
    assert vae_config.load_encoder is False
    assert vae_config.load_decoder is True
