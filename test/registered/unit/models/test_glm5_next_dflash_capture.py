import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from sglang.srt.models.glm5_next_nextn import (
    Glm5NextForConditionalGenerationNextN,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("language_only", "vision_config", "expects_visual"),
    [
        (True, SimpleNamespace(), False),
        (False, None, False),
        (False, SimpleNamespace(), True),
    ],
)
def test_glm5_next_initializes_vision_only_with_vision_config(
    monkeypatch, language_only, vision_config, expects_visual
):
    text_config = SimpleNamespace(
        q_lora_rank=None,
        tie_word_embeddings=False,
        vocab_size=16,
        hidden_size=8,
        mhc=False,
        rope_scaling=None,
    )
    config = SimpleNamespace(
        text_config=text_config,
        vision_config=vision_config,
        encoder_only=False,
        language_only=language_only,
    )
    pp_group = SimpleNamespace(is_last_rank=False, world_size=1)
    parallel = SimpleNamespace(tp_size=1)
    mm = SimpleNamespace(mm_enable_dp_encoder=False)
    attn_tp_context = SimpleNamespace(init_context=lambda *_args: None)
    language_model = nn.Module()
    language_model.layers = []
    visual = nn.Identity()
    update_calls = []

    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.vision_utils.update_vit_attn_dummy_heads_config",
        update_calls.append,
    )
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.Glm5NextVisionModel",
        lambda *_args, **_kwargs: visual,
    )
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.Glm5NextModel",
        lambda *_args, **_kwargs: language_model,
    )
    monkeypatch.setattr(
        Glm5NextForConditionalGeneration,
        "determine_num_fused_shared_experts",
        lambda self: setattr(self, "num_fused_shared_experts", 0),
    )
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.LogitsProcessor",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.PPMissingLayer", nn.Identity
    )
    monkeypatch.setattr("sglang.srt.models.glm5_next.get_pp_group", lambda: pp_group)
    monkeypatch.setattr("sglang.srt.models.glm5_next.get_parallel", lambda: parallel)
    monkeypatch.setattr("sglang.srt.models.glm5_next.get_mm", lambda: mm)
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.get_attn_tp_context", lambda: attn_tp_context
    )
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.is_deepseek_dsa", lambda _config: False
    )
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.is_dsa_enable_prefill_cp", lambda: False
    )
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.is_prefill_context_parallel_enabled",
        lambda: False,
    )

    model = Glm5NextForConditionalGeneration(config)

    assert model.language_only == (language_only or vision_config is None)
    assert model.visual is (visual if expects_visual else None)
    assert update_calls == ([config] if expects_visual else [])


def test_glm5_next_dflash_contracts_mhc_hidden_state():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = True

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    residual = torch.full_like(hidden_states, 2)

    actual = model._prepare_aux_hidden_state(hidden_states, residual)
    expected = (hidden_states + residual).unflatten(-1, (4, -1)).mean(dim=-2)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3)


def test_glm5_next_eagle_capture_keeps_mhc_hidden_state():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = False

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    residual = torch.full_like(hidden_states, 2)

    actual = model._prepare_aux_hidden_state(hidden_states, residual)

    torch.testing.assert_close(actual, hidden_states + residual)


def test_glm5_next_dflash_contracts_mhc_hidden_state_without_residual():
    # GLM-5.3-Flash runs with mhc=True, where MHCLayerCommunicator folds the
    # residual into the widened hidden state and returns residual=None.
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=True, hc_mult=4)
    model.dflash_capture = True

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    actual = model._prepare_aux_hidden_state(hidden_states, None)
    expected = hidden_states.unflatten(-1, (4, -1)).mean(dim=-2)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3)


def test_glm5_next_eagle_capture_without_residual():
    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(mhc=False, hc_mult=1)
    model.dflash_capture = False

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

    actual = model._prepare_aux_hidden_state(hidden_states, None)

    torch.testing.assert_close(actual, hidden_states)


def test_glm5_next_dflash_maps_target_layers_to_capture_points():
    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.model = SimpleNamespace(dflash_capture=False, layers_to_capture=[])
    model.capture_aux_hidden_states = False

    model.set_dflash_layers_to_capture([5, 14, 24, 33, 42])

    assert model.capture_aux_hidden_states
    assert model.model.dflash_capture
    assert model.model.layers_to_capture == [6, 15, 25, 34, 43]


def test_glm5_next_pp_embed_and_head_follow_stage_ownership():
    embed = nn.Parameter(torch.randn(4, 3))
    head = nn.Parameter(torch.randn(4, 3))

    first = Glm5NextForConditionalGeneration.__new__(
        Glm5NextForConditionalGeneration
    )
    nn.Module.__init__(first)
    first.model = nn.Module()
    first.model.embed_tokens = nn.Embedding(4, 3)
    first.lm_head = nn.Identity()
    first.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=False)

    last = Glm5NextForConditionalGeneration.__new__(
        Glm5NextForConditionalGeneration
    )
    nn.Module.__init__(last)
    last.model = nn.Module()
    last.model.embed_tokens = nn.Identity()
    last.lm_head = nn.Linear(3, 4, bias=False)
    last.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)

    first_embed, first_head = first.get_embed_and_head()
    last_embed, last_head = last.get_embed_and_head()

    assert first_embed is first.model.embed_tokens.weight
    assert first_head is None
    assert last_embed is None
    assert last_head is last.lm_head.weight


def test_glm5_nextn_loader_loads_checkpoint_embedding(monkeypatch):
    weights = [
        (
            "model.language_model.embed_tokens.weight",
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ),
        ("lm_head.weight", torch.ones(2, 2)),
    ]

    model = Glm5NextForConditionalGenerationNextN.__new__(
        Glm5NextForConditionalGenerationNextN
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        n_routed_experts=0,
        q_lora_rank=None,
    )
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(2, 2)
    model.model.decoder = nn.Module()
    model.quant_config = None
    model.num_fused_shared_experts = 0
    model.encoder_only = False
    model.language_only = True
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.DeepseekV2WeightLoaderMixin.post_load_weights",
        lambda *_args, **_kwargs: None,
    )

    model.load_weights(iter(weights))

    torch.testing.assert_close(model.model.embed_tokens.weight, weights[0][1])


def test_glm5_nextn_keeps_own_embedding_when_target_pp_stage_has_none(monkeypatch):
    original_embed = nn.Parameter(torch.randn(4, 3))
    target_head = nn.Parameter(torch.randn(4, 3))
    model = Glm5NextForConditionalGenerationNextN.__new__(
        Glm5NextForConditionalGenerationNextN
    )
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(4, 3)
    model.model.embed_tokens.weight = original_embed
    model.lm_head = nn.Linear(3, 4, bias=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    model.set_embed_and_head(None, target_head)

    assert model.model.embed_tokens.weight is original_embed
    assert model.lm_head.weight is target_head


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
