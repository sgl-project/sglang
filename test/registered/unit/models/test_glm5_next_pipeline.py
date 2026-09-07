"""GLM5 pipeline-stage ownership and native NextN weight loading."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration, Glm5NextModel
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_glm5_next_pp_embed_and_head_follow_stage_ownership(monkeypatch):
    first = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(first)
    first.model = nn.Module()
    first.model.embed_tokens = nn.Embedding(4, 3)
    first.lm_head = nn.Identity()
    first.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=False)

    last = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
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

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    new_embed = nn.Parameter(torch.randn(4, 3))
    new_head = nn.Parameter(torch.randn(4, 3))
    first.set_embed_and_head(new_embed, new_head)
    last.set_embed_and_head(new_embed, new_head)
    assert first.model.embed_tokens.weight is new_embed
    assert not hasattr(first.lm_head, "weight")
    assert last.lm_head.weight is new_head
    assert not hasattr(last.model.embed_tokens, "weight")


@pytest.mark.parametrize("first_k_dense_replace", [2, 4])
def test_glm5_next_tbo_stays_within_later_pp_stage(monkeypatch, first_k_dense_replace):
    class GuardedLayers(list):
        def __getitem__(self, index):
            if isinstance(index, int) and index < 0:
                raise AssertionError("TBO scatter mode crossed the PP stage boundary")
            return super().__getitem__(index)

    model = Glm5NextModel.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.start_layer = 4
    model.end_layer = 7
    model.first_k_dense_replace = first_k_dense_replace
    model.dflash_capture = False
    model.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    missing_layer = object()
    model.layers = GuardedLayers(
        [missing_layer] * model.start_layer
        + [nn.Identity() for _ in range(model.end_layer - model.start_layer)]
    )
    model.layers_to_capture = []
    model.enable_a2a_moe = False

    stage_entry_scatter_mode = object()
    tbo_calls = []

    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.BumpAllocator", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.ScatterMode.model_input_output",
        lambda: stage_entry_scatter_mode,
    )

    def capture_tbo(**kwargs):
        tbo_calls.append(kwargs)
        return kwargs["hidden_states"], kwargs["residual"]

    monkeypatch.setattr(
        "sglang.srt.models.glm5_next.model_forward_maybe_tbo", capture_tbo
    )

    hidden_states = torch.randn(2, 4)
    residual = torch.randn(2, 4)
    result = model.forward(
        input_ids=torch.empty(0, dtype=torch.int64),
        positions=torch.arange(2),
        forward_batch=SimpleNamespace(can_run_tbo=True),
        pp_proxy_tensors={
            "hidden_states": hidden_states,
            "residual": residual,
        },
    )

    assert len(tbo_calls) == 1
    assert tbo_calls[0]["layers"] == model.layers[model.start_layer : model.end_layer]
    assert all(layer is not missing_layer for layer in tbo_calls[0]["layers"])
    assert tbo_calls[0]["input_data_scatter_mode"] is stage_entry_scatter_mode
    assert result["hidden_states"] is hidden_states
    assert result["residual"] is residual


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
