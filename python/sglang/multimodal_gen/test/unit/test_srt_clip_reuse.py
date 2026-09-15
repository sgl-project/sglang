from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.encoders import clip as mmgen_clip
from sglang.srt.models import clip as srt_clip


def _clip_config():
    return SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        layer_norm_eps=1e-5,
        hidden_act="quick_gelu",
        vocab_size=32,
        max_position_embeddings=8,
        eos_token_id=2,
        output_hidden_states=False,
        attention_dropout=0.0,
    )


class _FakeQKV(nn.Module):
    def forward(self, hidden_states):
        return torch.cat((hidden_states, hidden_states, hidden_states), dim=-1), None


class _FakeProjection(nn.Module):
    def forward(self, hidden_states):
        return hidden_states, None


def test_clip_encoder_propagates_causal_semantics():
    with (
        patch.object(srt_clip, "CLIPAttention", return_value=nn.Identity()) as attn,
        patch.object(srt_clip, "CLIPMLP", return_value=nn.Identity()),
    ):
        srt_clip.CLIPEncoder(_clip_config(), causal=True)

    assert attn.call_args.kwargs["causal"] is True


def test_mmgen_text_clip_requests_masked_srt_attention():
    with patch.object(mmgen_clip, "CLIPEncoder", return_value=nn.Identity()) as encoder:
        mmgen_clip.CLIPTextTransformer(_clip_config(), prefix="text_model.encoder")

    assert encoder.call_args.kwargs["causal"] is True


def test_clip_attention_separates_text_and_vision_semantics():
    parallel = SimpleNamespace(attn_tp_size=1, attn_tp_rank=0)
    hidden_states = torch.randn(2, 3, 16)
    padding_mask = torch.zeros(2, 1, 3, 3)

    with (
        patch.object(srt_clip, "get_parallel", return_value=parallel),
        patch.object(srt_clip, "QKVParallelLinear", return_value=_FakeQKV()),
        patch.object(srt_clip, "RowParallelLinear", return_value=_FakeProjection()),
        patch.object(
            srt_clip.F,
            "scaled_dot_product_attention",
            side_effect=lambda query, key, value, **kwargs: query,
        ) as sdpa,
    ):
        text_attention = srt_clip.CLIPAttention(_clip_config(), causal=True)
        vision_attention = srt_clip.CLIPAttention(_clip_config())
        text_attention(hidden_states)
        text_attention(hidden_states, attention_mask=padding_mask)
        vision_attention(hidden_states)

    assert sdpa.call_args_list[0].kwargs["is_causal"] is True
    assert sdpa.call_args_list[1].kwargs["is_causal"] is False
    assert sdpa.call_args_list[1].kwargs["attn_mask"] is padding_mask
    assert sdpa.call_args_list[2].kwargs["is_causal"] is False


def test_prepare_clip_attention_mask_combines_causal_and_padding_masks():
    mask = srt_clip.prepare_clip_attention_mask(
        torch.Size((1, 3)),
        torch.float32,
        torch.device("cpu"),
        torch.tensor([[1, 1, 0]]),
    )

    assert mask.shape == (1, 1, 3, 3)
    assert mask[0, 0, 0, 0] == 0
    assert mask[0, 0, 0, 1] < -1e20
    assert torch.all(mask[..., 2] < -1e20)


def test_prepare_clip_attention_mask_keeps_unmasked_fast_path():
    assert (
        srt_clip.prepare_clip_attention_mask(
            torch.Size((2, 3)), torch.float32, torch.device("cpu")
        )
        is None
    )


def test_srt_clip_weight_name_mapping():
    assert (
        mmgen_clip._srt_clip_param_name(
            "text_model.encoder.layers.0.self_attn.out_proj.weight"
        )
        == "text_model.encoder.layers.0.self_attn.proj.weight"
    )


def _encoder_loader_fixture(model_cls, root):
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    parent = model
    for component in root.split("."):
        child = nn.Module()
        parent.add_module(component, child)
        parent = child
    layer = nn.Module()
    layer.ffn = nn.Module()
    layer.ffn.fc1 = nn.Linear(2, 4)
    layer.ffn.fc2 = nn.Linear(4, 2)
    parent.layers = nn.ModuleList([layer])
    for param in model.parameters():
        nn.init.zeros_(param)
    return model


@pytest.mark.parametrize("checkpoint_member", ["mlp", "ffn"])
@pytest.mark.parametrize(
    "model_cls,root",
    [
        (mmgen_clip.CLIPTextModel, "text_model.encoder"),
        (mmgen_clip.CLIPTextModelWithProjection, "text_model.encoder"),
        (mmgen_clip.CLIPVisionModel, "vision_model.encoder"),
    ],
)
def test_mmgen_clip_loads_ffn_weights(model_cls, root, checkpoint_member):
    model = _encoder_loader_fixture(model_cls, root)
    model.config = SimpleNamespace(
        arch_config=SimpleNamespace(stacked_params_mapping=[])
    )
    expected = {
        name: torch.full_like(param, index + 1)
        for index, (name, param) in enumerate(model.named_parameters())
    }
    loaded = model.load_weights(
        (name.replace(".ffn.", f".{checkpoint_member}."), tensor)
        for name, tensor in expected.items()
    )
    assert loaded == set(expected)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected[name])


@pytest.mark.parametrize("checkpoint_member", ["mlp", "ffn"])
@pytest.mark.parametrize(
    "source_root",
    ["model.vision_tower.vision_model", "vision_tower", "vision_model"],
)
def test_mmgen_gemma3_loads_siglip_ffn_weights(source_root, checkpoint_member):
    from sglang.multimodal_gen.runtime.models.encoders.gemma_3 import (
        Gemma3ForConditionalGeneration,
    )

    root = "vision_tower.vision_model"
    model = _encoder_loader_fixture(Gemma3ForConditionalGeneration, f"{root}.encoder")
    expected = {
        name: torch.full_like(param, index + 1)
        for index, (name, param) in enumerate(model.named_parameters())
    }
    loaded = model.load_weights(
        (
            name.replace(root, source_root).replace(".ffn.", f".{checkpoint_member}."),
            tensor,
        )
        for name, tensor in expected.items()
    )
    assert loaded == set(expected)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected[name])
