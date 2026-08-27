from types import SimpleNamespace

import pytest
import torch
from torch import nn

import sglang.multimodal_gen.runtime.models.encoders.qwen_vl_rope as qwen_vl_rope
import sglang.srt.layers.rotary_embedding.base as rope_base
import sglang.srt.layers.rotary_embedding.factory as rope_factory
from sglang.multimodal_gen.runtime.models.encoders.qwen_vl_rope import (
    apply_qwen_vl_text_rope,
    build_qwen_vl_text_rope,
)


class _RecordingRotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.positions = None
        self.query_shape = None
        self.key_shape = None

    def forward_native(self, positions, query, key):
        self.positions = positions
        self.query_shape = query.shape
        self.key_shape = key.shape
        return query + 1, key + 2


def test_qwen_vl_rope_supports_transformers_v5_config(monkeypatch):
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": 1_000_000.0,
        "mrope_section": [2, 1, 1],
    }
    config = SimpleNamespace(
        head_dim=None,
        hidden_size=32,
        num_attention_heads=4,
        max_position_embeddings=128,
        rope_parameters=rope_parameters,
    )
    captured_kwargs = {}
    rotary_emb = object()

    def get_rope(**kwargs):
        captured_kwargs.update(kwargs)
        return rotary_emb

    monkeypatch.setattr(qwen_vl_rope, "get_rope", get_rope)

    assert build_qwen_vl_text_rope(config) is rotary_emb
    assert captured_kwargs == {
        "head_size": 8,
        "rotary_dim": 8,
        "max_position": 128,
        "base": 1_000_000.0,
        "is_neox_style": True,
        "rope_scaling": {**rope_parameters, "mrope_interleaved": False},
    }


def test_qwen_vl_rope_enables_interleaved_layout_explicitly(monkeypatch):
    config = SimpleNamespace(
        head_dim=8,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "mrope_section": [2, 1, 1],
        },
    )
    captured_kwargs = {}

    def get_rope(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(qwen_vl_rope, "get_rope", get_rope)

    build_qwen_vl_text_rope(config, mrope_interleaved=True)

    assert captured_kwargs["rope_scaling"] == {
        **config.rope_parameters,
        "mrope_interleaved": True,
    }


def test_qwen_vl_rope_does_not_require_srt_runtime_context(monkeypatch):
    def fail_get_exec():
        raise AssertionError("Qwen-VL RoPE must not read an unpublished context")

    monkeypatch.setattr(rope_base, "get_exec", fail_get_exec)
    monkeypatch.setattr(rope_base, "publish_role", lambda: None)
    monkeypatch.setattr(rope_factory, "_ROPE_DICT", {})
    config = SimpleNamespace(
        head_dim=None,
        hidden_size=32,
        num_attention_heads=4,
        max_position_embeddings=37,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 123_457.0,
            "mrope_section": [2, 1, 1],
        },
    )

    rotary_emb = build_qwen_vl_text_rope(config)
    positions = torch.arange(9).view(3, 3)
    query = torch.randn(3, 16)
    key = torch.randn(3, 8)

    rotated_query, rotated_key = rotary_emb.forward_native(positions, query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape


def test_qwen_vl_rope_adapts_batched_gqa_layout():
    rotary_emb = _RecordingRotaryEmbedding()
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 2, 5, 8)
    position_ids = torch.arange(30).view(3, 2, 5)

    rotated_query, rotated_key = apply_qwen_vl_text_rope(
        rotary_emb, position_ids, query, key
    )

    assert rotary_emb.query_shape == (10, 32)
    assert rotary_emb.key_shape == (10, 16)
    assert torch.equal(rotary_emb.positions, position_ids.reshape(3, -1))
    torch.testing.assert_close(rotated_query, query + 1)
    torch.testing.assert_close(rotated_key, key + 2)


@pytest.mark.parametrize(
    ("position_ids", "key"),
    [
        (torch.zeros(2, 1, 3, dtype=torch.long), torch.zeros(1, 1, 3, 4)),
        (torch.zeros(3, 1, 2, dtype=torch.long), torch.zeros(1, 1, 3, 4)),
    ],
)
def test_qwen_vl_rope_rejects_incompatible_shapes(position_ids, key):
    with pytest.raises(ValueError):
        apply_qwen_vl_text_rope(
            _RecordingRotaryEmbedding(),
            position_ids,
            torch.zeros(1, 1, 3, 4),
            key,
        )
