from types import SimpleNamespace

import torch
from torch import nn

import sglang.multimodal_gen.runtime.models.encoders.qwen3vl as qwen3vl


class _IdentityAttention(nn.Module):
    def forward(self, query, key, value):
        return query


def test_qwen3vl_attention_uses_interleaved_mrope(monkeypatch):
    captured_kwargs = {}

    def build_rope(_config, **kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(qwen3vl, "build_qwen_vl_text_rope", build_rope)
    monkeypatch.setattr(
        qwen3vl, "_make_text_linear", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        qwen3vl, "_make_text_row_linear", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        qwen3vl, "_make_text_rms_norm", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(qwen3vl, "LocalAttention", lambda **kwargs: nn.Identity())
    config = SimpleNamespace(
        head_dim=8,
        hidden_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )

    qwen3vl.Qwen3VLTextAttention(config, layer_idx=0)

    assert captured_kwargs == {"mrope_interleaved": True}


def test_qwen3vl_attention_passes_three_axis_positions_to_srt_rope(monkeypatch):
    attention = qwen3vl.Qwen3VLTextAttention.__new__(qwen3vl.Qwen3VLTextAttention)
    nn.Module.__init__(attention)
    attention.q_proj = nn.Identity()
    attention.k_proj = nn.Identity()
    attention.v_proj = nn.Identity()
    attention.o_proj = nn.Identity()
    attention.q_norm = nn.Identity()
    attention.k_norm = nn.Identity()
    attention.head_dim = 4
    attention.rotary_emb = object()
    attention.attn = _IdentityAttention()

    captured_position_ids = None

    def apply_rope(_rotary_emb, position_ids, query, key):
        nonlocal captured_position_ids
        captured_position_ids = position_ids
        return query, key

    monkeypatch.setattr(qwen3vl, "apply_qwen_vl_text_rope", apply_rope)

    hidden_states = torch.randn(1, 2, 4)
    position_ids = torch.arange(6).view(3, 1, 2)
    output = attention(
        hidden_states,
        position_ids=position_ids,
        attention_mask=None,
    )

    assert captured_position_ids is position_ids
    torch.testing.assert_close(output, hidden_states)
