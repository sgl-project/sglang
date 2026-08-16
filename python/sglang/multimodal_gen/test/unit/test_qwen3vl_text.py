import torch
from torch import nn

import sglang.multimodal_gen.runtime.models.encoders.qwen3vl as qwen3vl


class _IdentityAttention(nn.Module):
    def forward(self, query, key, value):
        return query


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
