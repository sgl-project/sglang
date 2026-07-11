import sys

import torch
from torch import nn

from sglang.srt.layers.attention import vision
from sglang.srt.models.kimi_k25 import MoonViTEncoderLayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_vision_flash3_uses_precomputed_max_seqlen(monkeypatch):
    """A vision encoder can provide one host max-seqlen for all its blocks."""

    recorded = {}

    def fake_flash_attn(q, k, v, **kwargs):
        recorded.update(kwargs)
        return q

    monkeypatch.setattr(vision, "_is_cuda", True)
    monkeypatch.setattr(vision, "flash_attn_varlen_func", fake_flash_attn)

    attention = vision.VisionFlash3Attention(use_data_parallel=True)
    q = torch.zeros(3, 1, 8)
    cu_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32)
    output = attention(
        q,
        q,
        q,
        cu_seqlens=cu_seqlens,
        bsz=1,
        seq_len=3,
        max_seqlen=17,
    )

    assert output is q
    assert recorded["max_seqlen_q"] == 17
    assert recorded["max_seqlen_k"] == 17


def test_kimi_moonvit_forwards_one_precomputed_max_seqlen():
    """MoonViT must share its encoder-level scalar with each attention block."""

    recorded = {}

    class CapturingAttention(nn.Module):
        def forward(self, hidden_states, **kwargs):
            recorded.update(kwargs)
            return hidden_states

    layer = MoonViTEncoderLayer.__new__(MoonViTEncoderLayer)
    nn.Module.__init__(layer)
    layer.norm0 = nn.Identity()
    layer.norm1 = nn.Identity()
    layer.attn = CapturingAttention()
    layer.mlp = nn.Identity()

    hidden_states = torch.ones(3, 4)
    output = layer(
        hidden_states,
        cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        max_seqlen=19,
        rope_freqs_cis=torch.ones(3, 2, dtype=torch.complex64),
    )

    assert torch.equal(output, hidden_states * 4)
    assert recorded["max_seqlen"] == 19


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
