import sys

import torch
from torch import nn

from sglang.srt.layers.attention import vision
from sglang.srt.models.kimi_k25 import MoonViT3dEncoder, MoonViTEncoderLayer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_vision_flash3_uses_precomputed_max_seqlen(monkeypatch):
    """A vision encoder can provide one host max-seqlen for all its blocks."""

    recorded = {}

    def fake_flash_attn(q, k, v, **kwargs):
        recorded.update(kwargs)
        return q

    monkeypatch.setattr(vision, "_is_cuda", True)
    # This symbol is imported only on CUDA/MUSA hosts; inject the stub on the
    # CPU CI path too so the backend-selection behavior stays unit-testable.
    monkeypatch.setattr(
        vision, "flash_attn_varlen_func", fake_flash_attn, raising=False
    )
    monkeypatch.setattr(vision, "flash_attn_func", fake_flash_attn, raising=False)

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


def test_vision_triton_uses_precomputed_max_seqlen(monkeypatch):
    """Triton vision attention must share the encoder-level host scalar."""

    recorded = {}

    def fake_context_attention(q, k, v, output, *args, **kwargs):
        recorded["max_seqlen"] = args[2]
        recorded["sequence_lengths"] = args[1]
        output.copy_(q)

    monkeypatch.setattr(vision, "context_attention_fwd", fake_context_attention)

    attention = vision.VisionTritonAttention(use_data_parallel=True)
    q = torch.zeros(3, 1, 8)
    cu_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32)
    sequence_lengths = torch.tensor([1, 2], dtype=torch.int32)
    output = attention(
        q,
        q,
        q,
        cu_seqlens=cu_seqlens,
        bsz=1,
        seq_len=3,
        max_seqlen=17,
        sequence_lengths=sequence_lengths,
    )

    assert torch.equal(output, q)
    assert recorded["max_seqlen"] == 17
    assert recorded["sequence_lengths"] is sequence_lengths


def test_vision_flash4_uses_precomputed_max_seqlen(monkeypatch):
    """FA4 must not re-synchronize for every vision transformer layer."""

    recorded = {}

    def fake_flash_attn(q, k, v, **kwargs):
        recorded.update(kwargs)
        return q

    monkeypatch.setattr(vision, "_is_cuda", True)
    monkeypatch.setattr(
        vision, "flash_attn_varlen_func", fake_flash_attn, raising=False
    )
    monkeypatch.setattr(vision, "flash_attn_func", fake_flash_attn, raising=False)

    attention = vision.VisionFlash4Attention(use_data_parallel=True)
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


def test_kimi_moonvit_precomputes_sequence_lengths_once():
    """MoonViT shares packed sequence metadata across all attention blocks."""

    recorded = {}

    class CapturingRope:
        def get_freqs_cis(self, grid_thws, device):
            return torch.ones(7, 2, dtype=torch.complex64, device=device)

    class CapturingBlock(nn.Module):
        def forward(
            self,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            rope_freqs_cis,
            sequence_lengths,
            forward_metadata=None,
        ):
            recorded["cu_seqlens"] = cu_seqlens
            recorded["max_seqlen"] = max_seqlen
            recorded["sequence_lengths"] = sequence_lengths
            return hidden_states

    encoder = MoonViT3dEncoder.__new__(MoonViT3dEncoder)
    nn.Module.__init__(encoder)
    encoder.rope_2d = CapturingRope()
    encoder.blocks = nn.ModuleList([CapturingBlock()])
    encoder.final_layernorm = nn.Identity()

    hidden_states = torch.ones(7, 4)
    grid_thws = torch.tensor([[1, 1, 3], [1, 2, 2]], dtype=torch.int32)
    output = encoder(hidden_states, grid_thws)

    assert torch.equal(output, hidden_states)
    assert torch.equal(recorded["sequence_lengths"], torch.tensor([3, 4]))
    assert torch.equal(recorded["cu_seqlens"], torch.tensor([0, 3, 7]))
    assert recorded["max_seqlen"] == 4


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
