import sys

import pytest
import torch
from torch import nn

from sglang.srt.layers.attention import vision
from sglang.srt.models import kimi_k25
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
        ):
            recorded["cu_seqlens"] = cu_seqlens
            recorded["max_seqlen"] = max_seqlen
            recorded["sequence_lengths"] = sequence_lengths
            return hidden_states

    encoder = MoonViT3dEncoder.__new__(MoonViT3dEncoder)
    nn.Module.__init__(encoder)
    encoder.rope_2d = CapturingRope()
    encoder.use_fused_rope = False
    encoder.blocks = nn.ModuleList([CapturingBlock()])
    encoder.final_layernorm = nn.Identity()

    hidden_states = torch.ones(7, 4)
    grid_thws = torch.tensor([[1, 1, 3], [1, 2, 2]], dtype=torch.int32)
    output = encoder(hidden_states, grid_thws)

    assert torch.equal(output, hidden_states)
    assert torch.equal(recorded["sequence_lengths"], torch.tensor([3, 4]))
    assert torch.equal(recorded["cu_seqlens"], torch.tensor([0, 3, 7]))
    assert recorded["max_seqlen"] == 4


def test_kimi_moonvit_prepares_cuda_rope_inputs_once(monkeypatch):
    recorded = {}

    class CapturingRope:
        def get_freqs_cis(self, grid_thws, device):
            real = torch.arange(14, dtype=torch.float32, device=device).view(7, 2)
            return torch.complex(real, real + 1)

    class CapturingBlock(nn.Module):
        def forward(
            self,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            rope_freqs_cis,
            **kwargs,
        ):
            recorded["rope_freqs_cis"] = rope_freqs_cis
            return hidden_states

    monkeypatch.setattr(kimi_k25, "_is_cuda", True)
    encoder = MoonViT3dEncoder.__new__(MoonViT3dEncoder)
    nn.Module.__init__(encoder)
    encoder.rope_2d = CapturingRope()
    encoder.use_fused_rope = True
    encoder.blocks = nn.ModuleList([CapturingBlock()])
    encoder.final_layernorm = nn.Identity()

    hidden_states = torch.ones(7, 4)
    encoder(hidden_states, torch.tensor([[1, 1, 7]], dtype=torch.int32))

    cos_sin_cache, positions = recorded["rope_freqs_cis"]
    assert cos_sin_cache.shape == (7, 4)
    assert torch.equal(cos_sin_cache[:, :2] + 1, cos_sin_cache[:, 2:])
    assert torch.equal(positions, torch.arange(7))


def test_kimi_moonvit_cuda_rope_uses_fused_inplace_kernel(monkeypatch):
    recorded = {}

    def fake_apply_rope_inplace(q, k, cos_sin_cache, positions, **kwargs):
        recorded.update(
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            kwargs=kwargs,
        )
        q.add_(1)
        k.add_(2)

    monkeypatch.setattr(
        kimi_k25, "apply_rope_inplace", fake_apply_rope_inplace, raising=False
    )
    q = torch.zeros(3, 2, 4)
    k = torch.zeros_like(q)
    cache = torch.ones(3, 4, dtype=torch.float32)
    positions = torch.arange(3, dtype=torch.int64)

    q_out, k_out = kimi_k25.apply_rope(q, k, (cache, positions))

    assert q_out is q and k_out is k
    assert torch.equal(q, torch.ones_like(q))
    assert torch.equal(k, torch.full_like(k, 2))
    assert recorded["cos_sin_cache"] is cache
    assert recorded["positions"] is positions
    assert recorded["kwargs"] == {"is_neox": False, "rope_dim": 4}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kimi_moonvit_fused_rope_matches_portable_path():
    torch.manual_seed(0)
    q = torch.randn(256, 4, 72, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    angles = torch.randn(256, 36, device="cuda", dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    q_ref, k_ref = kimi_k25.apply_rope(q.clone(), k.clone(), freqs_cis)
    cache = torch.cat((freqs_cis.real, freqs_cis.imag), dim=-1)
    positions = torch.arange(256, device="cuda", dtype=torch.long)
    q_fused, k_fused = kimi_k25.apply_rope(q.clone(), k.clone(), (cache, positions))

    torch.testing.assert_close(q_fused, q_ref, rtol=0.01, atol=0.01)
    torch.testing.assert_close(k_fused, k_ref, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
