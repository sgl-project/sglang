import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models import hunyuan_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def test_attention_gate_uses_attention_tp(monkeypatch):
    attn_tp_size = 2
    parallel = SimpleNamespace(
        attn_tp_rank=min(1, attn_tp_size - 1), attn_tp_size=attn_tp_size
    )
    captured = {}

    def fake_attention_init(module, **kwargs):
        nn.Module.__init__(module)
        module.hidden_size = kwargs["hidden_size"]
        module.num_local_heads = kwargs["num_heads"] // parallel.attn_tp_size

    class FakeColumnParallelLinear(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            captured.update(kwargs)
            self.output_size_per_partition = output_size // parallel.attn_tp_size

    monkeypatch.setattr(
        hunyuan_v4.DeepseekV2AttentionMLA, "__init__", fake_attention_init
    )
    monkeypatch.setattr(hunyuan_v4, "ColumnParallelLinear", FakeColumnParallelLinear)
    monkeypatch.setattr(hunyuan_v4, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        hunyuan_v4.HYV4Attention,
        "_hpc_gated_mla_supported",
        staticmethod(lambda *args: False),
    )

    config = SimpleNamespace(
        rope_parameters={"rope_theta": 10_000, "rope_type": "default"},
        hidden_size=6144,
        num_attention_heads=64,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=256,
        q_lora_rank=32,
        kv_lora_rank=16,
        max_position_embeddings=1024,
        gating_type="elementwise",
    )

    attention = hunyuan_v4.HYV4Attention(config, layer_id=0)

    assert captured["parallel_group"] == "attn_tp"
    assert attention.local_gate_width == (64 // attn_tp_size) * 256
    assert attention.linear_gate.output_size_per_partition == attention.local_gate_width


class TupleLinear(nn.Module):
    def __init__(self, input_size, output_size, dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(output_size, input_size, dtype=dtype), requires_grad=False
        )

    def forward(self, inputs):
        return nn.functional.linear(inputs, self.weight), None


def test_attention_gate_non_bf16_model_fallback_parity():
    torch.manual_seed(0)
    attention = hunyuan_v4.HYV4Attention.__new__(hunyuan_v4.HYV4Attention)
    nn.Module.__init__(attention)
    attention.linear_gate = TupleLinear(8, 256, torch.float32)
    attention.local_gate_width = 256
    attention._gate_backend = "eager"
    attention._gate_fallback_backend = "eager"
    hidden_states = torch.randn(3, 8)
    attn_out = torch.randn(3, 256)

    gate = attention.prepare_attention_output_gate(hidden_states)
    actual = attention.apply_attention_output_gate(attn_out, gate)
    expected = attn_out * torch.sigmoid(
        nn.functional.linear(hidden_states, attention.linear_gate.weight)
    )

    torch.testing.assert_close(actual, expected)


def test_hpc_attention_gate_is_bf16_only(monkeypatch):
    fake_hpc = SimpleNamespace(
        gemm=SimpleNamespace(gated_mla_gemm=object()), __version__="test"
    )
    monkeypatch.setitem(sys.modules, "hpc", fake_hpc)
    monkeypatch.setattr(hunyuan_v4, "get_device_capability", lambda: (10, 0))
    supported = hunyuan_v4.HYV4Attention._hpc_gated_mla_supported
    supported.cache_clear()
    try:
        assert supported("elementwise", torch.bfloat16, (256, 6144), 256, 6144)
        assert not supported("elementwise", torch.float32, (256, 6144), 256, 6144)
    finally:
        supported.cache_clear()


def test_hunyuan_v4_flash_topk_matches_torch_topk():
    """Regression test for #53683: hy_v4 `_topk` must not hard-depend on
    flashinfer's JIT (which needs nvcc and fails on non-CUDA accelerators,
    blocking decode CUDA-graph capture). Without flashinfer it must fall
    back to torch.topk and return the same result."""
    from sglang.srt.models import hunyuan_v4_flash

    torch.manual_seed(0)
    logits = torch.randn(4, 512)
    # Padded non-candidate columns score -inf, as in _select.
    logits[:, 300:] = float("-inf")
    values, indices = hunyuan_v4_flash._topk(logits, 8)
    ref_values, _ = torch.topk(logits, 8, dim=-1)
    assert indices.dtype == torch.int64
    torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
    torch.testing.assert_close(logits.gather(-1, indices), ref_values, rtol=0, atol=0)


def test_hunyuan_v4_flash_topk_falls_back_when_flashinfer_jit_fails(monkeypatch):
    """Regression test for #53683: even when the flashinfer top-k path is
    selected but its JIT raises (no nvcc), `_topk` must fall back to
    torch.topk instead of failing the decode graph capture."""
    from sglang.srt.models import hunyuan_v4_flash

    def _raise_jit_error(*args, **kwargs):
        raise RuntimeError(
            "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist"
        )

    monkeypatch.setattr(hunyuan_v4_flash, "radix_topk", _raise_jit_error)
    torch.manual_seed(0)
    logits = torch.randn(4, 512)
    values, indices = hunyuan_v4_flash._topk(logits, 8)
    ref_values, _ = torch.topk(logits, 8, dim=-1)
    torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
    torch.testing.assert_close(logits.gather(-1, indices), ref_values, rtol=0, atol=0)


def test_hunyuan_v4_flash_topk_returns_all_columns_when_k_covers_width():
    from sglang.srt.models import hunyuan_v4_flash

    logits = torch.randn(2, 8)
    values, indices = hunyuan_v4_flash._topk(logits, 8)
    torch.testing.assert_close(values, logits)
    torch.testing.assert_close(indices, torch.arange(8).expand(2, -1).contiguous())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
