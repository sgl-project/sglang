import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models import hunyuan_v4
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def test_indexer_block_fp8_scales_are_not_permuted():
    config = SimpleNamespace(
        index_n_heads=32,
        index_head_dim=128,
        qk_rope_head_dim=64,
    )
    tensors = {
        "model.layers.0.self_attn.indexer.wk.weight_scale": torch.empty(1, 48),
        "model.layers.0.self_attn.indexer.wq_b.weight_scale": torch.empty(32, 16),
    }

    for name, tensor in tensors.items():
        result = hunyuan_v4.permute_hyv4_indexer_weight(name, tensor, config)
        assert result is tensor


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
            self.output_size_per_partition = output_size // kwargs["tp_size"]

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

    assert captured["tp_rank"] == parallel.attn_tp_rank
    assert captured["tp_size"] == parallel.attn_tp_size
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


def test_prepared_attention_gate_requires_model_application_hook():
    with pytest.raises(RuntimeError, match="unsigmoided"):
        forward_mla._apply_attention_output_gate(
            SimpleNamespace(), torch.ones(1), torch.ones(1)
        )


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


class _PPDecoderLayer(nn.Module):
    def __init__(self, next_hidden, next_topk):
        super().__init__()
        self.next_hidden = next_hidden
        self.next_topk = next_topk

    def forward(self, *args):
        return self.next_hidden, self.next_topk


def test_non_last_pp_stage_forwards_flattened_hidden_and_shared_topk():
    model = hunyuan_v4.HYV4Model.__new__(hunyuan_v4.HYV4Model)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    model.start_layer = 1
    model.end_layer = 2
    model.send_topk_indices = True
    model.topk_indices_width = 8

    next_hidden = torch.randn(2, 4, 16)
    next_topk = torch.ones(2, 8, dtype=torch.int32)
    model.layers = nn.ModuleList(
        [nn.Identity(), _PPDecoderLayer(next_hidden, next_topk)]
    )

    initial_hidden = torch.randn(2, 4, 16)
    initial_topk = torch.zeros(2, 8, dtype=torch.int32)
    proxy = hunyuan_v4.PPProxyTensors(
        {"hidden_states": initial_hidden, "topk_indices": initial_topk}
    )

    forward_batch = SimpleNamespace(reuse_dsa_topk_indices=False, spec_info=None)
    result = model(None, None, forward_batch, pp_proxy_tensors=proxy)

    assert result["hidden_states"].shape == (2, 64)
    assert result["topk_indices"] is next_topk


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
