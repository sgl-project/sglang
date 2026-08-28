import sys
from types import SimpleNamespace

import pytest
from torch import nn

from sglang.srt.models import hunyuan_v4, hunyuan_v4_nextn
from sglang.srt.runtime_context import get_context, get_flags, reset_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def test_attention_gate_uses_attention_tp(monkeypatch):
    parallel = SimpleNamespace(attn_tp_rank=1, attn_tp_size=2)
    captured = {}

    def fake_attention_init(module, **kwargs):
        nn.Module.__init__(module)
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

    config = SimpleNamespace(
        rope_parameters={"rope_theta": 10_000, "rope_type": "default"},
        hidden_size=64,
        num_attention_heads=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=16,
        q_lora_rank=32,
        kv_lora_rank=16,
        max_position_embeddings=1024,
    )

    attention = hunyuan_v4.HYV4Attention(config, layer_id=0)

    assert captured["tp_rank"] == parallel.attn_tp_rank
    assert captured["tp_size"] == parallel.attn_tp_size
    assert attention.linear_gate.output_size_per_partition == (
        attention.num_local_heads * config.v_head_dim
    )


def test_vocab_embeddings_follow_attention_tp(monkeypatch):
    calls = []

    class CaptureEmbedding(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, **kwargs):
            super().__init__()
            calls.append(kwargs)

    class DummyLayer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    monkeypatch.setattr(hunyuan_v4, "VocabParallelEmbedding", CaptureEmbedding)
    monkeypatch.setattr(hunyuan_v4_nextn, "VocabParallelEmbedding", CaptureEmbedding)
    monkeypatch.setattr(hunyuan_v4, "HYV4HCHeadLayer", DummyLayer)
    monkeypatch.setattr(hunyuan_v4, "RMSNorm", DummyLayer)
    monkeypatch.setattr(hunyuan_v4_nextn, "RMSNorm", DummyLayer)
    monkeypatch.setattr(hunyuan_v4_nextn, "HYV4MTPDecoderLayer", DummyLayer)
    monkeypatch.setattr(
        hunyuan_v4, "get_pp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(hunyuan_v4, "is_cuda", lambda: False)
    monkeypatch.setattr(hunyuan_v4_nextn, "is_cuda", lambda: False)
    config = SimpleNamespace(
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=0,
        rms_norm_eps=1e-5,
    )

    for enabled in (False, True):
        calls.clear()
        with get_flags().dp.override(enabled=enabled):
            hunyuan_v4.HYV4Model(config)
            hunyuan_v4_nextn.HYV4ModelNextN(config)

        assert [(call["enable_tp"], call["use_attn_tp_group"]) for call in calls] == [
            (True, enabled),
            (True, enabled),
        ]


def test_lm_heads_follow_dp_lm_head_config(monkeypatch):
    calls = []

    class CaptureLMHead(nn.Module):
        def __init__(self, vocab_size, hidden_size, **kwargs):
            super().__init__()
            calls.append(kwargs)

    class TargetModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.layers = nn.ModuleList()

    class NextNModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.decoder = SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=0)
            )

    class DummyLogitsProcessor:
        def __init__(self, config):
            pass

    monkeypatch.setattr(hunyuan_v4, "HYV4Model", TargetModel)
    monkeypatch.setattr(hunyuan_v4_nextn, "HYV4ModelNextN", NextNModel)
    monkeypatch.setattr(hunyuan_v4, "ParallelLMHead", CaptureLMHead)
    monkeypatch.setattr(hunyuan_v4_nextn, "ParallelLMHead", CaptureLMHead)
    monkeypatch.setattr(hunyuan_v4, "LogitsProcessor", DummyLogitsProcessor)
    monkeypatch.setattr(hunyuan_v4_nextn, "LogitsProcessor", DummyLogitsProcessor)
    monkeypatch.setattr(hunyuan_v4, "get_pp_group", lambda: SimpleNamespace())
    monkeypatch.setattr(hunyuan_v4_nextn, "get_pp_group", lambda: SimpleNamespace())
    config = SimpleNamespace(
        vocab_size=128,
        hidden_size=16,
        enable_lm_head_fp32=False,
    )

    reset_context()
    try:
        for enabled in (False, True):
            calls.clear()
            with get_context().override_server_args(
                enable_dp_attention=enabled,
                enable_dp_lm_head=enabled,
            ):
                hunyuan_v4.HYV4ForCausalLM(config)
                hunyuan_v4_nextn.HYV4ForCausalLMNextN(config)

            assert [call["use_attn_tp_group"] for call in calls] == [
                enabled,
                enabled,
            ]
    finally:
        reset_context()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
