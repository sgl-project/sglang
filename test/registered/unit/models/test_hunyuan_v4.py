import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models import hunyuan_v4, hunyuan_v4_nextn
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.srt.runtime_context import get_context, get_flags, reset_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


@pytest.mark.parametrize("attn_tp_size", [1, 2, 4, 8, 16, 32, 64])
def test_attention_gate_uses_attention_tp(monkeypatch, attn_tp_size):
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


def test_attention_gate_rejects_mismatched_local_width(monkeypatch):
    parallel = SimpleNamespace(attn_tp_rank=0, attn_tp_size=2)

    def fake_attention_init(module, **kwargs):
        nn.Module.__init__(module)
        module.hidden_size = kwargs["hidden_size"]
        module.num_local_heads = kwargs["num_heads"] // parallel.attn_tp_size

    class MismatchedColumnParallelLinear(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.output_size_per_partition = output_size // kwargs["tp_size"] - 1

    monkeypatch.setattr(
        hunyuan_v4.DeepseekV2AttentionMLA, "__init__", fake_attention_init
    )
    monkeypatch.setattr(
        hunyuan_v4, "ColumnParallelLinear", MismatchedColumnParallelLinear
    )
    monkeypatch.setattr(hunyuan_v4, "get_parallel", lambda: parallel)
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

    with pytest.raises(ValueError, match="gate shard width"):
        hunyuan_v4.HYV4Attention(config, layer_id=0)


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
        for local_gate_width in [256, 512, 1024, 2048, 4096, 8192, 16384]:
            assert supported(
                "elementwise",
                torch.bfloat16,
                (local_gate_width, 6144),
                local_gate_width,
                6144,
            )
        assert not supported("elementwise", torch.float32, (256, 6144), 256, 6144)
    finally:
        supported.cache_clear()


def test_hpc_ihc_dispatch_guards(monkeypatch):
    from sglang.kernels.ops.layernorm import hy4_ihc
    from sglang.srt import utils

    op = object()
    monkeypatch.setitem(
        sys.modules,
        "hpc",
        SimpleNamespace(fuse_ihc_pre=op, __version__="test"),
    )
    monkeypatch.setattr(utils, "get_device_capability", lambda: (10, 3))
    hy4_ihc._hpc_ihc_op.cache_clear()
    try:
        assert hy4_ihc._hpc_ihc_op("fuse_ihc_pre", 4, 6144) is op
        assert hy4_ihc._hpc_ihc_op("fuse_ihc_head", 4, 6144) is None

        hy4_ihc._hpc_ihc_op.cache_clear()
        monkeypatch.setattr(utils, "get_device_capability", lambda: (8, 0))
        assert hy4_ihc._hpc_ihc_op("fuse_ihc_pre", 4, 6144) is None

        hy4_ihc._hpc_ihc_op.cache_clear()
        monkeypatch.setattr(utils, "get_device_capability", lambda: (10, 3))
        assert hy4_ihc._hpc_ihc_op("fuse_ihc_pre", 2, 6144) is None
        assert hy4_ihc._hpc_ihc_op("fuse_ihc_pre", 4, 8192) is None
    finally:
        hy4_ihc._hpc_ihc_op.cache_clear()


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
