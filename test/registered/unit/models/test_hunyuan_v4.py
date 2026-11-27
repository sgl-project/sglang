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


def _input_scattered_allowed(monkeypatch, **blocked):
    """Whether HYV4 leaves input-scattered attention on, with every condition
    of the shared context met and HYV4's own conditions as ``blocked`` names."""
    from sglang.srt.layers.layer_boundary.adapters import attention

    context = attention.AttnTpContext()
    parallel = SimpleNamespace(
        enable_attn_tp_input_scattered=True,
        dcp_enabled=blocked.get("dcp", False),
    )
    spec = SimpleNamespace(
        speculative_algorithm="DFLY" if blocked.get("speculative") else None
    )
    monkeypatch.setattr(attention, "_is_cuda", True)
    monkeypatch.setattr(attention, "input_scattered_configured", lambda: True)
    monkeypatch.setattr(attention, "check_cuda_graph_backend", lambda *_: False)
    monkeypatch.setattr(attention, "get_parallel", lambda: parallel)
    # The shared context only rules out EAGLE3; HYV4 rules out every algorithm.
    monkeypatch.setattr(
        attention, "get_spec", lambda: SimpleNamespace(speculative_algorithm=None)
    )
    monkeypatch.setattr(hunyuan_v4, "get_attn_tp_context", lambda: context)
    monkeypatch.setattr(hunyuan_v4, "get_parallel", lambda: parallel)
    monkeypatch.setattr(hunyuan_v4, "get_spec", lambda: spec)
    monkeypatch.setattr(
        hunyuan_v4, "is_deepseek_dsa", lambda config: not blocked.get("no_dsa", False)
    )
    monkeypatch.setattr(
        hunyuan_v4,
        "check_cuda_graph_backend",
        lambda phase, backend: not blocked.get("prefill_graph", False),
    )
    monkeypatch.setattr(
        hunyuan_v4,
        "get_embedding_tp_kwargs",
        lambda: {"enable_tp": not blocked.get("replicated_embeddings", False)},
    )
    hunyuan_v4._init_attn_tp_context(
        SimpleNamespace(q_lora_rank=32),
        SimpleNamespace(world_size=2 if blocked.get("pipeline") else 1),
    )
    return context.allow_input_scattered


@pytest.mark.parametrize(
    "blocker",
    [
        None,
        "no_dsa",
        "speculative",
        "pipeline",
        "dcp",
        "prefill_graph",
        "replicated_embeddings",
    ],
)
def test_input_scattered_attention_is_refused_where_it_would_be_wrong(
    monkeypatch, blocker
):
    """Each condition leaves the streams or the stack input on rows the
    input-scattered path does not expect, which computes wrong hidden states
    rather than failing."""
    blocked = {} if blocker is None else {blocker: True}
    assert _input_scattered_allowed(monkeypatch, **blocked) is (blocker is None)


@pytest.mark.parametrize(
    "dense_fully_dp, moe_on_dp_shards, tp_size, attn_tp_size, refused",
    [
        (False, False, 8, 8, False),
        (True, False, 8, 1, True),
        (True, False, 1, 1, False),
        (False, True, 8, 2, True),
        (False, True, 8, 1, False),
    ],
)
def test_an_ffn_on_rank_rows_is_refused_at_startup(
    monkeypatch, dense_fully_dp, moe_on_dp_shards, tp_size, attn_tp_size, refused
):
    """The dense MLP is TP-sharded and the gated streams cannot be gathered
    back from a slice, so these layouts must not reach a forward."""
    monkeypatch.setattr(hunyuan_v4, "is_dense_ffn_fully_dp", lambda: dense_fully_dp)
    monkeypatch.setattr(
        hunyuan_v4, "is_moe_input_scattered_across_dp_ranks", lambda: moe_on_dp_shards
    )
    monkeypatch.setattr(
        hunyuan_v4,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=tp_size, attn_tp_size=attn_tp_size),
    )
    if refused:
        with pytest.raises(ValueError, match="HYV4 does not support"):
            hunyuan_v4._reject_ffn_on_rank_rows()
    else:
        hunyuan_v4._reject_ffn_on_rank_rows()


@pytest.mark.parametrize(
    "architecture", ["HYV4ForCausalLM", "HYV4VLForConditionalGeneration"]
)
def test_pipeline_parallelism_resolves_input_scattered_attention_off(
    architecture, caplog
):
    """With the flag left on, a later pipeline rank fails to bind its stages and
    the server never starts; the flag must be resolved off before any stage
    binds, since the model's own conditions run after that."""
    from sglang.srt.arg_groups.parallel_hook import _boundary_parallelism_overrides
    from sglang.srt.layers.layer_boundary import (
        BatchVariant,
        append_stages,
        layer_stack,
    )
    from sglang.test.boundary_fixtures import Norm, parallel_of, planning

    def resolved(pp_size):
        parallel = parallel_of(
            attn_dp=1, attn_tp=2, enable_attn_tp_input_scattered=True, pp_size=pp_size
        )
        overrides = _boundary_parallelism_overrides(
            parallel, "hunyuan_v4", architecture
        )
        return SimpleNamespace(**(vars(parallel) | overrides))

    assert resolved(pp_size=1).enable_attn_tp_input_scattered
    with caplog.at_level("WARNING"):
        parallel = resolved(pp_size=2)
    assert "pipeline parallelism" in caplog.text
    assert not parallel.enable_attn_tp_input_scattered

    config = SimpleNamespace(
        mlp_layer_types=["dense", "moe", "moe", "moe"], attn_post_proj_norm=True
    )
    stages = []
    # The second of two pipeline ranks: layers 0-1 are the previous rank's.
    with (
        planning(parallel),
        layer_stack(
            previous_layers=[
                lambda idx=idx: hunyuan_v4._stage_facts(config, idx) for idx in (0, 1)
            ],
        ),
    ):
        for layer_id in (2, 3):
            attn, ffn = hunyuan_v4._stage_facts(config, layer_id, post_proj_norm=Norm())
            stages.extend(append_stages((attn, Norm()), (ffn, Norm())))
    # The stack binds its stages on exit.
    assert len(stages) == 4
    for stage in stages:
        assert stage.plan.paths.get(BatchVariant.INPUT_SCATTERED) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
