"""CPU contracts for scale: production construction, routing and peer maps."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from types import SimpleNamespace

import pytest

from sglang.srt import runtime_context
from sglang.srt.afd import config, contracts, model_hooks, profiles
from sglang.test.afd.config_fixtures import _lane_server_args
from sglang.test.afd.moe_fixtures import (
    bare_module,
    torch,
)


def _install_constructor(routing, monkeypatch):
    args = routing["get_server_args"]()
    args.afd_config = config.AFDConfig()
    args.speculative_algorithm = None
    routing["get_server_args"] = lambda: args
    routing["afd_execution_mode"] = lambda: "attention"
    calls = []
    routing["add_prefix"] = lambda a, b: b + "." + a

    def forbidden(**kwargs):
        raise AssertionError("A role must never construct dense/shared/routed experts")

    routing.update(
        DeepseekV2AttentionMLA=lambda **kw: SimpleNamespace(prepare_qkv_latent=None),
        DeepseekV2MoE=forbidden,
        DeepseekV2MLP=forbidden,
        SpeculativeAlgorithm=SimpleNamespace(from_string=lambda x: x),
        LayerFacts=SimpleNamespace(init_new=lambda **kw: kw),
        LayerCommunicator=lambda **kw: SimpleNamespace(**kw),
        RMSNorm=lambda *a, **kw: torch.nn.Identity(),
        _is_gfx95_supported=False,
        enable_moe_dense_fully_dp=lambda: False,
    )
    # install() checks exact production base class identity.
    return calls


def _config():
    return SimpleNamespace(
        model_type="glm_moe_dsa",
        hidden_size=4,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        hidden_act="silu",
        rope_theta=1000000,
        rope_scaling=None,
        max_position_embeddings=202752,
        num_attention_heads=64,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        q_lora_rank=2048,
        kv_lora_rank=512,
        num_hidden_layers=78,
        rms_norm_eps=1e-5,
        n_routed_experts=8,
        first_k_dense_replace=3,
        moe_layer_freq=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
    )


@pytest.mark.parametrize("layer_id", [0, 3])
@pytest.mark.parametrize("tp", [3, 8, 20])
def test_attention_constructor_never_allocates_router_or_experts(
    routing, monkeypatch, layer_id, tp
):
    calls = _install_constructor(routing, monkeypatch)
    context = runtime_context.RuntimeContext(parallel=runtime_context.ParallelContext())
    context.set_server_args(SimpleNamespace(afd_execution_mode="attention"))
    context.parallel.override_permanently(
        tp_size=tp,
        tp_rank=0,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_dp_size=tp,
        attn_cp_size=1,
        enable_prefill_cp=False,
    )
    monkeypatch.setattr(runtime_context, "_CONTEXT", context)
    monkeypatch.setattr(runtime_context, "_PARALLEL", context.parallel)
    routing["afd_execution_mode"] = model_hooks.afd_execution_mode
    routing["get_parallel"] = runtime_context.get_parallel
    model_config = _config()
    model_config.quantization_config = {"weight_block_size": [128, 128]}
    layer = routing["DeepseekV2DecoderLayer"](
        model_config, layer_id, quant_config=SimpleNamespace(get_name=lambda: "fp8")
    )
    routing["GlmMoeDsaAFDDecoderLayer"].install(layer)
    assert isinstance(layer.mlp, routing["AFDProxyMLP"])
    assert list(layer.mlp.parameters()) == [] and calls == []
    assert not hasattr(layer.mlp, "quant_method")
    loaded = []

    def load_weights(self, weights, is_nextn=False):
        assert not is_nextn
        loaded.extend(name for name, _ in weights)

    monkeypatch.setattr(routing["DeepseekV2ForCausalLM"], "load_weights", load_weights)
    names = (
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.self_attn.q_a_proj.weight",
        "lm_head.weight",
    )
    bare_module(routing["GlmMoeDsaForCausalLM"]).load_weights(
        iter((name, torch.zeros(1)) for name in names)
    )
    assert loaded == list(names[-2:])


@pytest.mark.parametrize("kind", ["hash", "activation", "quantization", "role"])
def test_projection_refuses_unsupported_model_contract(routing, monkeypatch, kind):
    _install_constructor(routing, monkeypatch)
    cfg = _config()
    quant = None
    if kind == "hash":
        cfg.num_hash_layers = 4
    elif kind == "activation":
        cfg.hidden_act = "relu"
    elif kind == "quantization":
        quant = SimpleNamespace(get_name=lambda: "modelopt_fp4")
    else:
        routing["afd_execution_mode"] = lambda: "ffn"
    with pytest.raises(RuntimeError, match="AFD_GLM"):
        routing["make_glm_dsa_attention_mlp"](
            config=cfg, layer_id=3, quant_config=quant
        )


@pytest.mark.parametrize(
    "attention,ffn", [(12, 4), (16, 8), (24, 8), (32, 8), (32, 16)]
)
@pytest.mark.parametrize("backend", ["nsa", "fa4"])
def test_scale_maps_every_rank_exactly_once_and_checks_both_roles(
    attention, ffn, backend
):
    cfg = config.AFDConfig(
        lanes=ffn,
        attention_lanes=attention,
        attention_backend=backend,
    )
    cfg.validate()
    profile = profiles.GLM5_PAIRED_C1 if backend == "nsa" else profiles.QWEN3_PAIRED_C1
    profile.validate_shape(config=cfg)
    topology = contracts.AFDPairedTopology.paired(lanes=ffn, attention_lanes=attention)
    topology.validate()
    assert topology.coordination_world_size == attention + ffn
    assert topology.pair_world_size == 1 + attention // ffn
    groups = [topology.attention_lane_group(ffn_ordinal=i) for i in range(ffn)]
    assert sorted(rank for group in groups for rank in group) == list(range(attention))
    for rank in range(attention):
        (peer,) = topology.peers(role=contracts.AFDRole.ATTENTION, ordinal=rank)
        assert rank in groups[peer.ordinal]
    for role, size in (("attention", attention), ("ffn", ffn)):
        args = _lane_server_args(
            role=role,
            lanes=ffn,
            attention_lanes=attention,
            nnodes=size // 4,
            tp_size=size,
            dp_size=size if role == "attention" else 1,
            ep_size=1 if role == "attention" else ffn,
            enable_dp_attention=role == "attention",
        )
        args.afd_config = cfg
        args.attention_backend = backend
        config.validate_afd_server_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
