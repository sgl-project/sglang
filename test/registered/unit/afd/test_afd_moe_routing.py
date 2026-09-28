from __future__ import annotations

from sglang.test.afd.graph_fixtures import make_shape as planned_shape
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import copy
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd import config as afd_config
from sglang.srt.afd import contracts as afd_contracts
from sglang.srt.afd import pipeline as afd_pipeline
from sglang.srt.afd.model_adapters import base as afd_adapter
from sglang.srt.afd.model_adapters import glm5 as afd_glm5_adapter

nn = torch.nn


from sglang.test.afd.moe_fixtures import _moe, bare_module


def _configure_a2a(namespace, backend="none"):
    namespace["get_moe_a2a_backend"] = lambda: SimpleNamespace(
        is_none=lambda: backend == "none"
    )


def _deepep_moe(namespace, *, policy="normal", prescaled=False, shared=True):
    moe, expert_weights = _moe(
        namespace, prescaled=prescaled, tp_size=2, shared_tp1=True
    )
    events = []

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.should_fuse_routed_scaling_factor_in_topk = prescaled
            self.seen = []

        def forward(self, hidden_states, topk_output):
            self.seen.append(topk_output)
            events.append("dispatch")
            per_expert = torch.einsum("ri,eki->rek", hidden_states, expert_weights)
            chosen = per_expert.gather(
                1, topk_output.topk_ids.long().unsqueeze(-1).expand(-1, -1, 4)
            )
            if policy == "normal":
                events.append("ep_gather_weights")
                chosen = chosen * topk_output.topk_weights.unsqueeze(-1)
            events.append("combine")
            if policy == "low_latency":
                events.append("combine_weights")
                chosen = chosen * topk_output.topk_weights.unsqueeze(-1)
            return chosen.sum(1)

    moe.experts = Experts()
    moe.experts.register_forward_pre_hook(lambda *args: events.append("pre_experts"))
    moe.experts.register_forward_hook(lambda *args: events.append("post_experts"))
    moe._enable_a2a_moe = True
    moe.alt_stream = None
    moe.forward = namespace["DeepseekV2MoE"].forward.__get__(moe)
    moe.forward_deepep = namespace["DeepseekV2MoE"].forward_deepep.__get__(moe)

    def shared_experts(x):
        events.append("shared")
        return x * 0.125 if shared else None

    moe._forward_shared_experts = shared_experts
    return moe, expert_weights, events


def _reference(x, gate, expert_weights):
    scores = torch.sigmoid(torch.nn.functional.linear(x, gate.weight))
    choice = scores + gate.e_score_correction_bias
    outputs = []
    ids_list = []
    weights_list = []
    for row in range(x.shape[0]):
        group_scores = [
            sum(
                sorted(choice[row, group * 4 : (group + 1) * 4].tolist(), reverse=True)[
                    :2
                ]
            )
            for group in range(2)
        ]
        group = max(range(2), key=lambda g: group_scores[g])
        ids = sorted(
            range(group * 4, (group + 1) * 4),
            key=lambda e: float(choice[row, e]),
            reverse=True,
        )[:2]
        weight = scores[row, ids] / scores[row, ids].sum()
        outputs.append(
            sum((expert_weights[e] @ x[row]) * w for e, w in zip(ids, weight)) * 2.5
            + x[row] * 0.125
        )
        ids_list.append(ids)
        weights_list.append(weight)
    if not outputs:
        return torch.empty_like(x), [], []
    return torch.stack(outputs), ids_list, weights_list


@pytest.mark.parametrize("rows", [0, 1, 5, 32])
@pytest.mark.parametrize("prescaled", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_ffn_router_output_matches_independent_reference(
    routing, rows, prescaled, inplace
):
    moe, expert_weights = _moe(routing, prescaled=prescaled, inplace=inplace)
    x = torch.randn(rows, 4, generator=torch.Generator().manual_seed(31))
    layer = SimpleNamespace(mlp=moe)
    actual = routing["GlmMoeDsaAFDDecoderLayer"].compute_ffn_output(layer, x)
    expected, _, _ = _reference(x, moe.gate, expert_weights)
    torch.testing.assert_close(actual, expected)
    assert moe.gate.calls == moe.topk.calls == int(rows > 0)


@pytest.mark.parametrize("is_nextn", [False, True])
@pytest.mark.parametrize("is_hash", [False, True])
def test_deepep_locally_computed_routes_keep_forward_batch_and_location(
    routing, monkeypatch, is_nextn, is_hash
):
    moe, _, _ = _deepep_moe(routing)
    moe.is_nextn = is_nextn
    moe.is_hash = is_hash
    batch = SimpleNamespace(num_token_non_padded=2, moe_num_token_non_padded=lambda: 2)
    input_ids = torch.tensor([11, 17, 23])
    observed = []
    locations = []
    dispatch_info = object()
    original_topk = moe.topk.forward

    def topk(hidden, logits, **kwargs):
        observed.append(kwargs)
        return original_topk(hidden, logits, **kwargs)

    def location(**kwargs):
        locations.append(kwargs)
        return dispatch_info

    monkeypatch.setattr(moe.topk, "forward", topk)
    routing["ExpertLocationDispatchInfo"].init_new = location
    moe(torch.ones(3, 4), forward_batch=batch, input_ids_global=input_ids)
    assert observed[0]["num_token_non_padded"] == 2
    assert observed[0]["expert_location_dispatch_info"] is (
        None if is_nextn else dispatch_info
    )
    assert ("input_ids" in observed[0]) == is_hash
    if is_hash:
        assert observed[0]["input_ids"] is input_ids
    assert locations == ([] if is_nextn else [{"layer_id": 0}])
    assert moe.gate.calls == moe.topk.calls == 1


@pytest.mark.parametrize("role", ["attention", "ffn", "off"])
def test_weight_filter_preserves_role_owned_gates(routing, role):

    class Base:
        def load_weights(self, weights, is_nextn=False):
            return list(weights)

    routing["afd_execution_mode"] = lambda: role
    _configure_a2a(routing, backend="none")
    routing.monkeypatch.setattr(
        routing["DeepseekV2ForCausalLM"], "load_weights", Base.load_weights
    )
    names = [
        "model.layers.3.mlp.gate.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.3.mlp.shared_experts.down_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.3.mlp.gate.weight.extra",
        "model.layers.3.self_attn.q_a_proj.weight",
    ]
    model = bare_module(routing["GlmMoeDsaForCausalLM"])
    loaded = model.load_weights(((name, object()) for name in names))
    expected = names
    if role == "attention":
        expected = [] + names[-1:]
    elif role == "ffn":
        expected = names[:-1]
    assert [name for (name, _) in loaded] == expected
    if role == "off":
        assert model.load_weights([(name, None) for name in names], is_nextn=True) == [
            (name, None) for name in names
        ]
    else:
        with pytest.raises(RuntimeError, match="AFD_GLM_DSA_MTP_UNSUPPORTED"):
            model.load_weights([], is_nextn=True)


@pytest.mark.parametrize("runner", ["triton", "deep_gemm"])
@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("sparse", [False, True])
def test_role_projection_preserves_only_owned_modules(routing, role, sparse, runner):
    routing["get_moe_runner_backend"] = lambda: SimpleNamespace(value=runner)
    routing["afd_execution_mode"] = lambda: role
    layer = bare_module(routing["DeepseekV2DecoderLayer"])
    layer.config = SimpleNamespace(num_experts_per_tok=2)
    layer.is_layer_sparse = sparse
    layer.mlp, _ = _moe(routing)
    moe = layer.mlp
    layer.self_attn = attention = nn.Identity()
    layer.layer_communicator = SimpleNamespace(qkv_latent_func=object())
    routing["GlmMoeDsaAFDDecoderLayer"].install(layer)
    if role == "attention":
        assert isinstance(layer.mlp, routing["AFDProxyMLP"])
        assert not list(layer.mlp.parameters())
        assert layer.self_attn is attention
    else:
        assert layer.mlp is moe
        assert isinstance(layer.self_attn, routing["AFDProxyAttention"])
        assert layer.layer_communicator.qkv_latent_func is None


@pytest.mark.parametrize("a2a_backend", ["none"])
@pytest.mark.parametrize("resolved_only", [False, True])
@pytest.mark.parametrize(
    "bad",
    [
        "enable_eplb",
        "init_expert_location",
        "ep_num_redundant_experts",
        "enable_waterfill",
        "is_hash",
        "fused_shared",
        "backend",
    ],
)
def test_role_projection_refuses_unsupported_routing_contract(
    routing, bad, a2a_backend, resolved_only
):
    moe, _ = _moe(routing)
    layer = bare_module(routing["DeepseekV2DecoderLayer"])
    layer.config = SimpleNamespace(num_experts_per_tok=2)
    layer.is_layer_sparse = True
    layer.mlp = moe
    layer.layer_communicator = SimpleNamespace()
    args = SimpleNamespace(
        enable_eplb=False,
        init_expert_location="trivial",
        ep_num_redundant_experts=0,
        enable_waterfill=False,
    )
    if bad == "init_expert_location":
        args.init_expert_location = "random"
    elif bad in ("enable_eplb", "ep_num_redundant_experts", "enable_waterfill"):
        setattr(args, bad, 1)
    elif bad == "is_hash":
        moe.is_hash = True
    elif bad == "fused_shared":
        moe.num_fused_shared_experts = 1
    else:
        routing["get_moe_runner_backend"] = lambda: SimpleNamespace(
            value="flashinfer_trtllm"
        )
    routing["get_server_args"] = lambda: args
    _configure_a2a(routing, backend=a2a_backend)
    if resolved_only:
        # v0.5.20 keeps raw CLI input separate from the published resolution.
        raw = copy.copy(args)
        raw.enable_eplb = False
        raw.init_expert_location = "trivial"
        raw.ep_num_redundant_experts = 0
        raw.enable_waterfill = False
        routing["get_server_args"] = lambda: raw
        routing["get_exec"] = lambda: SimpleNamespace(moe=args)
    with pytest.raises(RuntimeError, match="AFD_GLM_ROUTING_"):
        routing["GlmMoeDsaAFDDecoderLayer"].install(layer)


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("backend", ["deepep", "mooncake"])
def test_role_projection_and_compute_refuse_unsupported_collective(
    routing, role, backend
):
    _configure_a2a(routing, backend=backend)
    routing["afd_execution_mode"] = lambda: role
    layer = bare_module(routing["DeepseekV2DecoderLayer"])
    layer.mlp, _ = _moe(routing)
    with pytest.raises(
        RuntimeError, match="AFD_GLM_DSA_INTERNAL_COLLECTIVE_UNSUPPORTED"
    ):
        routing["GlmMoeDsaAFDDecoderLayer"].install(layer)
    if role == "ffn":
        with pytest.raises(
            RuntimeError, match="AFD_GLM_DSA_INTERNAL_COLLECTIVE_UNSUPPORTED"
        ):
            routing["GlmMoeDsaAFDDecoderLayer"].compute_ffn_output(
                layer, torch.ones(3, 4)
            )
    assert not layer.mlp.experts.seen


@pytest.mark.parametrize("runner", ["triton", "deep_gemm"])
@pytest.mark.parametrize("rows", [0, 5])
def test_ffn_router_loads_gate_and_rejects_old_route_payloads(routing, runner, rows):
    _configure_a2a(routing, backend="none")
    routing["get_moe_runner_backend"] = lambda: SimpleNamespace(value=runner)
    layer = bare_module(routing["DeepseekV2DecoderLayer"])
    layer.config = SimpleNamespace(num_experts_per_tok=2)
    layer.is_layer_sparse = True
    layer.layer_communicator = SimpleNamespace()
    layer.self_attn = nn.Identity()
    layer.mlp, expert_weights = _moe(routing)
    routing["GlmMoeDsaAFDDecoderLayer"].install(layer)
    gate = layer.mlp.gate
    prefix = "model.layers.0.mlp.gate."
    params = {prefix + name: param for (name, param) in gate.named_parameters()}
    loaded = []

    class Base:
        def load_weights(self, weights, is_nextn=False):
            for name, value in weights:
                loaded.append(name)
                params[name].copy_(value)

    routing.monkeypatch.setattr(
        routing["DeepseekV2ForCausalLM"], "load_weights", Base.load_weights
    )
    weights = [
        (prefix + "weight", -gate.weight.clone()),
        (prefix + "e_score_correction_bias", -gate.e_score_correction_bias.clone()),
    ]
    bare_module(routing["GlmMoeDsaForCausalLM"]).load_weights(
        iter(weights + [("model.layers.0.self_attn.q_a_proj.weight", None)])
    )
    assert loaded == [name for (name, _) in weights]
    for name, value in weights:
        torch.testing.assert_close(params[name], value)
    x = torch.randn(rows, 4, generator=torch.Generator().manual_seed(31))
    actual = layer.compute_ffn_output(x)
    expected, _, _ = _reference(x, gate, expert_weights)
    torch.testing.assert_close(actual, expected)
    assert gate.calls == layer.mlp.topk.calls == int(rows > 0)
    ids = torch.zeros(rows, 2, dtype=torch.int32)
    weights = torch.ones(rows, 2, dtype=torch.float32)
    for route_ids, route_weights in ((ids, weights), (ids, None), (None, weights)):
        with pytest.raises(TypeError, match="unexpected keyword"):
            layer.compute_ffn_output(x, topk_ids=route_ids, topk_weights=route_weights)
    assert gate.calls == layer.mlp.topk.calls == int(rows > 0)


@pytest.mark.parametrize("mode", ["off", "ffn"])
def test_quantization_restriction_applies_only_to_afd(routing, mode):
    class Base:
        def __init__(self, **kwargs):
            pass

    routing["afd_execution_mode"] = lambda: mode
    routing.monkeypatch.setattr(
        routing["DeepseekV2ForCausalLM"], "__init__", Base.__init__
    )
    model_type = routing["GlmMoeDsaForCausalLM"]
    quant = SimpleNamespace(get_name=lambda: "modelopt_fp4")
    if mode == "off":
        model_type(config=SimpleNamespace(), quant_config=quant)
    else:
        with pytest.raises(
            RuntimeError, match="AFD_GLM_ROUTING_QUANTIZATION_UNSUPPORTED"
        ):
            model_type(config=SimpleNamespace(), quant_config=quant)


@pytest.mark.parametrize("rows", [(3, 2), (3, 0), (1, 5)])
@pytest.mark.parametrize("runner", ["triton", "deep_gemm"])
def test_cpu_pipeline_runs_dense_and_moe_with_ffn_router(
    routing, monkeypatch, rows, runner
):
    _configure_a2a(routing, backend="none")
    routing["get_moe_runner_backend"] = lambda: SimpleNamespace(value=runner)
    layers_count = 3
    layers = []
    a_layers = []
    references = []

    class Dense(nn.Module):
        def forward(self, x, forward_batch=None):
            return x * 0.75

    class AttentionLayer:
        def __init__(self, index):
            self.index = index
            self.self_attn = SimpleNamespace(skip_topk=False, next_skip_topk=False)
            self.layer_communicator = SimpleNamespace(
                postprocess_layer=lambda hidden, residual, batch: (hidden, residual)
            )

        def forward_attention_for_afd(self, *, hidden_states, residual, **kwargs):
            return (hidden_states + (self.index + 1) * 0.01, residual, None)

    for index in range(layers_count):
        f_layer = bare_module(routing["DeepseekV2DecoderLayer"])
        f_layer.config = SimpleNamespace(num_experts_per_tok=2)
        f_layer.is_layer_sparse = index > 0
        f_layer.self_attn = nn.Identity()
        f_layer.layer_communicator = SimpleNamespace(qkv_latent_func=object())
        if index > 0:
            f_layer.mlp, _ = _moe(routing)
            reference, _ = _moe(routing)
        else:
            f_layer.mlp = reference = Dense()
        routing["GlmMoeDsaAFDDecoderLayer"].install(f_layer)
        layers.append(f_layer)
        a_layers.append(AttentionLayer(index))
        references.append(reference)

    def adapter(role, layer_list):
        value = object.__new__(afd_glm5_adapter.Glm5AFDAdapter)
        value.role = role
        value.inner = SimpleNamespace(layers=layer_list)
        value.num_layers = layers_count
        value.hidden_size = 4
        value.attention_backend = SimpleNamespace(
            forward_metadata=object(), init_forward_metadata=lambda batch: None
        )
        value._step_id = 1
        value._stage_states = {
            i: SimpleNamespace(zero_allocator=None) for i in range(len(rows))
        }
        return value

    a_adapter = adapter(afd_contracts.AFDRole.ATTENTION, a_layers)
    f_adapter = adapter(afd_contracts.AFDRole.FFN, layers)
    cfg = afd_config.AFDConfig(stages=len(rows))
    from sglang.srt import runtime_context

    monkeypatch.setattr(
        afd_adapter, "get_parallel", lambda: SimpleNamespace(tp_group=object())
    )
    monkeypatch.setattr(runtime_context, "get_forward", lambda: SimpleNamespace())
    f_pipeline = afd_pipeline.AFDFFNPipeline(
        adapter=f_adapter,
        connector=SimpleNamespace(),
        config=cfg,
        device="cpu",
        dtype=torch.float32,
        shape_factory=planned_shape,
    )
    x = torch.randn(sum(rows), 4, generator=torch.Generator().manual_seed(127))
    stages = []
    offset = 0
    for i, count in enumerate(rows):
        stages.append(
            afd_adapter.AFDStage(
                index=i,
                request_start=offset,
                request_stop=offset + count,
                token_start=offset,
                token_stop=offset + count,
                hidden_states=x[offset : offset + count].clone(),
                residual=None,
                positions=None,
                forward_batch=None,
            )
        )
        offset += count
    packets = []

    class Transport:
        pending = None

        def dispatch(self, hidden, **routes):
            layer, index = divmod(len(packets), len(rows))
            packets.append((layer, index, routes))
            stage = afd_adapter.AFDStage(
                index=index,
                request_start=0,
                request_stop=hidden.shape[0],
                token_start=0,
                token_stop=hidden.shape[0],
                hidden_states=None,
                residual=None,
                positions=None,
                forward_batch=None,
                merge_sizes=(8,),
                group_widths=(8,),
            )
            self.pending = f_pipeline._adapter.local_compute(
                layer=layer,
                stage=stage,
                hidden_states=(hidden.clone(),),
                residual=None,
            )[0][0]

        def receive_return(self, buffer):
            buffer.copy_(self.pending)
            return None

    transport = Transport()
    transport.wait = lambda event: None
    a_pipeline = afd_pipeline.AFDAttentionPipeline(
        adapter=a_adapter,
        connector=SimpleNamespace(transport=transport),
        config=cfg,
        shape_factory=planned_shape,
    )
    a_pipeline._run_layers(
        stages=stages,
        recv_buffers=[torch.empty_like(stage.hidden_states) for stage in stages],
        participates=(True,) * len(rows),
    )
    actual = torch.cat([stage.hidden_states for stage in stages])
    expected = x
    for index, reference in enumerate(references):
        expected = reference(expected + (index + 1) * 0.01)
    torch.testing.assert_close(actual, expected)
    assert len(packets) == layers_count * len(rows)
    live_stages = sum((count > 0 for count in rows))
    for layer in layers[1:]:
        assert layer.mlp.gate.calls == layer.mlp.topk.calls == live_stages
    assert all(not routes for _, _, routes in packets)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
