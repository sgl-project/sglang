"""FFN plugin ownership and graph warmup lifecycle contracts."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd.config import AFDConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_backend_warmup_barrier_is_never_recorded_in_graph(monkeypatch):
    from sglang.srt.afd.pipeline import _finish_backend_warmup

    calls = []
    backend = SimpleNamespace(
        needs_warmup_barrier=True, finish_warmup=lambda: calls.append(1)
    )
    connector = SimpleNamespace(
        transport=SimpleNamespace(ffn_backend=backend), graph_strategy=SimpleNamespace()
    )
    for capturing, in_graph, count in [
        (False, False, 0),
        (True, False, 1),
        (True, True, 1),
    ]:
        connector.graph_strategy.capturing = capturing
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: in_graph)
        _finish_backend_warmup(connector)
        assert len(calls) == count
    connector.graph_strategy.capturing = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    backend.needs_warmup_barrier = False
    _finish_backend_warmup(connector)
    assert calls == [1]


def test_backend_construction_failure_retains_partial_cleanup_ownership(monkeypatch):
    from sglang.srt.afd.expert_exchange import AFDDirectExchange
    from sglang.srt.afd.ffn_backend import build_ffn_backend

    transport = SimpleNamespace(ffn_backend=None)
    config = AFDConfig(lanes=4, attention_lanes=8, experimental_direct=True)
    comm = object()

    def fail_after_first_communicator(self, **kwargs):
        assert transport.ffn_backend is self
        self.communicators.append(("partial", comm))
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(AFDDirectExchange, "__init__", fail_after_first_communicator)
    with pytest.raises(RuntimeError, match="initialization failed"):
        build_ffn_backend(transport=transport, adapter=None, config=config, device=None)
    assert transport.ffn_backend.communicators == [("partial", comm)]
    transport.ffn_backend.close(seconds=1)

    empty_transport = SimpleNamespace()
    assert (
        build_ffn_backend(
            transport=empty_transport, adapter=None, config=AFDConfig(), device=None
        )
        is None
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    "case", ["qualified", "family", "dtype", "geometry", "placement", "runner", "a2a"]
)
def test_direct_admission_checks_actual_runtime_flags_and_expert_placement(
    monkeypatch, role, case
):
    from sglang.srt.afd.contracts import AFDRole
    from sglang.srt.afd.ffn_backend import validate_ffn_backend
    from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.runtime_context import get_flags

    flags = get_flags().moe
    monkeypatch.setattr(
        flags,
        "a2a_backend",
        MoeA2ABackend.DEEPEP if case == "a2a" else MoeA2ABackend.NONE,
    )
    monkeypatch.setattr(
        flags,
        "runner_backend",
        MoeRunnerBackend.AUTO if case == "runner" else MoeRunnerBackend.TRITON,
    )
    experts = SimpleNamespace(
        moe_tp_size=1, moe_ep_size=2, num_local_experts=8, reduce_results=False
    )
    mlp = SimpleNamespace(route=lambda value: value, experts=experts)
    if case == "placement":
        experts.moe_tp_size = 2
        mlp.route = None
    adapter = SimpleNamespace(
        role=AFDRole(role),
        hidden_size=8,
        num_layers=1,
        inner=SimpleNamespace(
            config=SimpleNamespace(
                num_experts=16, num_experts_per_tok=17 if case == "geometry" else 2
            ),
            layers=[SimpleNamespace(mlp=mlp)],
        ),
    )
    result = validate_ffn_backend(
        config=AFDConfig(lanes=2),
        adapter=adapter,
        model_family="other" if case == "family" else "qwen3_moe",
        dtype=torch.float16 if case == "dtype" else torch.bfloat16,
        capture_sizes=(8,),
    )
    expected = {
        "qualified": "qualified",
        "family": "model_family",
        "dtype": "hidden_dtype",
        "geometry": "model_geometry",
        "placement": "non_routed_layer"
        if role == "attention"
        else "native_expert_placement",
        "runner": "moe_runner",
        "a2a": "explicit_moe_a2a",
    }[case]
    assert result["reason"] == expected
    assert result["available"] is (case == "qualified")
    if case == "qualified":
        assert result["geometry"] == dict(
            hidden_size=8, num_layers=1, num_experts=16, top_k=2
        )


@pytest.mark.parametrize(
    "failure", [None, "mode", "geometry", "capacity", "capture", "unsupported"]
)
def test_peer_direct_plan_is_jointly_qualified_before_data_exchange(failure):
    from sglang.srt.afd.contracts import AFDError
    from sglang.srt.afd.ffn_backend import negotiate_direct_capabilities

    offers = [
        dict(
            mode="auto",
            available=True,
            reason="qualified",
            geometry={"experts": 16},
            eager_capacity=8,
        )
        for _ in range(3)
    ]
    descriptors = [
        dict(
            role=role,
            runtime_contract=dict(direct_execution=offer, capture_stage_sizes=(4, 8)),
        )
        for role, offer in zip(("ffn", "attention", "attention"), offers)
    ]
    if failure == "mode":
        offers[0]["mode"], offers[1]["mode"] = "fixed", "direct"
    elif failure == "geometry":
        offers[0]["geometry"] = {"experts": 32}
    elif failure == "capacity":
        offers[1]["eager_capacity"] = 16
    elif failure == "capture":
        descriptors[1]["runtime_contract"]["capture_stage_sizes"] = (16,)
    elif failure == "unsupported":
        offers[0].update(available=False, reason="model_family")
        offers[1]["mode"] = "direct"
    if failure:
        with pytest.raises(
            AFDError,
            match={
                "mode": "POLICY_CONFLICT",
                "geometry": "GEOMETRY_MISMATCH",
                "capacity": "CAPACITY_MISMATCH",
                "capture": "CAPTURE_PLAN_MISMATCH",
                "unsupported": "PEER_UNSUPPORTED",
            }[failure],
        ):
            negotiate_direct_capabilities(descriptors)
    else:
        assert negotiate_direct_capabilities(descriptors) == dict(
            enabled=True, reason="qualified", capture_sizes=(4, 8), buffer_sizes=(4, 8)
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
