"""CPU constructor/wire/packing tests; CUDA transport is separately qualified."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd.config import AFDConfig
from sglang.srt.afd.contracts import (
    AFDError,
    AFDRole,
    AFDStepDescriptor,
)
from sglang.srt.afd.expert_exchange import AFDDirectExchange
from sglang.srt.afd.ffn_backend import direct_step_reason
from sglang.srt.afd.transport import AFDPairedP2PTransport
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def descriptor(role, *, mode="auto", available=True, sizes=(8, 16, 32), top_k=3):
    from sglang.srt.afd.integration import _model_descriptor
    from sglang.srt.afd.profiles import QWEN3_PAIRED_C1

    geometry = dict(hidden_size=12, num_layers=2, num_experts=48, top_k=top_k)
    return _model_descriptor(
        role=AFDRole(role),
        dtype=torch.bfloat16,
        profile=QWEN3_PAIRED_C1,
        adapter=SimpleNamespace(
            num_layers=2, hidden_size=12, attention_capability=lambda **kw: {}
        ),
        config=AFDConfig(
            ffn_backend=mode,
            experimental_direct_stage_rows=32 if role == "attention" else 7,
        ),
        capture_sizes=sizes,
        direct_capability=dict(
            mode=mode,
            available=available,
            reason="qualified" if available else "native_expert_placement",
            geometry=geometry,
        ),
    )


def negotiate(rows):
    transport = AFDPairedP2PTransport.__new__(AFDPairedP2PTransport)
    transport._topology = SimpleNamespace(coordination_world_size=len(rows))
    transport._validate_descriptors(descriptors=rows, local_descriptor=rows[0])
    return transport


@pytest.mark.parametrize("a,f", [(12, 4), (8, 4), (5, 3), (3, 4), (1, 8)])
def test_actual_startup_validator_negotiates_role_local_plans(a, f):
    rows = [descriptor("ffn", sizes=(3,)) for _ in range(f)] + [
        descriptor("attention") for _ in range(a)
    ]
    transport = negotiate(rows)
    assert transport.direct_enabled
    assert transport.capture_sizes == transport.direct_buffer_sizes == (8, 16, 32)


def test_one_explicit_legacy_fixed_peer_disables_auto_without_duplicate_knobs():
    transport = negotiate(
        [descriptor("ffn", mode="fixed", available=False), descriptor("attention")]
    )
    assert not transport.direct_enabled
    assert AFDConfig(experimental_direct=False).direct_mode == "fixed"
    assert AFDConfig().direct_mode == "auto"
    assert AFDConfig(ffn_backend="direct").direct_mode == "direct"


def test_forced_direct_checks_peer_capability_and_geometry():
    with pytest.raises(AFDError, match="PEER_UNSUPPORTED"):
        negotiate(
            [descriptor("ffn", available=False), descriptor("attention", mode="direct")]
        )
    with pytest.raises(AFDError, match="GEOMETRY_MISMATCH"):
        negotiate([descriptor("ffn", top_k=4), descriptor("attention")])
    with pytest.raises(AFDError, match="POLICY_CONFLICT"):
        negotiate(
            [descriptor("ffn", mode="fixed"), descriptor("attention", mode="direct")]
        )


def test_step_selection_uses_shared_rows_capacity_and_explicit_reasons():
    config = AFDConfig(lanes=4, attention_lanes=12)
    backend = SimpleNamespace(max_rows=32)
    rows = ((0, 0), (1, 0), (7, 12), (32, 4)) * 3
    assert direct_step_reason(config, rows, False, backend=backend) == "direct"
    assert direct_step_reason(config, rows, True, backend=backend) == "prefill"
    assert (
        direct_step_reason(config, ((0, 0),) * 12, False, backend=backend)
        == "empty_step"
    )
    assert (
        direct_step_reason(config, ((33, 1),) * 12, False, backend=backend)
        == "capacity_overflow"
    )
    descriptor = AFDStepDescriptor(
        kind="STEP",
        step_id=1,
        lane_stage_rows=rows,
        hidden_size=12,
        dtype="bfloat16",
        num_layers=2,
        graph_eligible=False,
        uses_direct=True,
    )
    descriptor.validate_mode()
    with pytest.raises(AFDError, match="DIRECT_STEP"):
        AFDStepDescriptor(
            kind="STEP",
            step_id=1,
            lane_stage_rows=((-1, 0),),
            hidden_size=12,
            dtype="bfloat16",
            num_layers=2,
            graph_eligible=False,
            uses_direct=True,
        )


def make_exchange(monkeypatch, *, a, f, role, widths=(4, 8)):
    # Only CUDA communication construction is replaced. The production
    # constructor allocates/checks every real torch buffer and native axis.
    monkeypatch.setattr(
        AFDDirectExchange,
        "_make_binding",
        lambda self, **kw: SimpleNamespace(_stream=object()),
    )
    experts = SimpleNamespace(
        moe_tp_size=1, moe_ep_size=f, num_local_experts=48 // f, reduce_results=False
    )
    layers = [SimpleNamespace(mlp=SimpleNamespace(experts=experts)) for _ in range(2)]
    adapter = SimpleNamespace(
        hidden_size=12,
        num_layers=2,
        inner=SimpleNamespace(
            layers=layers, config=SimpleNamespace(num_experts=48, num_experts_per_tok=3)
        ),
    )
    transport = SimpleNamespace(_device="cpu", role=role, direct_buffer_sizes=widths)
    config = AFDConfig(lanes=f, attention_lanes=a)
    return AFDDirectExchange(
        transport=transport, adapter=adapter, config=config, device="cpu"
    )


@pytest.mark.parametrize("a,f", [(12, 4), (8, 4), (5, 3), (2, 4)])
@pytest.mark.parametrize("role", [AFDRole.ATTENTION, AFDRole.FFN])
def test_real_constructor_derives_native_axes_and_all_bucket_storage(
    monkeypatch, a, f, role
):
    plane = make_exchange(monkeypatch, a=a, f=f, role=role)
    assert plane.top_k == 3 and plane.hidden == 12 and plane.num_experts == 48
    assert plane.max_rows == 8 and set(plane.bucket_buffers) == {4, 8}
    assert plane.hbm_bytes == sum(
        t.numel() * t.element_size()
        for buckets in plane.bucket_buffers.values()
        for fields in buckets
        for t in fields
    )
    for width, buckets in plane.bucket_buffers.items():
        for fields in buckets:
            assert fields[0].shape == (a if role == AFDRole.FFN else f, width, 12)
            assert fields[-2].shape[-1] == 3
    pointers = {
        width: buckets[0][0].data_ptr()
        for width, buckets in plane.bucket_buffers.items()
    }
    for width in [4, 8, 4]:
        shape = SimpleNamespace(
            lane_bucket_rows=((width, width),) * a, lane_stage_rows=((width, 0),) * a
        )
        plane.prepare_step(shape=shape)
        assert plane.buffers[0][0].data_ptr() == pointers[width]
        assert plane.active_stages == (True, False)


@pytest.mark.parametrize("rows", [0, 1, 3, 4])
def test_actual_dispatch_pads_partial_empty_lanes_and_preserves_output_bounds(
    monkeypatch, rows
):
    plane = make_exchange(monkeypatch, a=5, f=3, role=AFDRole.ATTENTION, widths=(4,))
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda _: None)
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *a: None)
    hidden = torch.arange(rows * 12, dtype=torch.bfloat16).reshape(rows, 12) / 20
    ids = (torch.arange(rows * 3).reshape(rows, 3) * 13) % 48
    weights = torch.full((rows, 3), 1 / 3)
    output = torch.full_like(hidden, float("nan"))
    expected_partials = []
    for f in range(3):
        coefficient = torch.where(
            (ids >= f * 16) & (ids < (f + 1) * 16), (ids + 1) * weights, 0
        ).sum(1)
        expected_partials.append((hidden.float() * coefficient[:, None]).bfloat16())

    def exchange(stage, direction, fields, *, send):
        if send:
            assert direction == 0 and len(fields) == 3
            for h, i, w in fields:
                torch.testing.assert_close(h[:rows], hidden)
                torch.testing.assert_close(i[:rows], ids)
                assert (
                    torch.all(h[rows:] == 0)
                    and torch.all(i[rows:] == -1)
                    and torch.all(w[rows:] == 0)
                )
        else:
            assert direction == 1
            for (p,), expected in zip(fields, expected_partials):
                p.zero_()
                p[:rows].copy_(expected)
        return None

    monkeypatch.setattr(plane, "_exchange", exchange)
    plane.dispatch(hidden=hidden, ids=ids, weights=weights, stage=0, output=output)
    expected = torch.stack(expected_partials).sum(0, dtype=torch.float32).bfloat16()
    torch.testing.assert_close(output, expected)
    assert output.shape == hidden.shape and torch.isfinite(output).all()


@pytest.mark.parametrize("role", [AFDRole.ATTENTION, AFDRole.FFN])
@pytest.mark.parametrize(
    "a,f,e,k", [(12, 4, 128, 8), (8, 4, 128, 8), (5, 3, 48, 3), (2, 4, 128, 2)]
)
def test_actual_capability_uses_model_geometry_and_role_native_placement(
    monkeypatch, role, a, f, e, k
):
    import sys
    from types import ModuleType

    from sglang.srt.afd.ffn_backend import validate_ffn_backend

    # Only process-global backend lookup is supplied; capability admission and
    # per-layer native metadata checks execute unchanged.
    moe = ModuleType("sglang.srt.layers.moe")
    moe.get_moe_a2a_backend = lambda: SimpleNamespace(is_none=lambda: True)
    moe.get_moe_runner_backend = lambda: SimpleNamespace(is_triton=lambda: True)
    monkeypatch.setitem(sys.modules, moe.__name__, moe)
    experts = SimpleNamespace(
        moe_tp_size=1, moe_ep_size=f, num_local_experts=e // f, reduce_results=False
    )
    mlp = SimpleNamespace(experts=experts, route=lambda hidden: hidden)
    adapter = SimpleNamespace(
        role=role,
        hidden_size=24,
        num_layers=3,
        inner=SimpleNamespace(
            config=SimpleNamespace(num_experts=e, num_experts_per_tok=k),
            layers=[SimpleNamespace(mlp=mlp)] * 3,
        ),
    )
    config = AFDConfig(lanes=f, attention_lanes=a)
    result = validate_ffn_backend(
        config=config,
        adapter=adapter,
        model_family="qwen3_moe",
        dtype=torch.bfloat16,
        capture_sizes=(4, 8),
    )
    assert result["available"]
    assert result["geometry"] == dict(
        hidden_size=24, num_layers=3, num_experts=e, top_k=k
    )
    if role == AFDRole.FFN:
        experts.reduce_results = True
    else:
        del mlp.route
    result = validate_ffn_backend(
        config=config,
        adapter=adapter,
        model_family="qwen3_moe",
        dtype=torch.bfloat16,
        capture_sizes=(4, 8),
    )
    assert not result["available"]
    assert result["reason"] == (
        "native_expert_placement" if role == AFDRole.FFN else "non_routed_layer"
    )
    with pytest.raises(AFDError, match="SCOPE_UNSUPPORTED"):
        validate_ffn_backend(
            config=AFDConfig(lanes=f, attention_lanes=a, ffn_backend="direct"),
            adapter=adapter,
            model_family="qwen3_moe",
            dtype=torch.bfloat16,
            capture_sizes=(4, 8),
        )


def test_router_retention_respects_backend_priority_and_legacy_false(monkeypatch):
    from sglang.srt.afd import model_hooks

    args = SimpleNamespace(afd_config=AFDConfig(), moe_a2a_backend="none")
    monkeypatch.setattr(model_hooks, "get_server_args", lambda: args)
    assert model_hooks.afd_a_router_enabled(model_family="qwen3_moe")
    assert not model_hooks.afd_a_router_enabled(model_family="glm5")
    args.moe_a2a_backend = "deepep"
    assert not model_hooks.afd_a_router_enabled(model_family="qwen3_moe")
    args.moe_a2a_backend = "none"
    args.afd_config = AFDConfig(experimental_direct=False)
    assert not model_hooks.afd_a_router_enabled(model_family="qwen3_moe")

    args.afd_config = SimpleNamespace(experimental_m2n=True, experimental_direct=False)
    assert model_hooks.afd_a_router_enabled(model_family="qwen3_moe")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
