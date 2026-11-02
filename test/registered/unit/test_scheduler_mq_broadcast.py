"""Check existing routing and rank semantics across the optional transport."""

from types import SimpleNamespace

import pytest

from sglang.srt.managers.scheduler_components import request_receiver as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


@pytest.mark.parametrize("enabled,available", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("rank,payload", [(4, []), (4, ["work", "control"]), (7, None)])
def test_transport_selection_and_global_vs_local_source(monkeypatch, enabled, available, rank, payload):
    calls = []
    received = ["delivered"]
    group = SimpleNamespace(rank=rank, ranks=[4, 5, 6, 7], mq_broadcaster=object() if available else None)
    def legacy(data, global_rank, cpu_group, src):
        calls.append(("gloo", data, global_rank, cpu_group, src))
        return received
    def shared(data, src):
        calls.append(("mq", data, src))
        return received
    group.broadcast_object = shared
    monkeypatch.setattr(module, "broadcast_pyobj", legacy)
    with module.envs.SGLANG_SCHEDULER_MQ_BROADCAST.override(enabled):
        result = module.SchedulerRequestReceiver._broadcast_in_group(payload, group, "cpu4")
    assert result is received
    assert calls == ([("mq", payload, 0)] if enabled and available
        else [("gloo", payload, rank, "cpu4", 4)])


@pytest.mark.parametrize("local_control", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("leader", [False, True])
def test_dpa_preserves_work_control_group_order(local_control, cp_size, leader):
    calls = []
    received = ["work"]
    control = ["control"]
    def broadcast(data, group, cpu_group):
        calls.append((group, data))
        if group == "global":
            return control
        # Calls for work precede calls for control in the receiver.
        return received if len(calls) <= cp_size else control
    receiver = SimpleNamespace(
        ps=SimpleNamespace(attn_tp_rank=0 if leader else 1, attn_cp_rank=0,
            attn_tp_size=4, attn_cp_size=cp_size, tp_size=8),
        server_args=SimpleNamespace(enable_dp_attention=True,
            enable_dp_attention_local_control_broadcast=local_control),
        _split_work_and_control_reqs=lambda reqs: (received, control),
        _broadcast_in_group=broadcast,
        attn_tp_group="attention", attn_tp_cpu_group=None,
        attn_cp_group="context", attn_cp_cpu_group=None,
        tp_group="global", tp_cpu_group=None,
    )
    result = module.SchedulerRequestReceiver._broadcast_reqs_across_ranks(receiver, ["input"] if leader else None)
    work_groups = ["attention"] + (["context"] if cp_size == 2 else [])
    expected_groups = work_groups + (work_groups if local_control else ["global"])
    assert [g for g, _ in calls] == expected_groups
    assert calls[0][1] == (received if leader else None)
    assert result == ["work", "control"]


def test_ordinary_dp_keeps_one_tp_group_broadcast():
    calls = []
    def broadcast(data, group, cpu_group):
        calls.append((data, group, cpu_group))
        return data
    receiver = SimpleNamespace(server_args=SimpleNamespace(enable_dp_attention=False),
        ps=SimpleNamespace(tp_size=4), tp_group="replica_tp", tp_cpu_group="cpu",
        _broadcast_in_group=broadcast)
    result = module.SchedulerRequestReceiver._broadcast_reqs_across_ranks(receiver, ["request"])
    assert result == ["request"]
    assert calls == [(["request"], "replica_tp", "cpu")]
