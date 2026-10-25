"""CPU protocol doubles; the separate native Gloo run proves communication."""

import copy
import json
import sys
from dataclasses import replace
from types import ModuleType

import pytest
from sglang_simulator.simulation.sglang.dspark import CommittedHistory, DSparkTopology
from sglang_simulator.simulation.sglang.dspark_native_peers import (
    capture_native_dp_peer_step,
)
from sglang_simulator.simulation.sglang.dspark_runtime import canonical
from sglang_simulator.time_predictor.base import PredictorError
from test_dspark_runtime import batch
from test_dspark_runtime import runtime as runtime_fixture


@pytest.fixture(name="runtime")
def shared_runtime(monkeypatch):
    return runtime_fixture.__wrapped__(monkeypatch)


@pytest.fixture
def peers(runtime, monkeypatch):
    runtime.topology = DSparkTopology(2, 2)

    class NativeCPUGroup:
        world_size = 2
        rank_in_group = 0
        cpu_group = object()
        calls = 0
        transform = None

        def __init__(self):
            self.ranks = [0, 1]

        def all_gather_object(self, value):
            self.calls += 1
            rows = [value, peer_payload()]
            if self.transform is not None:
                self.transform(rows)
            return rows

    # Only the transport/type boundary is doubled here, not claimed native.
    module = ModuleType("sglang.srt.distributed.parallel_state")
    module.GroupCoordinator = NativeCPUGroup
    monkeypatch.setitem(sys.modules, module.__name__, module)
    group = NativeCPUGroup()
    peer_batch = batch(prefix=8192, extend=37)
    peer_batch.reqs[0].rid = "peer"

    def peer_payload():
        local = runtime._capture_local_group(peer_batch, attention_dp_rank=1)
        local = replace(
            local,
            execution_json=canonical(
                {
                    "selection": "native_peer_before_work_unresolved",
                    "draft_buffer_capacity": 4,
                    "draft_buffer_device_matches": True,
                }
            ),
        )
        return {
            "forward_id": runtime._forward_id,
            "native_forward_ct": 7,
            "runtime_profile_json": runtime.runtime_profile_json,
            "group": local,
        }

    return group, peer_batch


def capture(runtime, group, local=None):
    return capture_native_dp_peer_step(
        runtime,
        batch() if local is None else local,
        tp_group=group,
        attention_dp_rank=0,
        native_forward_ct=7,
    )


def unchanged(runtime, before):
    assert runtime.session._history == before
    assert runtime.session.cost_model.calls == runtime.session.progress_model.calls == 0
    assert runtime.session._last_forward_id == -1
    assert runtime._forward_id == runtime._draft_buffer_capacity == 0


def test_complete_before_work_keeps_actual_ordered_peer_and_independent_buffers(
    runtime, peers
):
    group, _ = peers
    result = capture(runtime, group)
    assert result.forward_id == 0 and group.calls == 1
    assert tuple(r.rid for r in result.requests) == ("a", "peer")
    assert tuple(r.target_prefix_len for r in result.requests) == (0, 8192)
    assert tuple(r.extend_len for r in result.requests) == (512, 37)
    assert tuple(
        json.loads(row.execution_json)["draft_buffer_capacity"] for row in result.groups
    ) == (0, 4)
    unchanged(runtime, {})


@pytest.mark.parametrize("idle_rank", [0, 1])
def test_idle_is_the_actual_empty_native_partition(runtime, peers, idle_rank):
    group, peer_batch = peers
    local = batch()
    idle = local if idle_rank == 0 else peer_batch
    idle.forward_mode.name = "IDLE"
    idle.reqs = []
    result = capture(runtime, group, local)
    assert result.groups[idle_rank].forward_mode == "IDLE"
    assert result.groups[idle_rank].requests == ()
    assert len(result.requests) == 1
    unchanged(runtime, {})


def test_native_dp_prefill_flag_is_not_misclassified_as_mixed(runtime, peers):
    group, peer = peers
    local = batch()
    local.is_extend_in_batch = peer.is_extend_in_batch = True
    result = capture(runtime, group, local)
    assert tuple(g.forward_mode for g in result.groups) == ("EXTEND", "EXTEND")
    unchanged(runtime, {})


def test_decode_reads_each_native_prefix_and_existing_causal_history(runtime, peers):
    group, peer = peers
    local = batch("DECODE", prefix=512)
    peer.forward_mode.name = "DECODE"
    peer.seq_lens_cpu = peer.seq_lens = [8192]
    peer.reqs[0].kv.kv_committed_len = 8192
    runtime.session.record_history("a", CommittedHistory(512, 3, 512))
    runtime.session.record_history("peer", CommittedHistory(8192, 100, 8192))
    before = dict(runtime.session._history)
    result = capture(runtime, group, local)
    assert tuple(r.target_prefix_len for r in result.requests) == (512, 8192)
    assert tuple(r.draft_valid_from for r in result.requests) == (3, 100)
    assert tuple(r.extend_len for r in result.requests) == (6, 6)
    unchanged(runtime, before)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "fid",
        "native_fid",
        "bool_fid",
        "runtime",
        "rank",
        "group",
        "unknown",
        "buffer",
        "bool_buffer",
        "local_changed",
    ],
)
def test_mismatched_or_incomplete_peer_rejects_without_any_state(runtime, peers, fault):
    group, _ = peers

    def change(rows):
        peer = rows[1]
        if fault == "missing":
            rows.pop()
        elif fault in {"fid", "bool_fid"}:
            peer["forward_id"] = 1 if fault == "fid" else False
        elif fault == "native_fid":
            peer["native_forward_ct"] = 8
        elif fault == "runtime":
            peer["runtime_profile_json"] = '{"different":"runtime"}'
        elif fault == "rank":
            peer["group"] = replace(peer["group"], attention_dp_rank=0)
        elif fault == "group":
            peer["group"] = None
        elif fault == "unknown":
            peer["approved"] = True
        elif fault in {"buffer", "bool_buffer"}:
            execution = json.loads(peer["group"].execution_json)
            execution["draft_buffer_capacity"] = -1 if fault == "buffer" else True
            peer["group"] = replace(peer["group"], execution_json=canonical(execution))
        else:
            rows[0] = copy.deepcopy(rows[0])
            rows[0]["group"] = replace(
                rows[0]["group"], requests=peer["group"].requests
            )

    group.transform = change
    with pytest.raises(PredictorError):
        capture(runtime, group)
    assert group.calls == 1
    unchanged(runtime, {})


def test_local_capture_error_still_rendezvous_then_propagates_original(runtime, peers):
    group, _ = peers
    local = batch()
    local.reqs[0].grammar = object()
    with pytest.raises(PredictorError, match="grammar"):
        capture(runtime, group, local)
    assert group.calls == 1
    unchanged(runtime, {})


def test_actual_mixed_partitions_remain_rejected(runtime, peers):
    group, peer = peers
    peer.forward_mode.name = "DECODE"
    peer.seq_lens_cpu = [8192]
    peer.reqs[0].kv.kv_committed_len = 8192
    runtime.session.record_history("peer", CommittedHistory(8192, 100, 8192))
    before = dict(runtime.session._history)
    with pytest.raises(PredictorError, match="mixed"):
        capture(runtime, group)
    unchanged(runtime, before)


def test_public_runtime_and_individual_cost_resolution_stay_closed(runtime, peers):
    with pytest.raises(PredictorError, match="actual peer"):
        runtime.capture(batch())
    with pytest.raises(PredictorError, match="joint worker cost"):
        runtime._capture_local_group(batch(), attention_dp_rank=0, resolve=True)
    unchanged(runtime, {})


def test_single_physical_rank_cannot_claim_an_unobserved_peer(runtime, peers):
    group, _ = peers
    group.world_size = 1
    with pytest.raises(PredictorError, match="two-rank"):
        capture(runtime, group)
    assert group.calls == 0
    unchanged(runtime, {})
