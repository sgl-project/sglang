"""A<F ownership, complete expert sums and zero-ingress graph contracts."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt import runtime_context
from sglang.srt.afd.cache import AFDShapeCache, make_shape
from sglang.srt.afd.config import AFDConfig
from sglang.srt.afd.contracts import (
    AFDError,
    AFDPairedTopology,
    AFDRole,
    AFDStepDescriptor,
)
from sglang.srt.afd.model_adapters import base
from sglang.srt.afd.pipeline import AFDFFNPipeline
from sglang.srt.afd.role_graph import AFDRoleGraphService, TorchRoleGraphProgram
from sglang.srt.afd.transport import AFDPairedP2PTransport


@pytest.mark.parametrize("a,f", [(20, 32), (3, 4), (2, 4), (4, 4), (24, 4), (6, 4)])
def test_unique_ingress_and_shared_wire_identity(a, f):
    topology = AFDPairedTopology.paired(lanes=f, attention_lanes=a)
    topology.validate()
    groups = [topology.attention_lane_group(ffn_ordinal=j) for j in range(f)]
    assert tuple(i for group in groups for i in group) == tuple(range(a))
    assert max(map(len, groups)) - min(map(len, groups)) <= 1
    for i in range(a):
        (peer,) = topology.peers(role=AFDRole.ATTENTION, ordinal=i)
        assert i in groups[peer.ordinal]
        assert topology.local(role=AFDRole.ATTENTION, ordinal=i) in topology.peers(
            role=AFDRole.FFN, ordinal=peer.ordinal
        )
    if a % f:
        assert topology.pair_world_size == a + f
        assert [e.transport_rank for e in topology.endpoints] == list(range(a + f))
        assert all(
            topology.group_ordinal(role=e.role, ordinal=e.ordinal) == 0
            for e in topology.endpoints
        )
    else:
        assert topology.pair_world_size == 1 + a // f


@pytest.mark.parametrize("a,f", [(20, 32), (3, 4), (4, 4), (24, 4)])
def test_all_expert_shards_contribute_and_return_each_source_row_once(
    monkeypatch, a, f
):
    """Execute both S2 boundaries with blocking CPU collectives, including empty F."""
    rows = tuple((i % 4, (i + 2) % 4) for i in range(a))
    config = AFDConfig(lanes=f, attention_lanes=a)
    shapes = [
        make_shape(
            lane=j,
            lane_rows=rows,
            hidden_size=4,
            dtype="bfloat16",
            config=config,
            capture_sizes=(4,),
        )
        for j in range(f)
    ]
    inputs = [
        [
            torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) + 10 * i + stage
            for stage, n in enumerate(vector)
        ]
        for i, vector in enumerate(rows)
    ]
    barrier = threading.Barrier(f, timeout=10)
    lock = threading.Lock()
    values, merged, reduced, calls = {}, {}, {}, []
    state = threading.local()

    class Group:
        def __init__(self, rank):
            self.rank_in_group = rank
            self.step = 0

        def all_gatherv(self, local, sizes):
            key = (self.step, "gather")
            assert local.shape == (sizes[self.rank_in_group], 4)
            with lock:
                values.setdefault(key, {})[self.rank_in_group] = local
            barrier.wait()
            if self.rank_in_group == 0:
                merged[self.step] = torch.cat([values[key][j] for j in range(f)])
            barrier.wait()
            return [merged[self.step]]

        def reduce_scatterv(self, partial, sizes):
            key = (self.step, "reduce")
            with lock:
                values.setdefault(key, {})[self.rank_in_group] = partial
            barrier.wait()
            if self.rank_in_group == 0:
                reduced[self.step] = torch.stack(
                    [values[key][j] for j in range(f)]
                ).sum(0)
            barrier.wait()
            result = reduced[self.step].narrow(
                0, sum(sizes[: self.rank_in_group]), sizes[self.rank_in_group]
            )
            self.step += 1
            return result

    class Forward:
        @contextmanager
        def scoped(self, **kwargs):
            assert kwargs == {"mlp_reduce_scatter": True}
            yield

    monkeypatch.setattr(
        base, "get_parallel", lambda: SimpleNamespace(tp_group=state.group)
    )
    monkeypatch.setattr(runtime_context, "get_forward", lambda: Forward())

    def run(rank):
        state.group = Group(rank)
        shape = shapes[rank]
        capacities = shape.group_bucket_rows(ffn_ordinal=rank)
        buffers = tuple(torch.empty(n, 4) for n in capacities)
        adapter = object.__new__(base.AFDDecoderAdapter)
        adapter.role = AFDRole.FFN

        def compute(x, batch):
            calls.append((rank, state.group.step))
            # Each rank owns a distinct expert selected by global row ID.
            chosen = torch.arange(x.shape[0]) % f == rank
            return x * chosen[:, None] * (rank + 1)

        adapter.inner = SimpleNamespace(
            layers=[SimpleNamespace(compute_ffn_output=compute)]
        )
        descriptor = SimpleNamespace(stage_rows=lambda *, lane: rows[lane])
        stages, views = adapter.make_ffn_stages(
            descriptor=descriptor, buffers=buffers, lane=rank, shape=shape
        )
        group = shape.group_lanes(ffn_ordinal=rank)
        for stage in range(2):
            for tensor, lane in zip(views[stage], group):
                tensor.copy_(inputs[lane][stage])
        return [
            adapter.local_compute(
                layer=0, stage=stage, hidden_states=view, residual=None
            )[0]
            for stage, view in zip(stages, views)
        ]

    with ThreadPoolExecutor(max_workers=f) as executor:
        outputs = list(executor.map(run, range(f)))
    assert sorted(calls) == [(j, s) for j in range(f) for s in range(2)]
    for stage in range(2):
        # Independent serial oracle in global A row order, including bucket gaps.
        packed = torch.cat(
            [
                torch.nn.functional.pad(inputs[i][stage], (0, 0, 0, 4 - rows[i][stage]))
                for i in range(a)
            ]
        )
        oracle = packed * (torch.arange(a * 4) % f + 1)[:, None]
        for rank, shape in enumerate(shapes):
            lanes = shape.group_lanes(ffn_ordinal=rank)
            assert len(outputs[rank][stage]) == len(lanes)
            for output, lane in zip(outputs[rank][stage], lanes):
                torch.testing.assert_close(
                    output, oracle[lane * 4 : lane * 4 + rows[lane][stage]]
                )


def test_empty_f_graph_uses_global_bucket_and_real_work_for_sentinel():
    config = AFDConfig(lanes=32, attention_lanes=20)
    shape = make_shape(
        lane=31,
        lane_rows=((3, 2),) * 20,
        hidden_size=4,
        dtype="bfloat16",
        config=config,
        capture_sizes=(4,),
    )
    assert shape.stage_rows == shape.bucket_rows == (0, 0)
    assert shape.merge_plan(stage=0) == (4,) * 20 + (0,) * 12
    service = object.__new__(AFDRoleGraphService)
    service.role, service._num_layers = AFDRole.FFN, 1
    estimate = service._estimate_hbm_bytes(shape=shape)
    assert estimate > 0
    cache = AFDShapeCache(config=config, capture_sizes=(4,))
    selection = cache.select(shape=shape, estimated_hbm_bytes=estimate, capture=True)
    assert selection.arming
    program = object.__new__(TorchRoleGraphProgram)
    program._torch = torch
    partial = torch.tensor([[7.0, 8.0, 9.0, 10.0]])
    program._capture_sentinel(outputs=((partial,), (partial,)), device="cpu")
    assert program._sentinel_source is partial
    assert program._sentinel.item() == 7.0


def test_empty_wire_edges_do_not_skip_global_s2_compute():
    transport = object.__new__(AFDPairedP2PTransport)
    transport.role, transport._peer_ranks = AFDRole.FFN, ()
    transport._stats = {"a2e_recv": 0, "e2a_send": 0}
    transport._torch = SimpleNamespace(
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: True)
    )
    assert transport.receive_dispatch(()) is None
    transport.wait(None)
    transport.return_result(())
    with pytest.raises(AFDError, match="AFD_TRANSPORT_EDGE_COUNT_INVALID"):
        transport.return_result((torch.ones(1, 4),))
    calls = []
    stages = [SimpleNamespace(index=s, graph_output=None) for s in range(2)]

    def compute(**kwargs):
        stage = kwargs["stage"]
        calls.append((kwargs["layer"], stage.index))
        stage.graph_output = torch.full((1, 4), len(calls), dtype=torch.float32)
        return (), None

    pipeline = object.__new__(AFDFFNPipeline)
    pipeline._connector = SimpleNamespace(transport=transport)
    pipeline._adapter = SimpleNamespace(num_layers=2, local_compute=compute)
    result = pipeline._run_layers(
        stages=stages, recv_buffers=[(), ()], participates=(True, True)
    )
    assert calls == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [output[0][0, 0].item() for output in result] == [3, 4]
    assert transport._stats == {"a2e_recv": 0, "e2a_send": 0}


@pytest.mark.parametrize("kind", ["STEP", "CAPTURE", "READY", "CLOSE"])
def test_empty_f_receives_same_control_descriptor_and_ready_close_acks(kind):
    descriptor = AFDStepDescriptor(
        kind=kind,
        step_id=-1 if kind in ("READY", "CLOSE") else 1,
        lane_stage_rows=() if kind in ("READY", "CLOSE") else ((3, 2),) * 20,
        hidden_size=0 if kind in ("READY", "CLOSE") else 4,
        dtype="" if kind in ("READY", "CLOSE") else "bfloat16",
        num_layers=0 if kind in ("READY", "CLOSE") else 2,
        graph_eligible=kind in ("STEP", "CAPTURE"),
        close_usage={"owner": "A0"} if kind == "CLOSE" else None,
    )
    sent = []
    leader = object.__new__(AFDPairedP2PTransport)
    leader.role, leader._closed = AFDRole.FFN, False
    leader._peer_coordination_ranks, leader._control_followers = (
        (32,),
        tuple(range(20, 32)),
    )
    leader._control = SimpleNamespace(
        recv_obj=lambda src: descriptor,
        send_obj=lambda obj, dst: sent.append((dst, obj)),
    )
    leader._stats, leader._forked_streams = {"steps": 0}, []
    result = leader.begin_step(None)
    assert [dst for dst, _ in sent] == list(range(20, 32))
    assert all(item == result for _, item in sent)
    follower = object.__new__(AFDPairedP2PTransport)
    follower.role, follower._closed = AFDRole.FFN, False
    follower._peer_coordination_ranks, follower._control_upstream = (), 0
    follower._control = SimpleNamespace(
        recv_obj=lambda src: result, send_obj=lambda obj, dst: sent.append((dst, obj))
    )
    follower._stats, follower._forked_streams = {"steps": 0}, []
    assert follower.begin_step(None) == result
    for transport in (leader, follower):
        transport._retime_control_store = lambda timeout: None
        transport._idle_timeout_seconds, transport._close_timeout_seconds = 100, 1
        transport._close_exchange, transport._close_exchange_error = None, None
    if kind == "READY":
        follower.capture_ready()
        leader._control.recv_obj = lambda src: {"event": "AFD_CAPTURE_READY"}
        leader.capture_ready()
        assert sent[-1] == (32, {"event": "AFD_CAPTURE_READY"})
    if kind == "CLOSE":
        assert follower.exchange_close(usage={"owner": "F31"}) == {"owner": "A0"}
        leader._control.recv_obj = lambda src: {
            "event": "AFD_CLOSE_ACK",
            "usage": {"owner": src},
        }
        assert leader.exchange_close(usage={"owner": "F0"}) == {"owner": "A0"}
        assert sent[-1] == (
            32,
            {
                "event": "AFD_CLOSE_ACK",
                "usage": {
                    "owner": "F0",
                    "control_followers": {
                        str(rank): {"owner": rank} for rank in range(20, 32)
                    },
                },
            },
        )


def test_empty_f_control_lifecycle_with_bounded_message_queues():
    topology = AFDPairedTopology.paired(lanes=32, attention_lanes=20)
    channels = {(src, dst): queue.Queue() for src in range(52) for dst in range(52)}
    follower_ready = threading.Event()
    follower_closed = threading.Event()
    transports = []
    for endpoint in topology.endpoints:
        rank = endpoint.coordination_rank
        transport = object.__new__(AFDPairedP2PTransport)
        transport.role, transport._closed = endpoint.role, False
        peers = topology.peers(role=endpoint.role, ordinal=endpoint.ordinal)
        transport._peer_coordination_ranks = tuple(p.coordination_rank for p in peers)
        transport._control_upstream = 0 if rank < 32 and not peers else None
        transport._control_followers = tuple(range(20, 32)) if rank == 0 else ()

        def send(obj, dst, rank=rank):
            if rank == 31 and isinstance(obj, dict):
                if obj.get("event") == "AFD_CAPTURE_READY":
                    follower_ready.set()
                if obj.get("event") == "AFD_CLOSE_ACK":
                    follower_closed.set()
            if rank == 0 and dst == 32 and isinstance(obj, dict):
                assert (
                    follower_ready
                    if obj["event"] == "AFD_CAPTURE_READY"
                    else follower_closed
                ).is_set()
            channels[rank, dst].put(obj)

        transport._control = SimpleNamespace(
            send_obj=send,
            recv_obj=lambda src, rank=rank: channels[src, rank].get(timeout=3),
        )
        transport._forked_streams, transport._stats = [], {"steps": 0}
        transport._peer_close_usage = None
        transport._close_exchange = transport._close_exchange_error = None
        transport._idle_timeout_seconds = transport._close_timeout_seconds = 3
        transport._retime_control_store = lambda seconds: None
        transports.append(transport)

    def run(rank):
        transport = transports[rank]
        if rank >= 32:
            for kind in ("STEP", "CAPTURE"):
                descriptor = AFDStepDescriptor(
                    kind=kind,
                    step_id=1,
                    lane_stage_rows=((3, 2),) * 20,
                    hidden_size=4,
                    dtype="bfloat16",
                    num_layers=2,
                    graph_eligible=kind == "CAPTURE",
                )
                transport.begin_step(descriptor)
            transport.capture_ready()
        else:
            for kind in ("STEP", "CAPTURE", "READY", "CLOSE"):
                assert transport.begin_step(None).kind == kind
                if kind == "READY":
                    transport.capture_ready()
        result = transport.exchange_close(usage={"rank": rank})
        assert transport.exchange_close(usage={}) == result
        return result

    with ThreadPoolExecutor(max_workers=52) as pool:
        results = list(pool.map(run, range(52)))
    assert results[32]["control_followers"] == {
        str(rank): {"rank": rank} for rank in range(20, 32)
    }
    assert all(transport._stats["steps"] == 4 for transport in transports[:32])


@pytest.mark.parametrize("reply", [None, {"event": "WRONG", "usage": {}}])
def test_empty_f_invalid_close_ack_fails_without_acknowledging_a(reply):
    transport = object.__new__(AFDPairedP2PTransport)
    transport.role = AFDRole.FFN
    transport._peer_close_usage = {"rank": 32}
    transport._control_upstream, transport._control_followers = None, (31,)
    transport._peer_coordination_ranks = (32,)
    transport._close_exchange = transport._close_exchange_error = None
    transport._close_timeout_seconds = 0.01
    transport._retime_control_store = lambda seconds: None
    sent = []

    def receive(src):
        if reply is None:
            raise TimeoutError("Missing follower ACK")
        return reply

    transport._control = SimpleNamespace(
        recv_obj=receive, send_obj=lambda obj, dst: sent.append((dst, obj))
    )
    with pytest.raises(AFDError) as first:
        transport.exchange_close(usage={"rank": 0})
    with pytest.raises(AFDError) as repeated:
        transport.exchange_close(usage={"rank": 0})
    assert repeated.value is first.value
    assert not sent


def test_glm_tp1_shared_is_computed_once_before_inplace_routed(routing):
    calls = []
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    summed = torch.zeros_like(original)
    for rank in range(32):
        x = original.clone()

        def shared(hidden):
            calls.append((rank, "shared"))
            return hidden * 0.125

        def routed(hidden, *, skip_shared_experts):
            assert skip_shared_experts
            calls.append((rank, "routed"))
            hidden.fill_(1000)  # A native in-place expert is allowed to consume input.
            return torch.full_like(hidden, rank + 1.0)

        layer = SimpleNamespace(
            mlp=SimpleNamespace(
                _shared_expert_tp1=True, shared_experts=shared, forward_normal=routed
            )
        )
        routing["get_parallel"] = lambda: SimpleNamespace(tp_rank=rank)
        result = routing["GlmMoeDsaAFDDecoderLayer"].compute_ffn_output(layer, x)
        summed += result
    torch.testing.assert_close(
        summed, torch.full_like(original, sum(range(1, 33))) + original * 0.125
    )
    assert [item for item in calls if item[1] == "shared"] == [(0, "shared")]
    assert calls[:2] == [(0, "shared"), (0, "routed")]
