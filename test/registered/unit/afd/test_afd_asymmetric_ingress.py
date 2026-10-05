"""A<F ownership, complete expert sums and zero-ingress graph contracts."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
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
from sglang.srt.afd.transport import AFDPairedP2PTransport, _AFDControlChannels
from sglang.srt.distributed.utils import StatelessProcessGroup


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
    values, calls = [None] * f, []
    state = threading.local()

    class Group:
        def __init__(self, rank):
            self.rank_in_group = rank
            self.step = 0

        def collect(self, tensor, *, reduce=False):
            values[self.rank_in_group] = tensor
            barrier.wait()
            result = torch.stack(values).sum(0) if reduce else torch.cat(values)
            # Finish reading every rank before the next collective overwrites slots.
            barrier.wait()
            return result

        def all_gatherv(self, local, sizes):
            assert local.shape == (sizes[self.rank_in_group], 4)
            return [self.collect(local)]

        def reduce_scatterv(self, partial, sizes):
            result = self.collect(partial, reduce=True).narrow(
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


def _store_transports(a, f, *, timeout=3):
    topology = AFDPairedTopology.paired(lanes=f, attention_lanes=a)
    store = torch.distributed.HashStore()
    store.set_timeout(timedelta(seconds=timeout))
    transports = []
    for endpoint in topology.endpoints:
        rank = endpoint.coordination_rank
        transport = object.__new__(AFDPairedP2PTransport)
        transport.role, transport._closed = endpoint.role, False
        peers = topology.peers(role=endpoint.role, ordinal=endpoint.ordinal)
        transport._peer_coordination_ranks = tuple(p.coordination_rank for p in peers)
        transport._control_upstream = 0 if rank < f and not peers else None
        transport._control_followers = (
            tuple(
                j for j in range(f) if not topology.attention_lane_group(ffn_ordinal=j)
            )
            if rank == 0
            else ()
        )
        transport._control = _AFDControlChannels(
            StatelessProcessGroup(rank=rank, world_size=a + f, store=store)
        )
        transport._forked_streams, transport._stats = [], {"steps": 0}
        transport._peer_close_usage = None
        transport._close_exchange = transport._close_exchange_error = None
        transport._idle_timeout_seconds = transport._close_timeout_seconds = timeout
        transports.append(transport)
    return topology, transports


def _store_descriptor(a, kind, step_id=0, **overrides):
    fields = dict(
        kind=kind,
        step_id=step_id,
        lane_stage_rows=((1, 1),) * a,
        hidden_size=4,
        dtype="bfloat16",
        num_layers=2,
        graph_eligible=True,
    )
    fields.update(overrides)
    return AFDStepDescriptor(**fields)


@pytest.mark.parametrize(
    "a,f", [(2, 4), (3, 4), (20, 32), (1, 4), (4, 1), (1, 1), (6, 4)]
)
def test_control_lifecycle_uses_real_store_with_separate_wire_metadata(a, f):
    """Replay capture/data boundaries, READY, STEP and CLOSE on native stores.

    The wire group exchanges CPU metadata at the places GPU work completes;
    this exercises its native broadcast namespace, not CUDA/NCCL execution.
    """
    topology, transports = _store_transports(a, f)
    topology.validate()
    groups = [topology.attention_lane_group(ffn_ordinal=j) for j in range(f)]
    assert tuple(i for group in groups for i in group) == tuple(range(a))
    assert max(map(len, groups)) - min(map(len, groups)) <= 1
    assert topology.pair_world_size == (a + f if a % f else 1 + a // f)
    wire_stores = {}
    wire_groups = []
    for endpoint in topology.endpoints:
        ordinal = topology.group_ordinal(role=endpoint.role, ordinal=endpoint.ordinal)
        store = wire_stores.setdefault(ordinal, torch.distributed.HashStore())
        store.set_timeout(timedelta(seconds=3))
        wire_groups.append(
            StatelessProcessGroup(
                rank=endpoint.transport_rank,
                world_size=topology.pair_world_size,
                store=store,
            )
        )

    def data_boundary(rank, kind, step_id):
        replies = wire_groups[rank].all_gather_obj((rank, kind, step_id))
        assert len(replies) == topology.pair_world_size
        assert all(reply[1:] == (kind, step_id) for reply in replies)
        assert len({reply[0] for reply in replies}) == topology.pair_world_size

    def run(rank):
        transport = transports[rank]
        for step_id in (0, 1):
            expected = _store_descriptor(a, "CAPTURE", step_id)
            descriptor = transport.begin_step(expected if rank >= f else None)
            assert descriptor == expected
            data_boundary(rank, "CAPTURE", step_id)
        if rank < f:
            assert transport.begin_step(None).kind == "READY"
        transport.capture_ready()
        descriptor = transport.begin_step(
            _store_descriptor(a, "STEP", 2) if rank >= f else None
        )
        assert (descriptor.kind, descriptor.step_id) == ("STEP", 2)
        data_boundary(rank, "STEP", 2)
        if rank < f:
            assert transport.begin_step(None).kind == "CLOSE"
        result = transport.exchange_close(usage={"rank": rank})
        assert transport.exchange_close(usage={}) == result
        return result

    with ThreadPoolExecutor(max_workers=a + f) as pool:
        results = list(pool.map(run, range(a + f)))
    followers = transports[0]._control_followers
    if followers:
        assert all(results[j] == {"rank": f} for j in followers)
        assert results[f]["control_followers"] == {
            str(rank): {"rank": rank} for rank in followers
        }
    for j in range(f):
        group = topology.attention_lane_group(ffn_ordinal=j)
        if group:
            assert results[j] == {"rank": f + group[0]}
            assert all(results[f + i]["rank"] == j for i in group)
    assert all(transport._stats["steps"] == 5 for transport in transports[:f])


@pytest.mark.parametrize("a,f", [(2, 4), (3, 4), (20, 32)])
@pytest.mark.parametrize("late", [False, True])
def test_ready_cannot_reuse_one_followers_ack_for_a_missing_peer(a, f, late):
    _, transports = _store_transports(a, f, timeout=3 if late else 0.05)
    leader, attention = transports[0], transports[f]
    for step_id in (0, 1):
        attention.begin_step(_store_descriptor(a, "CAPTURE", step_id))
        assert leader.begin_step(None).kind == "CAPTURE"
    attention.begin_step(_store_descriptor(a, "READY", -1))
    assert leader.begin_step(None).kind == "READY"
    missing = leader._control_followers[-1]
    for follower in leader._control_followers[:-1]:
        transports[follower].capture_ready()
    # Native retained keys and earlier followers' identical ACKs must not
    # satisfy this wait. HashStore supplies a real bounded store timeout.
    if late:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(leader.capture_ready)
            with pytest.raises(TimeoutError):
                pending.result(timeout=0.02)
            assert not leader._control.store.check([f"afd-control/0/{f}/send_to/{f}/0"])
            transports[missing].capture_ready()
            pending.result(timeout=3)
        assert attention._control.recv_obj(src=0) == {"event": "AFD_CAPTURE_READY"}
    else:
        with pytest.raises(torch.distributed.DistStoreError, match="Wait timeout"):
            leader.capture_ready()
        assert not leader._control.store.check([f"afd-control/0/{f}/send_to/{f}/0"])


@pytest.mark.parametrize("a,f", [(4, 1), (6, 4)])
@pytest.mark.parametrize("drift", [False, True])
def test_real_store_fanin_keeps_lane_eligibility_and_descriptor_drift(a, f, drift):
    topology, transports = _store_transports(a, f)
    group = topology.attention_lane_group(ffn_ordinal=0)
    assert len(group) > 1
    for ordinal in group:
        transports[f + ordinal].begin_step(
            _store_descriptor(
                a,
                "STEP",
                7 if drift and ordinal else 0,
                graph_eligible=ordinal == group[0],
            )
        )
    if drift:
        with pytest.raises(AFDError, match="AFD_TRANSPORT_LANE_GROUP_DESCRIPTOR_DRIFT"):
            transports[0].begin_step(None)
    else:
        assert not transports[0].begin_step(None).graph_eligible


def test_ready_bad_payload_reports_source_and_kind_without_relaxing_validation():
    _, transports = _store_transports(3, 4)
    transports[3]._control.send_obj(_store_descriptor(3, "CAPTURE"), dst=0)
    with pytest.raises(
        AFDError, match="source=3 type=AFDStepDescriptor kind='CAPTURE'"
    ):
        transports[0].capture_ready()


@pytest.mark.parametrize("reply", [None, {"event": "WRONG", "usage": {}}])
def test_empty_f_invalid_close_ack_fails_without_acknowledging_a(reply):
    _, transports = _store_transports(3, 4, timeout=0.05)
    transport, attention = transports[0], transports[4]
    attention._control.send_obj(
        _store_descriptor(3, "CLOSE", -1, close_usage={"rank": 4}), dst=0
    )
    assert transport.begin_step(None).kind == "CLOSE"
    if reply is not None:
        transports[3]._control.send_obj(reply, dst=0)
    with pytest.raises(AFDError) as first:
        transport.exchange_close(usage={"rank": 0})
    with pytest.raises(AFDError) as repeated:
        transport.exchange_close(usage={"rank": 0})
    assert repeated.value is first.value
    assert not attention._control.store.check(["afd-control/0/4/send_to/4/0"])


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
