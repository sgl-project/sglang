"""Native CPU stores exercise CAPTURE/READY/STEP/CLOSE without CUDA emulation."""

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd import config as afd_config
from sglang.srt.afd import contracts
from sglang.srt.afd import transport as afd_transport
from sglang.srt.afd.contracts import AFDError, AFDPairedTopology, AFDStepDescriptor
from sglang.srt.afd.transport import AFDPairedP2PTransport, _AFDControlChannels
from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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


class FakeDeviceCuda:
    current_index = 2
    count = 4

    @classmethod
    def current_device(cls):
        return cls.current_index

    @classmethod
    def device_count(cls):
        return cls.count


def _install_transport_fakes(monkeypatch):
    seen = SimpleNamespace(
        communicator=[],
        max_ctas=[],
        stream=[],
        current_stream=[],
        tensor=[],
        groups=[],
        recorded_on=[],
        current_waited_on=[],
        stream_waited_on=[],
        capturing=[False],
        timeouts=[],
        wire=[],
        lifetime=[],
        timeline=[],
        empty_cache=[],
        aborted=[],
    )

    class Tensor:
        def __init__(self, device, shape=(1,), dtype="bfloat16"):
            self.device = device
            self.shape = shape
            self.dtype = dtype
            self.contiguous = True

        def is_contiguous(self):
            return self.contiguous

        def record_stream(self, stream):
            assert stream.device == self.device
            seen.lifetime.append((self, stream))

        def numel(self):
            result = 1
            for value in self.shape:
                result *= value
            return result

        def element_size(self):
            return 4 if self.dtype in ("int32", "float32") else 2

    class Event:
        def record(self, stream):
            self.stream = stream
            self.wire = tuple(seen.wire)
            seen.recorded_on.append(stream)
            seen.timeline.append(("event", self))

    class Stream:
        next_handle = 10

        def __init__(self, *, device):
            self.device = device
            self.cuda_stream = self.next_handle
            Stream.next_handle += 1
            seen.stream.append(device)

        def wait_event(self, event):
            seen.stream_waited_on.append((self, event.stream))

        def synchronize(self):
            pass

    class CurrentStream:
        def wait_event(self, event):
            seen.current_waited_on.append(event.stream)

    class Cuda(FakeDeviceCuda):
        @staticmethod
        def Stream(*, device):
            return Stream(device=device)

        @staticmethod
        def Event(*, enable_timing):
            del enable_timing
            return Event()

        @staticmethod
        def current_stream(device):
            seen.current_stream.append(device)
            return CurrentStream()

        @staticmethod
        def is_current_stream_capturing():
            return seen.capturing[0]

        @staticmethod
        def stream(stream):
            del stream
            return nullcontext()

        @staticmethod
        def memory_allocated(device):
            seen.tensor.append(device)
            return 0

        @staticmethod
        def memory_reserved(device):
            seen.tensor.append(device)
            return 0

        @staticmethod
        def empty_cache():
            seen.empty_cache.append(True)

    fake_torch = types.ModuleType("torch")
    fake_torch.device = torch.device
    fake_torch.cuda = Cuda
    fake_torch.float32 = "float32"
    fake_torch.int32 = "int32"

    def zeros(count, *, dtype, device):
        seen.tensor.append(device)
        materialized = torch.device("cuda", Cuda.current_device())
        return Tensor(materialized, (count,), dtype)

    fake_torch.zeros = zeros
    fake_torch.empty_like = lambda value: Tensor(value.device, value.shape, value.dtype)

    def empty(shape, *, dtype, device):
        seen.tensor.append(device)
        return Tensor(device, shape, dtype)

    fake_torch.empty = empty

    class Group:
        @classmethod
        def create(cls, *, host, port, rank, world_size, store_timeout_seconds):
            seen.groups.append((host, port, rank, world_size, store_timeout_seconds))
            result = cls()
            result.rank = rank
            result.world_size = world_size
            result.store = SimpleNamespace(set_timeout=seen.timeouts.append)
            return result

        def all_gather_obj(self, value):
            if not isinstance(value, dict):
                return [value] * self.world_size
            peers = []
            for rank in range(self.world_size):
                item = dict(value, role="ffn" if rank < seen.lanes else "attention")
                item["execution_mode"] = item["role"]
                item.pop("capability_digest", None)
                item["capability_digest"] = contracts.contract_digest(item)
                peers.append(item)
            return peers

    class Communicator:
        def __init__(self, *, group, device, max_ctas=None):
            self.available = True
            self.disabled = True
            self.group = group
            self.comm = object()
            self.nccl = SimpleNamespace(ncclCommAbort=seen.aborted.append)
            self.device = device
            self.rank = group.rank
            self.world_size = group.world_size
            # The cap has to arrive here: process-wide env cannot express it.
            assert max_ctas is not None
            seen.max_ctas.append(max_ctas)
            seen.communicator.append(device)
            warmup = zeros(1, dtype="float32", device=device)
            assert self.device == warmup.device

        @contextmanager
        def change_state(self, enable=None):
            previous = self.disabled
            self.disabled = not enable
            try:
                yield
            finally:
                self.disabled = previous

        def send(self, tensor, dst):
            entry = ("send", self.comm, dst, tensor)
            seen.wire.append(entry)
            seen.timeline.append(entry)

        def recv(self, tensor, src):
            entry = ("recv", self.comm, src, tensor)
            seen.wire.append(entry)
            seen.timeline.append(entry)

        def group_start(self):
            seen.timeline.append(("group_start", self.comm))

        def group_end(self):
            seen.timeline.append(("group_end", self.comm))

    pynccl = types.ModuleType("sglang.srt.distributed.device_communicators.pynccl")
    pynccl.PyNcclCommunicator = Communicator
    distributed = types.ModuleType("sglang.srt.distributed.utils")
    distributed.StatelessProcessGroup = Group
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.distributed.device_communicators.pynccl",
        pynccl,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.distributed.utils",
        distributed,
    )
    return seen


def _transport(
    monkeypatch,
    *,
    role=contracts.AFDRole.ATTENTION,
    lanes=1,
    attention_lanes=None,
    lane=0,
    **config_values,
):
    seen = _install_transport_fakes(monkeypatch)
    seen.lanes = lanes
    runtime = {"capture_stage_sizes": (8, 16)}
    descriptor = dict(
        role=role.value,
        execution_mode=role.value,
        model_family="qwen3_moe",
        num_layers=2,
        hidden_size=4,
        dtype="bfloat16",
        runtime_contract=runtime,
        runtime_contract_digest=contracts.contract_digest(runtime),
    )
    descriptor["capability_digest"] = contracts.contract_digest(descriptor)
    transport = afd_transport.AFDPairedP2PTransport(
        role=role,
        lane=lane,
        device=torch.device("cuda"),
        model_descriptor=descriptor,
        topology=contracts.AFDPairedTopology.paired(
            lanes=lanes, attention_lanes=attention_lanes
        ),
        config=afd_config.AFDConfig(
            lanes=lanes, attention_lanes=attention_lanes, **config_values
        ),
    )
    return seen, transport


def test_transport_construction_uses_one_canonical_device_everywhere(
    monkeypatch,
):
    seen, transport = _transport(monkeypatch)
    canonical = transport._device
    assert canonical == torch.device("cuda", 2)
    assert all(value is canonical for value in seen.communicator)
    # Both A2E and E2A must carry the configured budget, not NCCL's own tuning.
    budget = afd_config.AFDConfig().nccl_num_channels
    assert seen.max_ctas == [budget, budget]
    assert all(value is canonical for value in seen.stream)
    assert all(value is canonical for value in seen.current_stream)
    assert all(value is canonical for value in seen.tensor)
    assert transport._a2e_binding._identity[-1] is canonical
    assert transport._e2a_binding._identity[-1] is canonical
    buffers = transport.acquire_buffers(
        key="device-regression",
        capacities=(2, 3),
        hidden_size=4,
        dtype="bfloat16",
        retain=True,
    )
    assert all(buffer.device is canonical for buffer in buffers)
    assert all(value is canonical for value in seen.tensor)


@pytest.mark.parametrize("blocked", [False, True], ids=["success", "abort-timeout"])
def test_transport_close_drops_ownership_even_when_native_abort_times_out(blocked):
    release, aborted = threading.Event(), []

    def abort(comm):
        if blocked:
            release.wait()
        aborted.append(comm)

    def native(name):
        return SimpleNamespace(
            available=True,
            disabled=False,
            comm=name,
            nccl=SimpleNamespace(ncclCommAbort=abort),
        )

    transport = object.__new__(afd_transport.AFDPairedP2PTransport)
    transport._closed = False
    transport._stats = {}
    transport._close_timeout_seconds = 0.02 if blocked else 1
    transport._a2e, transport._e2a = native("a2e"), native("e2a")
    transport._a2e_binding = transport._e2a_binding = object()
    transport._control = object()
    transport._buffer_registry = {"bucket": (object(),)}
    transport._buffer_identities = {"bucket": (1,)}
    transport._buffer_hbm = {"bucket": 1}
    started = time.monotonic()
    try:
        with (
            pytest.raises(contracts.AFDError, match="NATIVE_CLOSE_TIMEOUT")
            if blocked
            else nullcontext()
        ):
            transport.close()
    finally:
        release.set()
    if blocked:
        assert time.monotonic() - started < 0.5
    else:
        assert sorted(aborted) == ["a2e", "e2a"]
    assert transport._closed
    assert transport._a2e is None and transport._e2a is None
    assert transport._control is None
    assert transport._buffer_registry == {}


def _wire_transport(monkeypatch, *, role, peers=2):
    seen, transport = _transport(monkeypatch, role=role, attention_lanes=peers)
    seen.wire.clear()
    seen.timeline.clear()
    seen.lifetime.clear()
    seen.recorded_on.clear()
    transport._forked_streams.clear()
    return seen, transport


def test_control_store_timeout_tracks_serving_and_close_phases(monkeypatch):
    seen, transport = _wire_transport(monkeypatch, role=contracts.AFDRole.ATTENTION)
    assert seen.timeouts == [
        timedelta(seconds=transport._config.rendezvous_timeout_seconds)
    ]
    transport._control.send_obj = lambda value, dst: None
    transport._control.recv_obj = lambda src: {"event": "AFD_CAPTURE_READY"}
    transport.capture_ready()
    assert seen.timeouts[-1] == timedelta(
        seconds=transport._config.idle_timeout_seconds
    )
    transport._control.recv_obj = lambda src: {
        "event": "AFD_CLOSE_ACK",
        "usage": {"ok": True},
    }
    assert transport.exchange_close(usage={}) == {"ok": True}
    assert seen.timeouts[-1] == timedelta(
        seconds=transport._config.close_timeout_seconds
    )
    transport._control = SimpleNamespace()
    with pytest.raises(contracts.AFDError, match="CONTROL_STORE_MISSING"):
        transport._retime_control_store(1)


@pytest.mark.parametrize("capturing", [False, True])
def test_hidden_only_fanin_keeps_peer_wire_order_and_stats(monkeypatch, capturing):
    seen, transport = _wire_transport(monkeypatch, role=contracts.AFDRole.FFN)
    seen.capturing[0] = capturing
    hidden = tuple(
        transport._torch.empty((3, 8), dtype="bfloat16", device=transport._device)
        for _ in range(2)
    )
    event = transport.receive_dispatch(hidden)
    expected = [
        ("recv", transport._a2e.comm, peer, tensor)
        for peer, tensor in zip((1, 2), hidden)
    ]
    assert seen.wire == expected
    assert event.wire == tuple(expected)
    assert transport._stats["a2e_recv"] == 2
    transport.return_result(hidden)
    assert seen.wire[2:] == [
        ("send", transport._e2a.comm, peer, tensor)
        for peer, tensor in zip((1, 2), hidden)
    ]
    assert transport._stats["e2a_send"] == 2
    assert not any(entry[0].startswith("group_") for entry in seen.timeline)
    assert event.stream is transport._a2e_stream
    assert transport._forked_streams == (
        [transport._a2e_stream, transport._e2a_stream]
        if capturing
        else [transport._e2a_stream]
    )


@pytest.mark.parametrize("role", [contracts.AFDRole.ATTENTION, contracts.AFDRole.FFN])
def test_new_receive_backing_orders_first_write_without_per_layer_wait(
    monkeypatch, role
):
    seen, transport = _wire_transport(monkeypatch, role=role)
    seen.stream_waited_on.clear()
    args = dict(
        key="new",
        capacities=(2, 3),
        hidden_size=8,
        dtype="bfloat16",
        retain=True,
    )
    buffers = transport.acquire_buffers(**args)
    incoming = (
        transport._a2e_stream
        if role == contracts.AFDRole.FFN
        else transport._e2a_stream
    )
    assert len(seen.stream_waited_on) == 1
    assert seen.stream_waited_on[0][0] is incoming
    assert transport.acquire_buffers(**args) is buffers
    for _ in range(3):
        if role == contracts.AFDRole.FFN:
            transport.receive_dispatch(buffers)
        else:
            transport.receive_return(buffers[0])
    assert len(seen.stream_waited_on) == 1
    assert not any(entry[0] == "synchronize" for entry in seen.timeline)
    transport.acquire_buffers(**dict(args, retain=False))
    assert len(seen.stream_waited_on) == 2
    transport.release_buffers(key="new")
    transport.acquire_buffers(**args)
    assert len(seen.stream_waited_on) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
