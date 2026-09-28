from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
import sys
import threading
import time
import types
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd import config as afd_config
from sglang.srt.afd import contracts as contracts
from sglang.srt.afd import transport as afd_transport


class FakeDeviceCuda:
    current_index = 2
    count = 4

    @classmethod
    def current_device(cls):
        return cls.current_index

    @classmethod
    def device_count(cls):
        return cls.count


def _fake_device_torch():
    return SimpleNamespace(device=torch.device, cuda=FakeDeviceCuda)


@pytest.mark.parametrize("value", ["cuda", torch.device("cuda")])
def test_abstract_cuda_device_binds_process_local_explicit_index(value):
    canonical = afd_transport._canonical_cuda_device(
        torch=_fake_device_torch(),
        device=value,
    )
    assert canonical == torch.device("cuda", 2)
    assert canonical.index is not None


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, torch.device("cuda", 0)),
        ("cuda:1", torch.device("cuda", 1)),
        (torch.device("cuda", 3), torch.device("cuda", 3)),
    ],
)
def test_indexed_cuda_device_forms_remain_stable(value, expected):
    assert (
        afd_transport._canonical_cuda_device(
            torch=_fake_device_torch(),
            device=value,
        )
        == expected
    )


@pytest.mark.parametrize(
    "value,code",
    [
        ("cpu", "AFD_TRANSPORT_CUDA_DEVICE_REQUIRED"),
        (True, "AFD_TRANSPORT_CUDA_DEVICE_INVALID"),
        (object(), "AFD_TRANSPORT_CUDA_DEVICE_INVALID"),
        ("cuda:7", "AFD_TRANSPORT_CUDA_DEVICE_UNAVAILABLE"),
    ],
)
def test_non_cuda_and_invalid_device_identity_fail_closed(value, code):
    with pytest.raises(contracts.AFDError, match=code):
        afd_transport._canonical_cuda_device(
            torch=_fake_device_torch(),
            device=value,
        )


def test_canonical_device_round_trip_mismatch_fails_closed():
    def drifting_device(value, index=None):
        return (
            torch.device(value, index + 1) if index is not None else torch.device(value)
        )

    with pytest.raises(
        contracts.AFDError,
        match="AFD_TRANSPORT_CUDA_DEVICE_IDENTITY_MISMATCH",
    ):
        afd_transport._canonical_cuda_device(
            torch=SimpleNamespace(
                device=drifting_device,
                cuda=FakeDeviceCuda,
            ),
            device="cuda:1",
        )


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
            return [value] * self.world_size

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
    monkeypatch.setattr(
        afd_transport.AFDPairedP2PTransport,
        "_validate_descriptors",
        lambda self, **kwargs: None,
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
    transport = afd_transport.AFDPairedP2PTransport(
        role=role,
        lane=lane,
        device=torch.device("cuda"),
        model_descriptor={},
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


@pytest.mark.parametrize("lanes,group", [(1, 1), (4, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("role", [contracts.AFDRole.ATTENTION, contracts.AFDRole.FFN])
def test_rendezvous_separates_coordination_world_from_wire_group(
    monkeypatch, lanes, group, role
):
    """Wire rendezvous is per FFN rank, so k lanes and their peer meet on one port."""

    base_port = 4500
    attention_lanes = group * lanes
    size = lanes if role == contracts.AFDRole.FFN else attention_lanes
    for ordinal in range(size):
        seen, transport = _transport(
            monkeypatch,
            role=role,
            lanes=lanes,
            attention_lanes=None if group == 1 else attention_lanes,
            lane=ordinal,
            rendezvous_port=base_port,
            rendezvous_timeout_seconds=1234,
        )
        if role == contracts.AFDRole.FFN:
            group_ordinal = ordinal
            expected_coordination = ordinal
            expected_transport = 0
            expected_peers = tuple(range(1, group + 1))
        else:
            group_ordinal = ordinal // group
            expected_coordination = lanes + ordinal
            expected_transport = 1 + ordinal % group
            expected_peers = (0,)
        assert seen.groups == [
            (
                "127.0.0.1",
                base_port,
                expected_coordination,
                lanes + attention_lanes,
                1234,
            ),
            (
                "127.0.0.1",
                base_port + 1 + group_ordinal,
                expected_transport,
                group + 1,
                1234,
            ),
        ]
        assert transport.lane == ordinal
        assert transport._rank == expected_transport
        assert transport._peer_ranks == expected_peers
        assert transport._coordination_rank == expected_coordination


def test_transport_rejects_a_lane_outside_the_topology(monkeypatch):
    """A mis-plumbed rank must fail at construction, not hang in the rendezvous."""

    _install_transport_fakes(monkeypatch)
    with pytest.raises(contracts.AFDError, match="AFD_TOPOLOGY_ROLE_IDENTITY_INVALID"):
        afd_transport.AFDPairedP2PTransport(
            role=contracts.AFDRole.ATTENTION,
            topology=contracts.AFDPairedTopology.paired(lanes=2),
            config=afd_config.AFDConfig(lanes=2),
            device=torch.device("cuda"),
            model_descriptor={},
            lane=2,
        )


class WarmupTorch:
    float32 = "float32"

    class cuda:
        current = None

        @classmethod
        def current_stream(cls, device):
            del device
            return cls.current

    @staticmethod
    def zeros(count, dtype, device):
        del count, dtype, device
        return object()

    @staticmethod
    def empty_like(value):
        del value
        return object()


@pytest.mark.parametrize(
    "role,edges",
    [
        (contracts.AFDRole.ATTENTION, 1),
        (contracts.AFDRole.FFN, 1),
        (contracts.AFDRole.FFN, 2),
    ],
)
def test_warmup_uses_distinct_buffers_and_ffn_happens_before(role, edges):
    """Warmup must never concurrently read and write one tensor on both streams."""

    transport = object.__new__(afd_transport.AFDPairedP2PTransport)
    transport.role = role
    transport._torch = WarmupTorch()
    transport._device = "cuda:0"
    transport._peer_ranks = tuple(range(edges))
    transport._a2e_stream = SimpleNamespace(synchronize=lambda: None)
    transport._e2a_stream = SimpleNamespace(synchronize=lambda: None)
    transport._a2e_binding = object()
    transport._e2a_binding = object()
    timeline = []
    WarmupTorch.cuda.current = SimpleNamespace(
        wait_event=lambda event: timeline.append(("wait", event))
    )
    transport._send = lambda **values: timeline.append(
        ("send", values["peer"], values["tensor"])
    )

    def receive(**values):
        timeline.append(("recv", values["peer"], values["buffer"]))
        return "received"

    transport._recv = receive
    transport._warmup()
    if role == contracts.AFDRole.ATTENTION:
        sent = next(item[2] for item in timeline if item[0] == "send")
        received = next(item[2] for item in timeline if item[0] == "recv")
        assert sent is not received
        return
    buffers = [item[2] for item in timeline if item[0] == "recv"]
    # A distinct buffer per edge, echoed back on the edge it arrived on.
    assert len(set(id(buffer) for buffer in buffers)) == edges
    assert timeline == [
        entry
        for peer, buffer in zip(range(edges), buffers)
        for entry in (
            ("recv", peer, buffer),
            ("wait", "received"),
            ("send", peer, buffer),
        )
    ]


def test_recv_records_buffer_on_comm_stream_before_nccl():
    """The allocator must retain a receive backing even when NCCL setup raises."""

    timeline = []

    class Buffer:
        def record_stream(self, stream):
            timeline.append(("record", stream))

    class Event:
        def record(self, stream):
            timeline.append(("event", stream))

    class Binding:
        def invoke(self, *, torch, operation):
            del torch, operation
            timeline.append(("invoke",))
            raise RuntimeError("sentinel")

    transport = object.__new__(afd_transport.AFDPairedP2PTransport)
    transport._torch = SimpleNamespace(
        cuda=SimpleNamespace(
            Event=lambda enable_timing: Event(),
            # The eager path is what this test is about: not capturing, so the
            # receive posts without forking its side stream.
            is_current_stream_capturing=lambda: False,
        )
    )
    stream = object()
    with pytest.raises(RuntimeError, match="sentinel"):
        transport._recv(
            buffer=Buffer(),
            peer=1,
            stream=stream,
            binding=Binding(),
        )
    assert timeline == [("record", stream), ("invoke",)]


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


def _attention_transport(monkeypatch):
    seen, transport = _transport(monkeypatch)
    transport._e2a_binding = SimpleNamespace(invoke=lambda *, torch, operation: None)
    seen.recorded_on.clear()
    seen.current_waited_on.clear()
    seen.stream_waited_on.clear()
    return seen, transport


def test_receive_forks_its_side_stream_only_while_capturing(monkeypatch):
    """A stream that joins a capture without forking into it fails the capture.

    On GB300 this was the whole-role graph's only failure: the send direction
    forked, the receive direction did not, so ncclRecv landed as uncaptured work
    and CUDA rejected the join with "dependency created on uncaptured work in
    another stream". Eagerly the fork is deliberately absent -- it would order the
    receive behind all queued compute and serialize transfer against it.
    """

    seen, transport = _attention_transport(monkeypatch)
    buffer = SimpleNamespace(
        record_stream=lambda stream: None,
        device=torch.device("cuda"),
    )

    transport.receive_return(buffer)
    assert seen.stream_waited_on == []

    seen.capturing[0] = True
    transport.receive_return(buffer)
    assert [stream for stream, _ in seen.stream_waited_on] == [transport._e2a_stream]
    # The event it waits on must come from the capturing (current) stream, not
    # from another side stream, or the fork does not enrol it in the capture.
    waited_on = seen.stream_waited_on[0][1]
    assert waited_on not in (transport._a2e_stream, transport._e2a_stream)


def test_rejoin_joins_only_the_streams_that_forked(monkeypatch):
    """Recording on a stream that never forked is itself uncaptured work.

    An unconditional loop over both directions would therefore fail the very
    capture this call exists to complete, on any step where a role touched only
    one of its two communicators.
    """

    seen, transport = _attention_transport(monkeypatch)
    seen.capturing[0] = True
    tensor = SimpleNamespace(
        record_stream=lambda stream: None,
        device=torch.device("cuda"),
    )
    transport._a2e_binding = SimpleNamespace(invoke=lambda *, torch, operation: None)

    transport.dispatch(tensor)
    seen.recorded_on.clear()
    seen.current_waited_on.clear()

    transport.rejoin_streams()

    # Only the dispatch direction was used, so only it is joined.
    assert seen.recorded_on == [transport._a2e_stream]
    assert seen.current_waited_on == [transport._a2e_stream]


def test_forked_stream_tracking_resets_each_step(monkeypatch):
    """Eager steps fork too and never rejoin, so the tracking must be per step."""

    seen, transport = _attention_transport(monkeypatch)
    transport._a2e_binding = SimpleNamespace(invoke=lambda *, torch, operation: None)
    transport._control = SimpleNamespace(send_obj=lambda obj, dst: None)
    tensor = SimpleNamespace(
        record_stream=lambda stream: None,
        device=torch.device("cuda"),
    )
    for _ in range(3):
        transport.dispatch(tensor)
    assert len(transport._forked_streams) == 1

    transport.begin_step(
        contracts.AFDStepDescriptor(
            kind="STEP",
            step_id=1,
            lane_stage_rows=((1,),),
            hidden_size=8,
            dtype="bfloat16",
            num_layers=2,
            graph_eligible=True,
        )
    )
    assert transport._forked_streams == []


def _fanin_ffn_transport(monkeypatch, *, attention_lanes):
    return _transport(
        monkeypatch,
        role=contracts.AFDRole.FFN,
        lanes=4,
        attention_lanes=None if attention_lanes == 4 else attention_lanes,
    )[1]


def _descriptor(**overrides):
    fields = dict(
        kind="STEP",
        step_id=295,
        lane_stage_rows=((64, 64), (64, 64), (64, 64), (176, 0)),
        hidden_size=6144,
        dtype="bfloat16",
        num_layers=78,
        graph_eligible=True,
    )
    fields.update(overrides)
    return contracts.AFDStepDescriptor(**fields)


def _feed(transport, descriptors):
    queue = list(descriptors)
    transport._peer_coordination_ranks = tuple(range(len(queue)))
    transport._control = SimpleNamespace(recv_obj=lambda src: queue[src])
    return transport.begin_step(None)


def test_one_lane_prefilling_does_not_kill_its_ffn_rank(monkeypatch):
    """Eligibility is a lane's own property, so a group may legitimately disagree.

    Measured at M=16: four lanes sent step_id=295 with identical rows but
    graph_eligible True/True/True/False, because that flag is
    `is_decode() and all(rows)` and forward_mode is not globally consistent under
    DP attention -- three lanes decoded while the fourth prefilled a freshly
    admitted request. Demanding equality made the FFN rank raise, exit, and wedge
    every remaining rank at 100% util until the 600 s NCCL watchdog fired.
    """

    transport = _fanin_ffn_transport(monkeypatch, attention_lanes=16)
    result = _feed(
        transport,
        [
            _descriptor(graph_eligible=True),
            _descriptor(graph_eligible=True),
            _descriptor(graph_eligible=True),
            _descriptor(graph_eligible=False),
        ],
    )
    # AND, not majority: one lane outside clean decode is enough to disarm the
    # graph, because the rank replays one capture over the whole packed group.
    assert result.graph_eligible is False
    assert result.step_id == 295
    assert result.lane_stage_rows == ((64, 64), (64, 64), (64, 64), (176, 0))

    every_lane_ready = _feed(
        transport, [_descriptor(graph_eligible=True) for _ in range(4)]
    )
    assert every_lane_ready.graph_eligible is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("step_id", 296),
        ("lane_stage_rows", ((64, 64), (64, 64), (64, 64), (64, 64))),
        ("hidden_size", 4096),
        ("dtype", "float16"),
        ("num_layers", 61),
        ("kind", "CLOSE"),
    ],
)
def test_a_group_disagreeing_about_the_step_itself_still_fails_closed(
    monkeypatch, field, value
):
    """Relaxing eligibility must not relax the fields that describe the step.

    These are what the rank sizes its buffers and its capture from, so a
    divergence means the rows about to arrive no longer match the plan.
    """

    transport = _fanin_ffn_transport(monkeypatch, attention_lanes=16)
    with pytest.raises(
        contracts.AFDError, match="AFD_TRANSPORT_LANE_GROUP_DESCRIPTOR_DRIFT"
    ):
        _feed(transport, [_descriptor(), _descriptor(**{field: value})])


def test_symmetric_ffn_rank_passes_its_single_descriptor_through(monkeypatch):
    """At k == 1 the AND is over one element, so nothing about NANF changes."""

    transport = _fanin_ffn_transport(monkeypatch, attention_lanes=4)
    for eligible in (True, False):
        sent = _descriptor(graph_eligible=eligible)
        assert _feed(transport, [sent]) == sent


def test_a_group_closing_may_report_different_usage_per_lane(monkeypatch):
    """Close usage is each lane's own graph statistics, so it cannot be required
    to match: k independent caches never agree, and demanding it would turn every
    fan-in teardown into the same wedge a live drift causes."""

    transport = _fanin_ffn_transport(monkeypatch, attention_lanes=16)
    result = _feed(
        transport,
        [
            _descriptor(kind="CLOSE", close_usage={"replays": 11}),
            _descriptor(kind="CLOSE", close_usage={"replays": 0}),
            _descriptor(kind="CLOSE", close_usage={"replays": 7}),
            _descriptor(kind="CLOSE", close_usage={"replays": 3}),
        ],
    )
    assert result.close_usage == {"replays": 11}
    assert transport._peer_close_usage == {"replays": 11}


def test_an_address_this_host_owns_is_dialed_verbatim(monkeypatch):
    """Ownership and outbound-address selection do not require network access."""
    import socket

    calls = []

    class Probe:
        def bind(self, endpoint):
            calls.append(("bind", endpoint))
            if endpoint[0] != "127.0.0.1":
                raise OSError("not locally owned")

        def connect(self, endpoint):
            calls.append(("connect", endpoint))

        def getsockname(self):
            return ("192.0.2.10", 49152)

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(socket, "socket", lambda *args: Probe())
    assert afd_transport._store_host("127.0.0.1") == "127.0.0.1"
    assert afd_transport._store_host("192.0.2.20") == "192.0.2.10"
    assert calls == [
        ("bind", ("127.0.0.1", 0)),
        ("close",),
        ("bind", ("192.0.2.20", 0)),
        ("close",),
        ("connect", ("192.0.2.20", 1)),
        ("close",),
    ]


@pytest.mark.parametrize("lanes,attention_lanes", [(4, 4), (4, 8), (8, 16), (2, 8)])
def test_a_wire_group_store_is_indexed_by_its_own_ffn_rank(lanes, attention_lanes):
    """The transport picks a wire group's store host by group ordinal, so that
    ordinal has to be the coordination rank of the FFN rank binding the store --
    rank 0 of the group. Once N outgrows one host that rank is on another machine,
    and an index that drifted here would send every lane to a dead address."""

    topology = contracts.AFDPairedTopology.paired(
        lanes=lanes, attention_lanes=attention_lanes
    )
    owner_of = {
        endpoint.ordinal: endpoint.coordination_rank
        for endpoint in topology.endpoints
        if endpoint.role == contracts.AFDRole.FFN and endpoint.transport_rank == 0
    }
    assert len(owner_of) == lanes
    for role, size in (
        (contracts.AFDRole.FFN, lanes),
        (contracts.AFDRole.ATTENTION, attention_lanes),
    ):
        for ordinal in range(size):
            group = topology.group_ordinal(role=role, ordinal=ordinal)
            assert group == owner_of[group]


def _wire_transport(monkeypatch, *, role, peers=2):
    seen, transport = _transport(monkeypatch, role=role, attention_lanes=peers)
    seen.wire.clear()
    seen.timeline.clear()
    seen.lifetime.clear()
    seen.recorded_on.clear()
    transport._forked_streams.clear()
    return seen, transport


@pytest.mark.parametrize("attention_lanes", [4, 8])
@pytest.mark.parametrize("extend", [False, True])
def test_step_mode_survives_symmetric_and_fanin_reconstruction(
    monkeypatch, attention_lanes, extend
):
    transport = _fanin_ffn_transport(monkeypatch, attention_lanes=attention_lanes)
    descriptor = _descriptor(graph_eligible=False, is_extend_in_batch=extend)
    result = _feed(transport, [descriptor] * (attention_lanes // 4))
    assert result.is_extend_in_batch is extend
    assert result.graph_eligible is False


def test_fanin_rejects_batch_mode_disagreement_even_when_both_are_eager(monkeypatch):
    transport = _fanin_ffn_transport(monkeypatch, attention_lanes=8)
    with pytest.raises(contracts.AFDError, match="LANE_GROUP_DESCRIPTOR_DRIFT"):
        _feed(
            transport,
            [
                _descriptor(graph_eligible=False, is_extend_in_batch=False),
                _descriptor(graph_eligible=False, is_extend_in_batch=True),
            ],
        )


@pytest.mark.parametrize("attention_lanes", [4, 8])
@pytest.mark.parametrize("mode", [None, 0, 1, "false"])
def test_transport_rejects_unchecked_non_boolean_step_modes(
    monkeypatch, attention_lanes, mode
):
    import msgspec

    transport = _fanin_ffn_transport(monkeypatch, attention_lanes=attention_lanes)
    descriptor = _descriptor(graph_eligible=False)
    # Simulate an unchecked object decoder, bypassing constructor validation.
    msgspec.structs.force_setattr(descriptor, "is_extend_in_batch", mode)
    with pytest.raises(contracts.AFDError, match="AFD_STEP_DESCRIPTOR_MODE_INVALID"):
        _feed(transport, [descriptor] * (attention_lanes // 4))


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
