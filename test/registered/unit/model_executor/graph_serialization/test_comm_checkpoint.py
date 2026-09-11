"""CPU unit tests for the communicator checkpoint (graph_serialization/comm.py).

Everything the CUDA driver would do is replaced by fakes: the custom all-reduce
object, the region registry (``classify``), the relocation map (``rebase``)
and the pynccl communicator. The real ``CustomAllReduceV2`` entry points are
exercised on an instance built with ``__new__`` and a stand-in ``torch``
namespace so no CUDA tensor is ever created.
"""

import logging
import sys
from types import SimpleNamespace
from unittest import mock

import msgspec
import pytest

from sglang.srt.distributed.device_communicators import custom_all_reduce_v2
from sglang.srt.distributed.device_communicators import pynccl as pynccl_mod
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    _MAX_GRAPH_INPUTS,
    CustomAllReduceV2,
)
from sglang.srt.distributed.device_communicators.pynccl import (
    NcclCommDescriptor,
    PyNcclCommunicator,
)
from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.srt.model_executor.graph_serialization.comm import (
    NCCL_KERNEL_PREFIXES,
    CommCheckpoint,
    CommCheckpointer,
    CommIdentity,
    CommWindowRecord,
    CustomAllReduceV1GraphState,
    CustomAllReduceV2GraphState,
    PyNcclCommGraphState,
    UnsupportedCommGraphState,
    collect_comm_graph_states,
    is_nccl_kernel,
)
from sglang.srt.model_executor.graph_serialization.format import (
    CommStateBlob,
    RegionKind,
    RegionRef,
    RegionSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GROUP = "tp:0"

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, data_ptr, nbytes):
        self._ptr = data_ptr
        self.nbytes = nbytes

    def data_ptr(self):
        return self._ptr


class _FakeRegistry:
    """``classify`` by bisect over ``{region_id: (base, nbytes)}``."""

    def __init__(self, regions):
        self._regions = regions

    def classify(self, word):
        for region_id, (base, nbytes) in self._regions.items():
            if base <= word < base + nbytes:
                return RegionRef(region_id, word - base)
        return None


class _FakeReloc:
    """``rebase`` = live base of the region + offset."""

    def __init__(self, base_by_region):
        self._bases = base_by_region

    def rebase(self, ref):
        return self._bases[ref.region_id] + ref.offset


def _fake_ca(*, rank=1, world_size=2, log=()):
    return SimpleNamespace(
        rank=rank,
        world_size=world_size,
        graph_params=_FakeTensor(0x7000_0000, _MAX_GRAPH_INPUTS * world_size * 8),
        _push_counter=_FakeTensor(0x7100_0000, 64),
        graph_row_log=tuple(log),
        graph_row_count=(max((r for r, _, _ in log), default=-1) + 1),
        pre_advance_graph_counter=mock.Mock(name="pre_advance_graph_counter"),
        register_graph_inputs_at=mock.Mock(name="register_graph_inputs_at"),
    )


def _fake_nccl_comm(*, rank=0, world_size=2, available=True, symmetric=True):
    comm = SimpleNamespace(available=available)
    comm.describe = lambda: NcclCommDescriptor(
        rank=rank,
        world_size=world_size,
        nccl_version=22807 if available else None,
        symmetric_memory=symmetric,
        available=available,
    )
    comm.register_comm_window_raw = mock.Mock(
        side_effect=lambda ptr, size: ("win", ptr, size)
    )
    return comm


# Two inputs live in a "static" region, one in a "pool" region.
_REGIONS = {"static:x": (0x1000, 0x1000), "pool:seg:0": (0x9000, 0x4000)}
_LOG = ((3, 0x1000, 64), (4, 0x1800, 128), (7, 0x9100, 256))


# --------------------------------------------------------------------------
# is_nccl_kernel
# --------------------------------------------------------------------------


def test_is_nccl_kernel_matches_prefixes_only():
    assert NCCL_KERNEL_PREFIXES == ("ncclDevKernel", "ncclKernel")
    assert is_nccl_kernel("ncclDevKernel_AllReduce_Sum_bf16_RING_LL")
    assert is_nccl_kernel("ncclKernel_AllGather_RING_LL_Sum_int8_t")
    assert not is_nccl_kernel("cross_device_reduce_1stage")
    assert not is_nccl_kernel("nvjet_tst_128x128")
    assert not is_nccl_kernel("")
    # substring matches are not prefix matches
    assert not is_nccl_kernel("wrapper_ncclDevKernel")


# --------------------------------------------------------------------------
# CustomAllReduceV2GraphState
# --------------------------------------------------------------------------


def test_ca_v2_identity_and_regions():
    ca = _fake_ca(rank=1, world_size=2)
    state = CustomAllReduceV2GraphState(ca, GROUP)

    assert state.identity() == CommIdentity(
        group=GROUP, impl="ca_v2", world_size=2, rank=1
    )

    regions = state.graph_visible_regions()
    assert [spec.region_id for spec, _ in regions] == [
        "comm:tp:0:ca_v2:table",
        "comm:tp:0:ca_v2:push_counter",
    ]
    table_spec, table_base = regions[0]
    assert isinstance(table_spec, RegionSpec)
    assert table_spec.kind == RegionKind.COMM_TABLE.value == "comm_table"
    assert table_spec.nbytes == ca.graph_params.nbytes
    assert table_base == ca.graph_params.data_ptr()
    counter_spec, counter_base = regions[1]
    assert counter_spec.nbytes == 64
    assert counter_base == ca._push_counter.data_ptr()

    with pytest.raises(NotImplementedError, match="peer_slab_regions.*section 6.10"):
        state.peer_slab_regions()


def test_ca_v2_export_rows_and_max_row():
    state = CustomAllReduceV2GraphState(_fake_ca(log=_LOG), GROUP)

    ckpt = state.export_graph_state(_FakeRegistry(_REGIONS))

    assert ckpt.identity.impl == "ca_v2"
    assert ckpt.in_graph_nccl_kernels is False
    assert ckpt.windows == ()
    blob = ckpt.graph_rows
    assert isinstance(blob, CommStateBlob)
    assert blob.impl == "ca_v2" and blob.group == GROUP
    assert blob.rows == (
        (3, RegionRef("static:x", 0), 64),
        (4, RegionRef("static:x", 0x800), 128),
        (7, RegionRef("pool:seg:0", 0x100), 256),
    )
    assert blob.max_row == 7


def test_ca_v2_export_empty_log_gives_max_row_minus_one():
    state = CustomAllReduceV2GraphState(_fake_ca(log=()), GROUP)
    blob = state.export_graph_state(_FakeRegistry({})).graph_rows
    assert blob.rows == ()
    assert blob.max_row == -1


def test_ca_v2_export_raises_when_a_row_is_unclassified():
    log = ((0, 0x1000, 64), (1, 0xDEAD0000, 64))
    state = CustomAllReduceV2GraphState(_fake_ca(log=log), GROUP)

    with pytest.raises(RuntimeError) as excinfo:
        state.export_graph_state(_FakeRegistry(_REGIONS))
    message = str(excinfo.value)
    assert "row 1" in message
    assert "0xdead0000" in message
    assert GROUP in message


def test_ca_v2_pre_advance_uses_max_row_plus_one():
    ca = _fake_ca(log=_LOG)
    state = CustomAllReduceV2GraphState(ca, GROUP)
    ckpt = state.export_graph_state(_FakeRegistry(_REGIONS))

    state.pre_advance(ckpt)

    ca.pre_advance_graph_counter.assert_called_once_with(8)


def test_ca_v2_pre_advance_without_blob_touches_nothing():
    ca = _fake_ca()
    state = CustomAllReduceV2GraphState(ca, GROUP)
    state.pre_advance(CommCheckpoint(identity=state.identity()))
    ca.pre_advance_graph_counter.assert_not_called()


def test_ca_v2_restore_passes_rebased_pointers():
    ca = _fake_ca(log=_LOG)
    state = CustomAllReduceV2GraphState(ca, GROUP)
    ckpt = state.export_graph_state(_FakeRegistry(_REGIONS))
    # in the loading process the two regions landed elsewhere
    reloc = _FakeReloc({"static:x": 0x5000_0000, "pool:seg:0": 0x6000_0000})

    state.restore_graph_state(ckpt, reloc)

    ca.register_graph_inputs_at.assert_called_once_with(
        [
            (3, 0x5000_0000, 64),
            (4, 0x5000_0800, 128),
            (7, 0x6000_0100, 256),
        ]
    )


def test_ca_v2_restore_without_rows_skips_the_exchange():
    ca = _fake_ca(log=())
    state = CustomAllReduceV2GraphState(ca, GROUP)
    ckpt = state.export_graph_state(_FakeRegistry({}))
    state.restore_graph_state(ckpt, _FakeReloc({}))
    ca.register_graph_inputs_at.assert_not_called()


def test_ca_v2_rejects_checkpoint_of_another_communicator():
    state = CustomAllReduceV2GraphState(_fake_ca(rank=1, world_size=2), GROUP)
    foreign = CommCheckpoint(
        identity=CommIdentity(group=GROUP, impl="ca_v2", world_size=2, rank=0)
    )
    with pytest.raises(RuntimeError, match="rank: saved 0 != live 1"):
        state.pre_advance(foreign)
    with pytest.raises(RuntimeError, match="does not match"):
        state.restore_graph_state(foreign, _FakeReloc({}))


# --------------------------------------------------------------------------
# CommCheckpoint round trip
# --------------------------------------------------------------------------


def test_comm_checkpoint_msgpack_round_trip():
    ckpt = CommCheckpoint(
        identity=CommIdentity(
            group=GROUP,
            impl="pynccl",
            world_size=4,
            rank=2,
            nccl_version=22807,
            symmetric_memory=True,
        ),
        graph_rows=CommStateBlob(
            impl="ca_v2",
            group=GROUP,
            rows=((3, RegionRef("static:x", 0), 64), (9, RegionRef("kv:k:0", 32), 8)),
            max_row=9,
        ),
        windows=(
            CommWindowRecord(region=RegionRef("pool:nccl:0", 4096), nbytes=1 << 20),
        ),
        in_graph_nccl_kernels=True,
    )

    packed = msgspec.msgpack.encode(ckpt)
    back = msgspec.msgpack.decode(packed, type=CommCheckpoint)

    assert back == ckpt
    assert back.graph_rows.rows[1][1] == RegionRef("kv:k:0", 32)
    assert back.needs_recapture is True
    # frozen records
    with pytest.raises(AttributeError):
        back.in_graph_nccl_kernels = False


# --------------------------------------------------------------------------
# PyNcclCommGraphState
# --------------------------------------------------------------------------


def test_pynccl_identity_comes_from_describe():
    comm = _fake_nccl_comm(rank=1, world_size=2, symmetric=True)
    state = PyNcclCommGraphState(comm, GROUP)
    assert state.identity() == CommIdentity(
        group=GROUP,
        impl="pynccl",
        world_size=2,
        rank=1,
        nccl_version=22807,
        symmetric_memory=True,
    )
    assert state.graph_visible_regions() == []


def test_pynccl_export_carries_windows_and_flag(caplog):
    comm = _fake_nccl_comm()
    state = PyNcclCommGraphState(comm, GROUP)
    state.record_window(RegionRef("pool:nccl:0", 0), 4096)
    state.record_window(RegionRef("pool:nccl:0", 8192), 1024)

    assert state.export_graph_state(registry=None).in_graph_nccl_kernels is False

    with caplog.at_level(logging.WARNING):
        state.mark_in_graph_nccl_kernel("ncclDevKernel_AllReduce_Sum_bf16_RING_LL")
    assert "needs_recapture" in caplog.text

    ckpt = state.export_graph_state(registry=None)
    assert ckpt.in_graph_nccl_kernels is True
    assert ckpt.needs_recapture is True
    assert ckpt.windows == (
        CommWindowRecord(region=RegionRef("pool:nccl:0", 0), nbytes=4096),
        CommWindowRecord(region=RegionRef("pool:nccl:0", 8192), nbytes=1024),
    )
    assert ckpt.graph_rows is None

    # constructor kwarg form
    assert (
        PyNcclCommGraphState(comm, GROUP, in_graph_nccl_kernels=True)
        .export_graph_state(registry=None)
        .in_graph_nccl_kernels
        is True
    )


def test_pynccl_restore_reregisters_windows_on_the_live_comm():
    comm = _fake_nccl_comm()
    state = PyNcclCommGraphState(comm, GROUP)
    state.record_window(RegionRef("pool:nccl:0", 0), 4096)
    state.record_window(RegionRef("pool:nccl:1", 256), 1024)
    ckpt = state.export_graph_state(registry=None)
    reloc = _FakeReloc({"pool:nccl:0": 0xA000_0000, "pool:nccl:1": 0xB000_0000})

    fresh = PyNcclCommGraphState(comm, GROUP)
    fresh.pre_advance(ckpt)  # no-op for pynccl
    fresh.restore_graph_state(ckpt, reloc)

    assert comm.register_comm_window_raw.call_args_list == [
        mock.call(0xA000_0000, 4096),
        mock.call(0xB000_0100, 1024),
    ]
    assert fresh.restored_windows == (
        ("win", 0xA000_0000, 4096),
        ("win", 0xB000_0100, 1024),
    )


def test_pynccl_restore_refuses_windows_without_a_comm():
    live = _fake_nccl_comm(available=False)
    saved = _fake_nccl_comm(available=False)
    saver = PyNcclCommGraphState(saved, GROUP)
    saver.record_window(RegionRef("pool:nccl:0", 0), 4096)
    ckpt = saver.export_graph_state(registry=None)

    with pytest.raises(RuntimeError, match="no ncclComm_t"):
        PyNcclCommGraphState(live, GROUP).restore_graph_state(ckpt, _FakeReloc({}))
    live.register_comm_window_raw.assert_not_called()


# --------------------------------------------------------------------------
# CustomAllReduceV1GraphState (stubs)
# --------------------------------------------------------------------------


def test_ca_v1_regions_and_stubs():
    ca = SimpleNamespace(rank=0, world_size=2, rank_data=_FakeTensor(0x4000, 8 << 20))
    state = CustomAllReduceV1GraphState(ca, GROUP)

    assert state.identity() == CommIdentity(
        group=GROUP, impl="ca_v1", world_size=2, rank=0
    )
    [(spec, base)] = state.graph_visible_regions()
    assert spec.region_id == "comm:tp:0:ca_v1:rank_data"
    assert spec.kind == "comm_table"
    assert spec.nbytes == 8 << 20
    assert base == 0x4000

    ckpt = CommCheckpoint(identity=state.identity())
    with pytest.raises(NotImplementedError, match="export_graph_state.*section 6.10"):
        state.export_graph_state(_FakeRegistry({}))
    with pytest.raises(NotImplementedError, match="pre_advance.*section 6.10"):
        state.pre_advance(ckpt)
    with pytest.raises(NotImplementedError, match="register_graph_buffers"):
        state.restore_graph_state(ckpt, _FakeReloc({}))


# --------------------------------------------------------------------------
# UnsupportedCommGraphState / collect_comm_graph_states
# --------------------------------------------------------------------------


def test_unsupported_state_exports_a_recapture_marker():
    comm = SimpleNamespace(world_size=4)  # no ``rank`` attribute, like symm-mem
    state = UnsupportedCommGraphState(comm, GROUP, "torch_symm_mem")
    ckpt = state.export_graph_state(registry=None)
    assert ckpt.in_graph_nccl_kernels is True
    assert ckpt.identity == CommIdentity(
        group=GROUP, impl="torch_symm_mem", world_size=4, rank=-1
    )
    assert state.graph_visible_regions() == []
    # both load hooks are implemented no-ops (the group recaptures)
    state.pre_advance(ckpt)
    state.restore_graph_state(ckpt, _FakeReloc({}))


def _fake_ca_named_like_the_real_class():
    """An unrelated class whose NAME is ``CustomAllReduceV2`` (class-name fallback)."""
    cls = type(
        "CustomAllReduceV2",
        (),
        {
            "rank": 0,
            "world_size": 2,
            "graph_params": _FakeTensor(0x7000_0000, 16),
            "_push_counter": _FakeTensor(0x7100_0000, 16),
            "graph_row_log": (),
            "graph_row_count": 0,
        },
    )
    assert cls is not CustomAllReduceV2
    return cls()


def test_collect_comm_graph_states_by_class_name_fallback(caplog):
    coordinator = SimpleNamespace(
        unique_name=GROUP,
        ca_comm=_fake_ca_named_like_the_real_class(),
        pynccl_comm=_fake_nccl_comm(),
        torch_symm_mem_comm=SimpleNamespace(world_size=2),
        pymscclpp_comm=None,
        qr_comm=None,
    )

    with caplog.at_level(logging.WARNING):
        states = collect_comm_graph_states(coordinator)

    assert [type(s).__name__ for s in states] == [
        "CustomAllReduceV2GraphState",
        "PyNcclCommGraphState",
        "UnsupportedCommGraphState",
    ]
    assert states[0].identity().impl == "ca_v2"
    assert states[0].group == GROUP
    assert states[2].impl == "torch_symm_mem"
    assert "torch_symm_mem_comm" in caplog.text
    assert "needs_recapture" in caplog.text


def test_collect_comm_graph_states_by_isinstance_and_skips_uninitialised():
    real = CustomAllReduceV2.__new__(CustomAllReduceV2)
    real.disabled = False
    real.rank = 1
    real.world_size = 2
    coordinator = SimpleNamespace(unique_name="tp:1", ca_comm=real, pynccl_comm=None)
    [state] = collect_comm_graph_states(coordinator)
    assert isinstance(state, CustomAllReduceV2GraphState)
    assert state.identity() == CommIdentity(
        group="tp:1", impl="ca_v2", world_size=2, rank=1
    )

    # a v2 instance whose constructor bailed out has no rank / world_size
    bailed = CustomAllReduceV2.__new__(CustomAllReduceV2)
    bailed.disabled = True
    assert (
        collect_comm_graph_states(
            SimpleNamespace(unique_name="tp:2", ca_comm=bailed, pynccl_comm=None)
        )
        == []
    )

    # no communicators at all
    assert collect_comm_graph_states(SimpleNamespace(unique_name="tp:3")) == []


# --------------------------------------------------------------------------
# CommCheckpointer
# --------------------------------------------------------------------------


def test_checkpointer_export_pre_advance_restore_in_lockstep():
    ca = _fake_ca(log=_LOG)
    comm = _fake_nccl_comm()
    ca_state = CustomAllReduceV2GraphState(ca, GROUP)
    nccl_state = PyNcclCommGraphState(comm, GROUP)
    nccl_state.record_window(RegionRef("pool:nccl:0", 0), 4096)
    checkpointer = CommCheckpointer([ca_state, nccl_state])

    assert [spec.region_id for spec, _ in checkpointer.graph_visible_regions()] == [
        "comm:tp:0:ca_v2:table",
        "comm:tp:0:ca_v2:push_counter",
    ]

    ckpts = checkpointer.export_all(_FakeRegistry(_REGIONS))
    assert len(ckpts) == 2
    assert CommCheckpointer.any_needs_recapture(ckpts) is False

    # msgpack round trip of the whole tuple, as the bundle would store it
    ckpts = msgspec.msgpack.decode(
        msgspec.msgpack.encode(ckpts), type=tuple[CommCheckpoint, ...]
    )

    checkpointer.pre_advance_all(ckpts)
    ca.pre_advance_graph_counter.assert_called_once_with(8)
    comm.register_comm_window_raw.assert_not_called()

    reloc = _FakeReloc(
        {"static:x": 0x10, "pool:seg:0": 0x20, "pool:nccl:0": 0xA000_0000}
    )
    checkpointer.restore_all(ckpts, reloc)
    ca.register_graph_inputs_at.assert_called_once_with(
        [(3, 0x10, 64), (4, 0x810, 128), (7, 0x120, 256)]
    )
    comm.register_comm_window_raw.assert_called_once_with(0xA000_0000, 4096)

    with pytest.raises(RuntimeError, match="1 checkpoints for 2"):
        checkpointer.pre_advance_all(ckpts[:1])
    with pytest.raises(RuntimeError, match="1 checkpoints for 2"):
        checkpointer.restore_all(ckpts[:1], reloc)


def test_checkpointer_any_needs_recapture_sees_unsupported_marker():
    unsupported = UnsupportedCommGraphState(SimpleNamespace(), GROUP, "mscclpp")
    ckpts = CommCheckpointer([unsupported]).export_all(registry=None)
    assert CommCheckpointer.any_needs_recapture(ckpts) is True


# --------------------------------------------------------------------------
# Real CustomAllReduceV2 entry points (no CUDA: __new__ + fake torch namespace)
# --------------------------------------------------------------------------


class _RecordingTable:
    """Stands in for the ``graph_params`` tensor: records slice writes."""

    def __init__(self):
        self.writes = []

    def __getitem__(self, key):
        table = self

        class _Slice:
            def copy_(self, rows):
                table.writes.append((key, rows))

        return _Slice()


def _build_real_ca(monkeypatch, *, counter, pending):
    sync = mock.Mock(name="synchronize")
    fake_torch = SimpleNamespace(
        uint64="uint64",
        tensor=lambda data, dtype=None, device=None: ("rows", data, dtype, device),
        cuda=SimpleNamespace(synchronize=sync),
    )
    monkeypatch.setattr(custom_all_reduce_v2, "torch", fake_torch)

    ca = CustomAllReduceV2.__new__(CustomAllReduceV2)
    ca.disabled = False  # keeps ``__del__ -> close()`` quiet; no ``obj`` exists
    ca.device = "fake-device"
    ca.rank = 0
    ca.world_size = 2
    ca.graph_params = _RecordingTable()
    ca._graph_inputs = list(pending)
    ca._graph_counter = counter
    ca._graph_row_log = []
    return ca, sync


def test_real_register_peer_mapped_inputs_appends_absolute_rows(monkeypatch):
    ca, sync = _build_real_ca(
        monkeypatch, counter=5, pending=[(0x1000, 64), (0x2000, 128)]
    )

    ca.register_peer_mapped_inputs([[0x1000, 0xA000], [0x2000, 0xB000]])

    # existing behaviour: rows written at the contiguous slice, synced, counter bumped
    assert ca.graph_params.writes == [
        (
            slice(5, 7),
            ("rows", [[0x1000, 0xA000], [0x2000, 0xB000]], "uint64", "fake-device"),
        )
    ]
    sync.assert_called_once_with()
    assert ca._graph_inputs == []
    assert ca.graph_row_count == 7
    # new: the log records (absolute row, local ptr, nbytes) for every row written
    assert ca.graph_row_log == ((5, 0x1000, 64), (6, 0x2000, 128))
    assert isinstance(ca.graph_row_log, tuple)

    # a second registration keeps appending at the advanced counter
    ca._graph_inputs = [(0x3000, 16)]
    ca.register_peer_mapped_inputs([[0x3000, 0xC000]])
    assert ca.graph_row_log == ((5, 0x1000, 64), (6, 0x2000, 128), (7, 0x3000, 16))
    assert ca.graph_row_count == 8


def test_real_pre_advance_graph_counter(monkeypatch):
    ca, _ = _build_real_ca(monkeypatch, counter=3, pending=[])

    ca.pre_advance_graph_counter(3)  # no move is allowed
    assert ca.graph_row_count == 3
    ca.pre_advance_graph_counter(10)
    assert ca.graph_row_count == 10
    ca.pre_advance_graph_counter(_MAX_GRAPH_INPUTS)  # table exhausted, still legal
    assert ca.graph_row_count == _MAX_GRAPH_INPUTS

    with pytest.raises(AssertionError, match="backwards"):
        ca.pre_advance_graph_counter(9)
    with pytest.raises(AssertionError, match="capacity"):
        ca.pre_advance_graph_counter(_MAX_GRAPH_INPUTS + 1)
    assert ca.graph_row_count == _MAX_GRAPH_INPUTS

    ca._graph_counter = 0
    ca._graph_inputs = [(0x1000, 64)]
    with pytest.raises(AssertionError, match="pending"):
        ca.pre_advance_graph_counter(4)
    assert ca.graph_row_count == 0

    # the log is untouched by pre-advance
    assert ca.graph_row_log == ()


def test_real_register_graph_inputs_at_is_a_documented_stub(monkeypatch):
    ca, sync = _build_real_ca(monkeypatch, counter=8, pending=[])
    with pytest.raises(NotImplementedError, match="register_graph_inputs_at.*6.10"):
        ca.register_graph_inputs_at([(3, 0x1000, 64)])
    # the stub touches nothing
    assert ca.graph_row_count == 8
    assert ca._graph_inputs == []
    sync.assert_not_called()
    doc = ca.register_graph_inputs_at.__doc__
    assert "absolute" in doc.lower()
    assert "_graph_counter" in doc
    assert "synchronize" in doc


def test_real_ca_v2_adapter_over_real_object(monkeypatch):
    ca, _ = _build_real_ca(monkeypatch, counter=5, pending=[(0x1000, 64)])
    ca.register_peer_mapped_inputs([[0x1000, 0xA000]])
    ca._push_counter = _FakeTensor(0x7100_0000, 64)
    ca.graph_params = _FakeTensor(0x7000_0000, 1024)

    state = CustomAllReduceV2GraphState(ca, GROUP)
    ckpt = state.export_graph_state(_FakeRegistry(_REGIONS))
    assert ckpt.graph_rows.rows == ((5, RegionRef("static:x", 0), 64),)
    assert ckpt.graph_rows.max_row == 5

    state.pre_advance(ckpt)  # real entry point: 5 -> 6
    assert ca.graph_row_count == 6
    with pytest.raises(NotImplementedError):
        state.restore_graph_state(ckpt, _FakeReloc({"static:x": 0x10}))


# --------------------------------------------------------------------------
# PyNcclCommunicator.describe()
# --------------------------------------------------------------------------


def _stateless_group(rank, world_size):
    group = StatelessProcessGroup.__new__(StatelessProcessGroup)
    object.__setattr__(group, "rank", rank)
    object.__setattr__(group, "world_size", world_size)
    return group


def test_describe_on_world_size_one_early_return():
    comm = PyNcclCommunicator(
        group=_stateless_group(0, 1), device="cpu", is_symmetric_memory_enabled=True
    )
    assert comm.is_symmetric_memory_enabled is True
    assert comm.describe() == NcclCommDescriptor(
        rank=0,
        world_size=1,
        nccl_version=None,
        symmetric_memory=True,
        available=False,
    )


def test_describe_when_the_nccl_library_is_missing(monkeypatch):
    def _missing(path):
        raise OSError("libnccl.so not found")

    monkeypatch.setattr(pynccl_mod, "NCCLLibrary", _missing)
    comm = PyNcclCommunicator(group=_stateless_group(1, 2), device="cpu")
    desc = comm.describe()
    assert desc == NcclCommDescriptor(
        rank=1, world_size=2, nccl_version=None, symmetric_memory=False, available=False
    )
    with pytest.raises(Exception):  # frozen dataclass
        desc.rank = 5


def test_describe_on_a_live_communicator_shape():
    comm = PyNcclCommunicator.__new__(PyNcclCommunicator)
    comm.rank = 3
    comm.world_size = 8
    comm.available = True
    comm.disabled = True  # the normal post-init toggle state; irrelevant here
    comm.is_symmetric_memory_enabled = False
    comm.nccl_version = 22807
    assert comm.describe() == NcclCommDescriptor(
        rank=3, world_size=8, nccl_version=22807, symmetric_memory=False, available=True
    )
    # the adapter reads the same record
    state = PyNcclCommGraphState(comm, "tp:0")
    assert state.identity().nccl_version == 22807
    assert state.identity().symmetric_memory is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
