# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Communicator checkpoint for serialized CUDA graphs (design section 6.10).

A replayed graph holds three kinds of communicator state that a fresh process
cannot reproduce by chance:

* **Custom all-reduce v2 pointer-table rows.** Every graph-captured all-reduce
  dereferences one row of ``CustomAllReduceV2.graph_params`` at replay; the
  row pointer is a plain word in the kernel parameters and the row index is
  absolute (fact 17). Row *contents* are peer virtual addresses of the saving
  process and are never stored: only ``(absolute row, input RegionRef,
  nbytes)`` is. The loader pre-advances the row counter past the saved rows
  before the shape loop, so shapes that fall back to capture allocate above
  them, and re-runs the IPC / VMM exchange for exactly the saved rows after
  the shape loop.
* **NCCL symmetric-memory windows.** ``ncclCommWindowRegister`` binds a
  buffer to one ``ncclComm_t``; a fresh communicator needs every window
  re-registered before the first replay (design section 9.4).
* **In-graph NCCL kernels.** The ``ncclComm_t`` inside every NCCL launch is
  not relocatable, so a graph that launches one is ``needs_recapture``
  (design sections 3 and 9.4). :func:`is_nccl_kernel` is the detector the
  codec applies to kernel names; :attr:`CommCheckpoint.in_graph_nccl_kernels`
  carries the verdict for the whole group.

Collective ordering contract (:class:`CommCheckpointer`): ``pre_advance_all``
runs on every rank BEFORE the shape loop; ``restore_all`` runs AFTER the shape
loop, inside ``parallel_state.graph_capture()``, on every rank at the same
program point, because the row exchange is an ``all_gather_object`` over the
group. Row contents are never copied from the artifact.

Nothing in this module calls the CUDA driver; the wrapped communicators do.
``RegionRegistry`` and ``RelocationMap`` (``regions.py``) are used by duck
typing only: ``registry.classify(word)`` and ``reloc.rebase(ref)``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Sequence

import msgspec

from sglang.srt.model_executor.graph_serialization.format import (
    CommStateBlob,
    RegionKind,
    RegionRef,
    RegionSpec,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.graph_serialization.regions import (
        RegionRegistry,
        RelocationMap,
    )

logger = logging.getLogger(__name__)

_DESIGN = "DESIGN_cuda_graph_serialization.md"

# Kernel-name prefixes of NCCL device kernels (design section 9.4). A graph
# node whose kernel name starts with one of these carries an ``ncclComm_t`` in
# its parameters and is not relocatable.
NCCL_KERNEL_PREFIXES = ("ncclDevKernel", "ncclKernel")

IMPL_CA_V2 = "ca_v2"
IMPL_CA_V1 = "ca_v1"
IMPL_PYNCCL = "pynccl"


def is_nccl_kernel(name: str) -> bool:
    """True when ``name`` is an NCCL device kernel (design section 9.4)."""
    return name.startswith(NCCL_KERNEL_PREFIXES)


# ----------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------


class CommIdentity(msgspec.Struct, frozen=True, kw_only=True):
    """Which communicator a checkpoint belongs to.

    ``group`` is ``GroupCoordinator.unique_name`` (``"tp:0"``), which is
    deterministic across processes with the same init order. ``impl`` is one
    of ``ca_v2``, ``ca_v1``, ``pynccl`` or the unsupported communicator's
    class tag.
    """

    group: str
    impl: str
    world_size: int
    rank: int
    nccl_version: Optional[int] = None
    symmetric_memory: bool = False


class CommWindowRecord(msgspec.Struct, frozen=True, kw_only=True):
    """One NCCL symmetric-memory window to re-register at load."""

    region: RegionRef
    nbytes: int


class CommCheckpoint(msgspec.Struct, frozen=True, kw_only=True):
    """Everything a loaded graph needs from one communicator (section 6.10).

    ``graph_rows`` are the custom all-reduce rows at absolute indices;
    ``windows`` are NCCL symmetric-memory windows; ``in_graph_nccl_kernels``
    set means every graph of the group is ``needs_recapture``.
    """

    identity: CommIdentity
    graph_rows: Optional[CommStateBlob] = None
    windows: tuple[CommWindowRecord, ...] = ()
    in_graph_nccl_kernels: bool = False

    @property
    def needs_recapture(self) -> bool:
        return self.in_graph_nccl_kernels


# ----------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------


class CommGraphState(Protocol):
    """Per-communicator adapter the checkpointer drives (design section 6.10).

    ``graph_visible_regions`` feeds the region registry at save (and the
    ``CommTableProvider`` of ``regions.py``); ``export_graph_state`` runs
    after capture; ``pre_advance`` runs at LOAD before the shape loop;
    ``restore_graph_state`` runs at LOAD after the shape loop and is
    collective over the group.
    """

    def identity(self) -> CommIdentity: ...

    def graph_visible_regions(self) -> list[tuple[RegionSpec, int]]: ...

    def export_graph_state(self, registry: RegionRegistry) -> CommCheckpoint: ...

    def pre_advance(self, ckpt: CommCheckpoint) -> None: ...

    def restore_graph_state(
        self, ckpt: CommCheckpoint, reloc: RelocationMap
    ) -> None: ...


def _require_matching_identity(expected: CommIdentity, ckpt: CommCheckpoint) -> None:
    """Fail closed when a checkpoint is handed to the wrong communicator.

    Only the fields that name the communicator are compared; ``nccl_version``
    and ``symmetric_memory`` are fingerprint concerns (design section 11).
    """
    got = ckpt.identity
    mismatches = [
        f"{field}: saved {getattr(got, field)!r} != live {getattr(expected, field)!r}"
        for field in ("group", "impl", "world_size", "rank")
        if getattr(got, field) != getattr(expected, field)
    ]
    if mismatches:
        raise RuntimeError(
            "communicator checkpoint does not match the live communicator: "
            + "; ".join(mismatches)
        )


def _tensor_region(region_id: str, tensor: Any) -> tuple[RegionSpec, int]:
    base = int(tensor.data_ptr())
    spec = RegionSpec(
        region_id=region_id,
        kind=RegionKind.COMM_TABLE.value,
        nbytes=int(tensor.nbytes),
        base_at_save=base,
    )
    return spec, base


# ----------------------------------------------------------------------------
# Custom all-reduce v2
# ----------------------------------------------------------------------------


class CustomAllReduceV2GraphState:
    """Checkpoint adapter for ``CustomAllReduceV2`` (design sections 6.10, 9.4).

    Uses ``ca.graph_row_log`` / ``ca.graph_row_count`` at save and
    ``ca.pre_advance_graph_counter`` / ``ca.register_graph_inputs_at`` at load.
    All of its regions RELOCATE: every pointer the kernels take is a plain
    8-byte-aligned word (fact 17).
    """

    impl = IMPL_CA_V2

    def __init__(self, ca: Any, group_name: str) -> None:
        self._ca = ca
        self._group = str(group_name)

    @property
    def group(self) -> str:
        return self._group

    def identity(self) -> CommIdentity:
        return CommIdentity(
            group=self._group,
            impl=self.impl,
            world_size=int(self._ca.world_size),
            rank=int(self._ca.rank),
        )

    def _region_id(self, leaf: str) -> str:
        return f"comm:{self._group}:{self.impl}:{leaf}"

    def graph_visible_regions(self) -> list[tuple[RegionSpec, int]]:
        """The ``graph_params`` table and the rank-local ``_push_counter``.

        The per-peer slabs and the multicast range are graph-visible too (a
        graph holding only ``1shot_push`` all-reduces bakes no row but still
        needs them); they come from :meth:`peer_slab_regions`, a stub in this
        draft, so a registry built from this method alone classifies slab
        words as unknown and the codec marks the shape ``needs_recapture``.
        """
        return [
            _tensor_region(self._region_id("table"), self._ca.graph_params),
            _tensor_region(self._region_id("push_counter"), self._ca._push_counter),
        ]

    def peer_slab_regions(self) -> list[tuple[RegionSpec, int]]:
        """Per-peer slab and multicast regions (design section 6.10, fact 17).

        Contract: one region per peer slab ``symm_mem.get_buffer(i)`` covering
        the whole ``[2 * world_size push slots | pull buffer | pull
        semaphores]`` layout (kernels take interior pointers: push slots at
        ``+0``, pull workspace and semaphores at fixed offsets), plus the
        multicast range ``[multicast_ptr, multicast_ptr + slab_bytes)`` when
        ``ca.has_multicast`` (workspace AND semaphores). Slab and multicast
        virtual addresses are driver-chosen imports, so every one of them is
        RELOCATE (design section 3).
        """
        raise NotImplementedError(
            "CustomAllReduceV2GraphState.peer_slab_regions: enumerating the "
            "per-peer symmetric-memory slabs and the multicast range is not "
            f"implemented in this draft; see {_DESIGN} section 6.10"
        )

    def export_graph_state(self, registry: RegionRegistry) -> CommCheckpoint:
        """Every graph input as ``(absolute row, RegionRef, nbytes)``.

        A row whose input pointer lies in no registered region cannot be
        rebased at load, so export fails loudly rather than saving a row the
        loader could only fill with a stale address.
        """
        rows: List[tuple[int, RegionRef, int]] = []
        for row, local_ptr, nbytes in self._ca.graph_row_log:
            ref = registry.classify(int(local_ptr))
            if ref is None:
                raise RuntimeError(
                    f"CustomAllReduceV2GraphState.export_graph_state: graph_params "
                    f"row {row} of group {self._group!r} points at 0x{local_ptr:x} "
                    f"({nbytes} bytes), which lies in no registered region"
                )
            rows.append((int(row), ref, int(nbytes)))
        max_row = max((row for row, _, _ in rows), default=-1)
        blob = CommStateBlob(
            impl=self.impl, group=self._group, rows=tuple(rows), max_row=max_row
        )
        return CommCheckpoint(identity=self.identity(), graph_rows=blob)

    def pre_advance(self, ckpt: CommCheckpoint) -> None:
        """LOAD, before the shape loop: counter = ``max_row + 1`` (fact 17)."""
        _require_matching_identity(self.identity(), ckpt)
        blob = ckpt.graph_rows
        if blob is None:
            logger.debug(
                "group %s: checkpoint carries no graph rows; counter stays at %d",
                self._group,
                self._ca.graph_row_count,
            )
            return
        self._ca.pre_advance_graph_counter(blob.max_row + 1)

    def restore_graph_state(self, ckpt: CommCheckpoint, reloc: RelocationMap) -> None:
        """LOAD, after the shape loop, collective: refill the saved rows.

        Rebases each saved input ``RegionRef`` into this process and hands
        ``(absolute row, local pointer, nbytes)`` to
        ``ca.register_graph_inputs_at``, which runs the peer exchange and
        writes the rows at their absolute indices (design section 6.10).
        """
        _require_matching_identity(self.identity(), ckpt)
        blob = ckpt.graph_rows
        if blob is None or not blob.rows:
            logger.debug("group %s: no graph_params rows to restore", self._group)
            return
        entries = [
            (int(row), int(reloc.rebase(ref)), int(nbytes))
            for row, ref, nbytes in blob.rows
        ]
        self._ca.register_graph_inputs_at(entries)


# ----------------------------------------------------------------------------
# Custom all-reduce v1
# ----------------------------------------------------------------------------


class CustomAllReduceV1GraphState:
    """Checkpoint adapter for ``custom_all_reduce.CustomAllreduce`` (v1).

    v1 keeps its graph-buffer table in the C++ ``rank_data`` buffer, indexed
    by the ordinal of the all-reduce within the capture session (design
    section 3); the Python side only sees ``register_graph_buffers()``.
    Export and restore are stubs in this draft (design section 9.4: a probe
    like fact 17 is required before v1 rows can be restored).
    """

    impl = IMPL_CA_V1

    def __init__(self, ca: Any, group_name: str) -> None:
        self._ca = ca
        self._group = str(group_name)

    @property
    def group(self) -> str:
        return self._group

    def identity(self) -> CommIdentity:
        return CommIdentity(
            group=self._group,
            impl=self.impl,
            world_size=int(self._ca.world_size),
            rank=int(self._ca.rank),
        )

    def graph_visible_regions(self) -> list[tuple[RegionSpec, int]]:
        """``rank_data``: the pointer-tuple table the v1 kernels read."""
        region_id = f"comm:{self._group}:{self.impl}:rank_data"
        return [_tensor_region(region_id, self._ca.rank_data)]

    def export_graph_state(self, registry: RegionRegistry) -> CommCheckpoint:
        """Contract: the ordered list of graph-buffer inputs
        ``ops.get_graph_buffer_ipc_meta`` would export, as RegionRefs."""
        raise NotImplementedError(
            "CustomAllReduceV1GraphState.export_graph_state: reading the v1 "
            "graph-buffer list back from the C++ side is not implemented in this "
            f"draft; see {_DESIGN} section 6.10"
        )

    def pre_advance(self, ckpt: CommCheckpoint) -> None:
        """v1 slot ordinals live in C++; how recaptured shapes allocate above
        restored slots needs the fact-17 style probe (design section 9.4)."""
        raise NotImplementedError(
            "CustomAllReduceV1GraphState.pre_advance: advancing the v1 slot "
            f"ordinal is not implemented in this draft; see {_DESIGN} section 6.10"
        )

    def restore_graph_state(self, ckpt: CommCheckpoint, reloc: RelocationMap) -> None:
        """Contract: re-run ``register_graph_buffers`` over the saved ordered
        list with rebased pointers (design section 6.10)."""
        raise NotImplementedError(
            "CustomAllReduceV1GraphState.restore_graph_state: re-running "
            "register_graph_buffers over the saved ordered list is not implemented "
            f"in this draft; see {_DESIGN} section 6.10"
        )


# ----------------------------------------------------------------------------
# pynccl
# ----------------------------------------------------------------------------


class PyNcclCommGraphState:
    """Checkpoint adapter for ``PyNcclCommunicator`` (design sections 6.10, 9.4).

    Two responsibilities: carry the NCCL symmetric-memory windows the graphs
    depend on so they are re-registered against the new ``ncclComm_t`` at
    load, and carry the ``in_graph_nccl_kernels`` verdict the codec raises
    when it meets an NCCL kernel node (the ``ncclComm_t`` in its parameters
    is not relocatable, design section 3).

    ``graph_visible_regions`` is empty: symmetric-memory segments are owned by
    the NCCL allocator pool and enumerated by its provider; the windows are
    recorded here by :meth:`record_window`.
    """

    impl = IMPL_PYNCCL

    def __init__(
        self, comm: Any, group_name: str, *, in_graph_nccl_kernels: bool = False
    ) -> None:
        self._comm = comm
        self._group = str(group_name)
        self.in_graph_nccl_kernels = bool(in_graph_nccl_kernels)
        self._windows: List[CommWindowRecord] = []
        self._restored_windows: List[Any] = []

    @property
    def group(self) -> str:
        return self._group

    @property
    def windows(self) -> tuple[CommWindowRecord, ...]:
        return tuple(self._windows)

    @property
    def restored_windows(self) -> tuple[Any, ...]:
        """Window handles returned by ``register_comm_window_raw`` at load."""
        return tuple(self._restored_windows)

    def identity(self) -> CommIdentity:
        desc = self._comm.describe()
        return CommIdentity(
            group=self._group,
            impl=self.impl,
            world_size=int(desc.world_size),
            rank=int(desc.rank),
            nccl_version=desc.nccl_version,
            symmetric_memory=bool(desc.symmetric_memory),
        )

    def mark_in_graph_nccl_kernel(self, kernel_name: str) -> None:
        """Codec hook: an NCCL kernel was captured; the group recaptures."""
        if not self.in_graph_nccl_kernels:
            logger.warning(
                "group %s: kernel %s launches NCCL inside a captured graph; every "
                "graph of this group is needs_recapture (%s section 9.4)",
                self._group,
                kernel_name,
                _DESIGN,
            )
        self.in_graph_nccl_kernels = True

    def record_window(self, region: RegionRef, nbytes: int) -> None:
        """Remember a window registered on this comm during capture."""
        self._windows.append(CommWindowRecord(region=region, nbytes=int(nbytes)))

    def graph_visible_regions(self) -> list[tuple[RegionSpec, int]]:
        return []

    def export_graph_state(self, registry: RegionRegistry) -> CommCheckpoint:
        return CommCheckpoint(
            identity=self.identity(),
            windows=tuple(self._windows),
            in_graph_nccl_kernels=self.in_graph_nccl_kernels,
        )

    def pre_advance(self, ckpt: CommCheckpoint) -> None:
        """No row counter to move; only the identity is checked."""
        _require_matching_identity(self.identity(), ckpt)

    def restore_graph_state(self, ckpt: CommCheckpoint, reloc: RelocationMap) -> None:
        """Re-register every saved window against the live ``ncclComm_t``."""
        _require_matching_identity(self.identity(), ckpt)
        if not ckpt.windows:
            logger.debug("group %s: no NCCL windows to restore", self._group)
            return
        if not getattr(self._comm, "available", False):
            raise RuntimeError(
                f"PyNcclCommGraphState.restore_graph_state: group {self._group!r} "
                f"saved {len(ckpt.windows)} NCCL windows but the live communicator "
                "has no ncclComm_t"
            )
        handles: List[Any] = []
        for record in ckpt.windows:
            ptr = int(reloc.rebase(record.region))
            handles.append(self._comm.register_comm_window_raw(ptr, record.nbytes))
        self._restored_windows.extend(handles)
        logger.debug(
            "group %s: re-registered %d NCCL windows", self._group, len(handles)
        )


# ----------------------------------------------------------------------------
# Unsupported communicators
# ----------------------------------------------------------------------------


class UnsupportedCommGraphState:
    """Marker for communicators v1 does not checkpoint (design section 9.4).

    ``TorchSymmMemCommunicator``, ``PyMscclppCommunicator`` and
    ``QuickAllReduce`` bake pointers we do not enumerate. Rather than save a
    graph that would replay against stale addresses, the export marks the
    whole group ``needs_recapture`` (``in_graph_nccl_kernels=True``
    semantics). Pre-advance and restore have nothing to do: every shape of
    the group is recaptured, and the communicator is rebuilt by init.
    """

    def __init__(self, comm: Any, group_name: str, impl: str) -> None:
        self._comm = comm
        self._group = str(group_name)
        self.impl = str(impl)

    @property
    def group(self) -> str:
        return self._group

    def identity(self) -> CommIdentity:
        return CommIdentity(
            group=self._group,
            impl=self.impl,
            world_size=int(getattr(self._comm, "world_size", -1)),
            rank=int(getattr(self._comm, "rank", -1)),
        )

    def graph_visible_regions(self) -> list[tuple[RegionSpec, int]]:
        return []

    def export_graph_state(self, registry: RegionRegistry) -> CommCheckpoint:
        return CommCheckpoint(identity=self.identity(), in_graph_nccl_kernels=True)

    def pre_advance(self, ckpt: CommCheckpoint) -> None:
        _require_matching_identity(self.identity(), ckpt)
        logger.debug(
            "group %s: %s is unsupported; shapes recapture, nothing to pre-advance",
            self._group,
            self.impl,
        )

    def restore_graph_state(self, ckpt: CommCheckpoint, reloc: RelocationMap) -> None:
        _require_matching_identity(self.identity(), ckpt)
        logger.debug(
            "group %s: %s is unsupported; shapes recapture, nothing to restore",
            self._group,
            self.impl,
        )


# ----------------------------------------------------------------------------
# Collection and orchestration
# ----------------------------------------------------------------------------

_UNSUPPORTED_COMM_FIELDS = (
    ("torch_symm_mem_comm", "torch_symm_mem"),
    ("pymscclpp_comm", "mscclpp"),
    ("qr_comm", "quick_all_reduce"),
)


def _ca_impl(ca: Any) -> Optional[str]:
    """``ca_v2`` / ``ca_v1`` by isinstance, falling back to the class name."""
    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )
    except Exception:  # pragma: no cover - import surface differs per platform
        CustomAllReduceV2 = None  # type: ignore[assignment]
    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
    except Exception:  # pragma: no cover
        CustomAllreduce = None  # type: ignore[assignment]

    if CustomAllReduceV2 is not None and isinstance(ca, CustomAllReduceV2):
        return IMPL_CA_V2
    if CustomAllreduce is not None and isinstance(ca, CustomAllreduce):
        return IMPL_CA_V1
    name = type(ca).__name__
    if name == "CustomAllReduceV2":
        return IMPL_CA_V2
    if name == "CustomAllreduce":
        return IMPL_CA_V1
    return None


def _is_initialized_ca(ca: Any) -> bool:
    """Both custom all-reduce constructors return early (``disabled=True``)
    before ``rank`` / ``world_size`` exist when the group cannot use them."""
    return hasattr(ca, "rank") and hasattr(ca, "world_size")


def collect_comm_graph_states(group_coordinator: Any) -> list[CommGraphState]:
    """Adapters for every communicator a ``GroupCoordinator`` owns.

    ``ca_comm`` maps to the v2 or v1 adapter; ``pynccl_comm`` to
    :class:`PyNcclCommGraphState`; ``torch_symm_mem_comm``, ``pymscclpp_comm``
    and ``qr_comm`` to :class:`UnsupportedCommGraphState` with a warning.
    The group name is ``group_coordinator.unique_name``.
    """
    group = str(group_coordinator.unique_name)
    states: list[CommGraphState] = []

    ca = getattr(group_coordinator, "ca_comm", None)
    if ca is not None:
        if not _is_initialized_ca(ca):
            logger.debug(
                "group %s: custom all-reduce %s never initialised; skipped",
                group,
                type(ca).__name__,
            )
        else:
            impl = _ca_impl(ca)
            if impl == IMPL_CA_V2:
                states.append(CustomAllReduceV2GraphState(ca, group))
            elif impl == IMPL_CA_V1:
                states.append(CustomAllReduceV1GraphState(ca, group))
            else:
                logger.warning(
                    "group %s: unknown custom all-reduce class %s; graphs of this "
                    "group are needs_recapture",
                    group,
                    type(ca).__name__,
                )
                states.append(UnsupportedCommGraphState(ca, group, type(ca).__name__))

    pynccl = getattr(group_coordinator, "pynccl_comm", None)
    if pynccl is not None:
        states.append(PyNcclCommGraphState(pynccl, group))

    for field, impl in _UNSUPPORTED_COMM_FIELDS:
        comm = getattr(group_coordinator, field, None)
        if comm is None:
            continue
        logger.warning(
            "group %s: %s (%s) is not checkpointed in v1; graphs of this group are "
            "needs_recapture (%s section 9.4)",
            group,
            field,
            type(comm).__name__,
            _DESIGN,
        )
        states.append(UnsupportedCommGraphState(comm, group, impl))

    return states


class CommCheckpointer:
    """Drives every :class:`CommGraphState` of a process in lockstep.

    Collective ordering contract (design section 6.10):

    * ``export_all`` runs after capture, per rank, no collectives.
    * ``pre_advance_all`` runs on every rank BEFORE the shape loop so shapes
      that fall back to capture allocate ``graph_params`` rows above the
      loaded ones (fact 17).
    * ``restore_all`` runs AFTER the shape loop, inside
      ``parallel_state.graph_capture()``, on every rank at the same program
      point: the row exchange is an ``all_gather_object`` over the group and
      window registration is per comm.
    * Row contents are never copied from the artifact; only indices and
      input regions are, and the live process re-exchanges the pointers.

    Checkpoints are matched to states by position; the two sequences must
    have the same length and every state checks the identity it is handed.
    """

    def __init__(self, states: Sequence[CommGraphState]) -> None:
        self._states: tuple[CommGraphState, ...] = tuple(states)

    @property
    def states(self) -> tuple[CommGraphState, ...]:
        return self._states

    def graph_visible_regions(self) -> list[tuple[RegionSpec, int]]:
        out: list[tuple[RegionSpec, int]] = []
        for state in self._states:
            out.extend(state.graph_visible_regions())
        return out

    def export_all(self, registry: RegionRegistry) -> tuple[CommCheckpoint, ...]:
        return tuple(state.export_graph_state(registry) for state in self._states)

    def _pair(self, ckpts: Sequence[CommCheckpoint]):
        if len(ckpts) != len(self._states):
            raise RuntimeError(
                f"CommCheckpointer: {len(ckpts)} checkpoints for "
                f"{len(self._states)} communicator states"
            )
        return zip(self._states, ckpts)

    def pre_advance_all(self, ckpts: Sequence[CommCheckpoint]) -> None:
        for state, ckpt in self._pair(ckpts):
            state.pre_advance(ckpt)

    def restore_all(
        self, ckpts: Sequence[CommCheckpoint], reloc: RelocationMap
    ) -> None:
        for state, ckpt in self._pair(ckpts):
            state.restore_graph_state(ckpt, reloc)

    @staticmethod
    def any_needs_recapture(ckpts: Sequence[CommCheckpoint]) -> bool:
        """True when any communicator forces the whole group to recapture."""
        return any(ckpt.in_graph_nccl_kernels for ckpt in ckpts)
