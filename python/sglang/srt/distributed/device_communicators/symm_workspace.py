"""Per-group symmetric workspace shared by every push/pull communication kernel.

One workspace is one symmetric-memory allocation for a process group, laid out as
``[push slots | pull staging | pull semaphores]`` per rank plus a rank-local push
phase word. The bytes the kernels exchange form the data plane (push slots and the
pull staging buffer); the synchronization state forms the control plane (the phase
word and the semaphores).

Consumers declare what they need with :func:`request_symm_workspace` and get a
:class:`WorkspaceLease`. Requests accumulate until the workspace is finalized, which
allocates once at the largest size requested in each dimension:
workspace itself on first use if nothing did before. A request that arrives after
finalization attaches to the first workspace it fits, and otherwise opens a
new one. Every rank must issue the same requests in the same order.

Every kernel sharing one workspace must be ordered on one stream: they share the
phase word, the double-buffered slots and the semaphores.
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import TYPE_CHECKING, Dict, Final, List, Optional, Tuple

import msgspec
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.kernels.arch import current_arch

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import Communicator
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

MB = 1024 * 1024

# FABRIC workspaces whose collective release timed out; unmapping them while a
# peer may still be draining would fault it, so they live until process exit.
_unreleased_buffers: list = []

_ALIGN_BYTES = 1024
_SEMAPHORE_BYTES = 128  # sizeof(Semaphore)


def _ceil_align(nbytes: int, align: int = _ALIGN_BYTES) -> int:
    return (nbytes + align - 1) // align * align


class WorkspaceRequest(msgspec.Struct, frozen=True, kw_only=True):
    """What one consumer needs; each dimension is sized independently."""

    owner: str
    push_slot_bytes: int = (
        0  # bytes per push slot; every rank holds 2 * world_size slots
    )
    pull_bytes: int = 0  # per-rank pull staging buffer
    semaphores: int = 0  # largest grid that barriers on the pull semaphores


class WorkspaceCapacity(msgspec.Struct, frozen=True, kw_only=True):
    push_slot_bytes: int
    pull_bytes: int
    semaphores: int

    def fits(self, request: WorkspaceRequest) -> bool:
        return (
            request.push_slot_bytes <= self.push_slot_bytes
            and request.pull_bytes <= self.pull_bytes
            and request.semaphores <= self.semaphores
        )


class DataPlane(msgspec.Struct, frozen=True, kw_only=True):
    """The exchanged bytes: views into every rank's slab, plus multicast VAs."""

    push_slots: List[torch.Tensor]  # per rank: [2 * world_size, push_slot_bytes] uint8
    push_mc: Optional[int]
    pull_buffers: Optional[List[torch.Tensor]]  # per rank: [pull_bytes] uint8
    pull_mc: Optional[int]


class ControlPlane(msgspec.Struct, frozen=True, kw_only=True):
    """The synchronization state: the local phase word and the symmetric semaphores."""

    epoch: torch.Tensor  # rank-local [1] int64: one EpochCounter
    semaphores: List[torch.Tensor]  # per rank: [capacity, 128] uint8
    semaphore_mc: Optional[int]


# NOTE: this is 100% internal and not stable; don't rely on this class
class _Workspace:
    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        from sglang.srt.distributed.parallel_state import in_the_same_node_as

        self.group: Final = group
        self.device: Final = device
        self.rank: Final = dist.get_rank(group=group)
        self.world_size: Final = dist.get_world_size(group=group)
        self._requests: List[WorkspaceRequest] = []
        self._capacity: Optional[WorkspaceCapacity] = None
        self._data: Optional[DataPlane] = None
        self._control: Optional[ControlPlane] = None
        self._symm_owner: Optional[object] = None
        self.multinode: Final = not all(in_the_same_node_as(group, source_rank=0))
        self.has_multicast: Final = is_multicast_supported(group, device)

    @property
    def capacity(self) -> WorkspaceCapacity:
        assert self._capacity is not None, "workspace not finalized"
        return self._capacity

    @property
    def data(self) -> DataPlane:
        assert self._data is not None, "workspace not finalized"
        return self._data

    @property
    def control(self) -> ControlPlane:
        assert self._control is not None, "workspace not finalized"
        return self._control

    @property
    def finalized(self) -> bool:
        return self._capacity is not None

    def finalize(self) -> None:
        if self.finalized:
            return
        assert not torch.cuda.is_current_stream_capturing(), (
            "symmetric workspace must be allocated before CUDA graph capture"
        )

        arch = current_arch()
        slot_bytes = _ceil_align(max([1] + [r.push_slot_bytes for r in self._requests]))
        pull_bytes = max([0] + [r.pull_bytes for r in self._requests])
        pull_bytes = _ceil_align(pull_bytes) if pull_bytes > 0 else 0
        num_semaphores = max([r.semaphores for r in self._requests] + [arch.num_sms])

        self._check_ranks_agree((slot_bytes, pull_bytes, num_semaphores))

        num_slots = 2 * self.world_size  # 2 phases x world_size producers
        push_bytes = num_slots * slot_bytes
        sem_bytes = _SEMAPHORE_BYTES * num_semaphores
        total_bytes = push_bytes + pull_bytes + sem_bytes
        pull_offset, sem_offset = push_bytes, push_bytes + pull_bytes

        self._symm_owner, slabs, multicast_ptr = _allocate_symmetric_memory(
            total_bytes, device=self.device, group=self.group, multinode=self.multinode
        )
        # consumers already branched on the device attribute; the mapping must honour it
        if self.has_multicast and multicast_ptr == 0:
            raise RuntimeError(
                f"{self.group.group_name}: the devices support multicast but the "
                "symmetric allocation has no multicast mapping"
            )
        if not self.has_multicast:
            multicast_ptr = 0
        # Push slots and semaphores must start zeroed; the pull staging rides along.
        slabs[self.rank].zero_()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

        def mc_at(offset: int) -> Optional[int]:
            return multicast_ptr + offset if multicast_ptr != 0 else None

        def slice_all(shape: List[int], offset: int) -> List[torch.Tensor]:
            nbytes = 1
            for dim in shape:
                nbytes *= dim
            assert offset + nbytes <= total_bytes
            return [slab[offset : offset + nbytes].view(shape) for slab in slabs]

        self._data = DataPlane(
            push_slots=slice_all([num_slots, slot_bytes], 0),
            push_mc=mc_at(0),
            pull_buffers=slice_all([pull_bytes], pull_offset) if pull_bytes else None,
            pull_mc=mc_at(pull_offset) if pull_bytes else None,
        )
        self._control = ControlPlane(
            epoch=torch.zeros((1,), dtype=torch.int64, device=self.device),
            semaphores=slice_all([num_semaphores, _SEMAPHORE_BYTES], sem_offset),
            semaphore_mc=mc_at(sem_offset) if num_semaphores else None,
        )
        self._capacity = WorkspaceCapacity(
            push_slot_bytes=slot_bytes,
            pull_bytes=pull_bytes,
            semaphores=num_semaphores,
        )
        if self.rank == 0:
            logger.info(
                "Symmetric workspace (%s, world_size=%d): %.2f MB, slot = %d B, "
                "pull = %d B, semaphores = %d, multicast = %s, owners = %s",
                self.group.group_name,
                self.world_size,
                total_bytes / MB,
                slot_bytes,
                pull_bytes,
                num_semaphores,
                self.has_multicast,
                sorted({r.owner for r in self._requests}),
            )
        dist.barrier(group=self.group)

    def _check_ranks_agree(self, layout: Tuple[int, int, int]) -> None:
        """Fail loudly, before the collective allocation, if ranks disagree.

        Every rank must issue the same requests in the same order; a mismatch would
        otherwise hang or mis-size the symmetric rendezvous.
        """
        mine = (
            layout,
            [
                (r.owner, r.push_slot_bytes, r.pull_bytes, r.semaphores)
                for r in self._requests
            ],
        )
        everyone: List[object] = [None] * self.world_size
        dist.all_gather_object(everyone, mine, group=self.group)
        if any(other != mine for other in everyone):
            lines = "\n".join(
                f"  rank {i}: layout={lay} requests={reqs}"
                for i, (lay, reqs) in enumerate(everyone)  # type: ignore
            )
            raise RuntimeError(
                f"symmetric workspace requests differ across ranks of "
                f"{self.group.group_name}:\n{lines}"
            )

    def close(self) -> None:
        if self._symm_owner is None:
            return
        owner, self._symm_owner = self._symm_owner, None
        self._data = self._control = None  # drop the views before the memory
        _release_symmetric_memory(owner, self.group, self.device, self.multinode)


class SymmetricBuffer:
    """A raw symmetric allocation on ``group``, outside the shared workspace.

    For private layouts the workspace does not model, such as the DPX pair / cross
    planes of custom all-reduce v2. Zero-filled; :meth:`close` frees it (collective).
    """

    def __init__(self, group: ProcessGroup, device: torch.device, nbytes: int) -> None:
        from sglang.srt.distributed.parallel_state import in_the_same_node_as

        self.group: Final = group
        self.device: Final = device
        self.multinode: Final = not all(in_the_same_node_as(group, source_rank=0))
        self._owner, self.slabs, self.multicast_ptr = _allocate_symmetric_memory(
            nbytes, device=device, group=group, multinode=self.multinode
        )
        self.slabs[dist.get_rank(group=group)].zero_()
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def close(self) -> None:
        """Free the allocation; drop every view into ``slabs`` first."""
        if self._owner is None:
            return
        owner, self._owner = self._owner, None
        self.slabs = []
        _release_symmetric_memory(owner, self.group, self.device, self.multinode)


def _release_symmetric_memory(
    owner: object, group: ProcessGroup, device: torch.device, multinode: bool
) -> None:
    fabric = _is_hip() and multinode
    peers_alive = False
    if fabric:
        from sglang.srt.utils.hip_vmm_utils import teardown_barrier

        # Free a FABRIC allocation only after every rank's kernels have drained;
        # a peer that never arrives leaves it to process exit.
        peers_alive = teardown_barrier(group)
        if peers_alive:
            torch.cuda.synchronize(device)
            peers_alive = teardown_barrier(group)
    else:
        dist.barrier(group=group)
    if fabric:
        if peers_alive:
            owner.close(group=group)  # type: ignore
        else:
            _unreleased_buffers.append(owner)


class WorkspaceLease(msgspec.Struct, frozen=True, kw_only=True):
    """One consumer's handle on a (possibly shared) symmetric workspace.

    Every accessor finalizes the workspace on first use if it is still open, which is
    a collective: all ranks must reach it together.

    """

    request: WorkspaceRequest
    group: ProcessGroup
    device: torch.device
    rank: int
    world_size: int
    has_multicast: bool
    # NOTE: never access the fields below outside this file; not stable API
    _workspace: _Workspace
    _cached_communicators: Dict[Tuple[Optional[int], Optional[int]], Communicator]

    @staticmethod
    def new(workspace: _Workspace, request: WorkspaceRequest) -> WorkspaceLease:
        return WorkspaceLease(
            request=request,
            group=workspace.group,
            device=workspace.device,
            rank=workspace.rank,
            world_size=workspace.world_size,
            has_multicast=workspace.has_multicast,
            _workspace=workspace,
            _cached_communicators={},
        )

    def _get_finalized(self) -> _Workspace:
        if not self._workspace.finalized:
            self._workspace.finalize()
        return self._workspace

    @property
    def capacity(self) -> WorkspaceCapacity:
        return self._get_finalized().capacity

    @property
    def data(self) -> DataPlane:
        return self._get_finalized().data

    @property
    def control(self) -> ControlPlane:
        return self._get_finalized().control

    def get_communicator(
        self,
        *,
        push_blocks: Optional[int] = None,
        pull_blocks: Optional[int] = None,
    ) -> Communicator:
        """This consumer's planes over the shared storage, with its own launch widths.

        The planes default to one block per SM; the result is cached per widths.
        """
        # launch paths call this per step: hit the cache on the arguments as given
        cached = self._cached_communicators.get((push_blocks, pull_blocks))
        if cached is not None:
            return cached
        from sglang.kernels.ops.communication.all_reduce import (
            Communicator,
            PullPlane,
            PushPlane,
        )

        num_sm = current_arch().num_sms
        key = (push_blocks or num_sm, pull_blocks or num_sm)
        if key not in self._cached_communicators:
            ws = self._get_finalized()
            push = PushPlane(
                self.rank,
                self.world_size,
                workspaces=ws.data.push_slots,
                counter=ws.control.epoch,
                default_blocks=key[0],
                mc_workspace=ws.data.push_mc,
            )
            pull = PullPlane(
                self.rank,
                self.world_size,
                workspaces=ws.data.pull_buffers,
                semaphores=ws.control.semaphores,
                mc_workspace=ws.data.pull_mc,
                mc_semaphore=ws.control.semaphore_mc,
                default_blocks=key[1],
            )
            self._cached_communicators[key] = Communicator(push=push, pull=pull)
        comm = self._cached_communicators[key]
        self._cached_communicators[(push_blocks, pull_blocks)] = comm
        return comm


_WORKSPACES: Dict[str, List[_Workspace]] = {}
_SUPPORTED: Dict[str, bool] = {}
_MULTICAST: Dict[str, bool] = {}
_V2_PUSH_SLOT_BYTES: Dict[str, int] = {}

# ``WorkspaceRequest.push_slot_bytes`` placeholder: the group's custom all-reduce slot
V2_PUSH_SLOT = -1


def register_v2_push_slot_bytes(group: ProcessGroup, nbytes: int) -> None:
    """Record the push slot of ``group``'s custom all-reduce (v2), for :data:`V2_PUSH_SLOT`.

    The first registration per group wins: that is the group's own instance, built
    when the group is.
    """
    _V2_PUSH_SLOT_BYTES.setdefault(group.group_name, nbytes)


# TODO: this should be removed in the future
def v2_push_slot_bytes(group: ProcessGroup) -> Optional[int]:
    """The push slot registered for ``group``'s custom all-reduce, or None."""
    return _V2_PUSH_SLOT_BYTES.get(group.group_name)


def is_multicast_supported(group: ProcessGroup, device: torch.device) -> bool:
    """Whether every rank's device can map multicast memory (collective, cached).

    A device attribute, independent of any workspace: callers may branch on it at
    model construction without allocating.
    """
    name = group.group_name
    if name not in _MULTICAST:
        local = not _is_hip() and _device_has_multicast(device)
        everyone: List[object] = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(everyone, local, group=group)
        _MULTICAST[name] = all(everyone)
    return _MULTICAST[name]


def _device_has_multicast(device: torch.device) -> bool:
    from torch._C._autograd import DeviceType
    from torch._C._distributed_c10d import _SymmetricMemory

    # torch then maps symmetric memory without a multicast address
    if os.environ.get("TORCH_SYMM_MEM_DISABLE_MULTICAST", "0") not in ("", "0"):
        return False

    index = device.index if device.index is not None else torch.cuda.current_device()
    return bool(_SymmetricMemory.has_multicast_support(DeviceType.CUDA, index))


def is_symm_workspace_supported(group: ProcessGroup, device: torch.device) -> bool:
    """Whether the push/pull kernels can run over this group (collective, cached)."""
    name = group.group_name
    if name not in _SUPPORTED:
        from .custom_all_reduce_v2 import can_use_custom_all_reduce_v2

        _SUPPORTED[name] = can_use_custom_all_reduce_v2(group=group, device=device)
    return _SUPPORTED[name]


def request_symm_workspace(
    group: ProcessGroup,
    device: torch.device,
    request: WorkspaceRequest,
) -> Optional[WorkspaceLease]:
    """Declare ``request`` on ``group``'s workspace; None when the group cannot host one.

    ``request.push_slot_bytes == V2_PUSH_SLOT`` (-1) asks for the push slot of the
    group's custom all-reduce, as registered by :func:`register_v2_push_slot_bytes`.
    It resolves here, to a fixed number stored in the lease's request, so it does not
    depend on which workspace the request lands in or on when that is allocated. A
    group with no registered slot cannot host such a request (None).
    """
    if not is_symm_workspace_supported(group, device):
        return None
    if request.push_slot_bytes == V2_PUSH_SLOT:
        nbytes = v2_push_slot_bytes(group)
        if nbytes is None:
            return None
        request = msgspec.structs.replace(request, push_slot_bytes=nbytes)
    workspaces = _WORKSPACES.setdefault(group.group_name, [])
    for workspace in reversed(workspaces):
        if not workspace.finalized or workspace.capacity.fits(request):
            workspace._requests.append(request)
            return WorkspaceLease.new(workspace, request)
    # TODO(dark): support borrow push & pull workspace separately
    if all(not ws for ws in _WORKSPACES.values()):
        atexit.register(_close_quietly)
    workspace = _Workspace(group, device)
    workspace._requests.append(request)
    workspaces.append(workspace)
    return WorkspaceLease.new(workspace, request)


def request_group_workspace(
    coordinator: GroupCoordinator,
    request: WorkspaceRequest,
) -> Optional[WorkspaceLease]:
    """:func:`request_symm_workspace` on a ``GroupCoordinator``'s group, the one its
    custom all-reduce uses, so every consumer of that group shares one workspace."""
    return request_symm_workspace(coordinator.cpu_group, coordinator.device, request)


def close_symm_workspaces() -> None:
    for workspaces in _WORKSPACES.values():
        for workspace in workspaces:
            workspace.close()
    _WORKSPACES.clear()


def is_multinode_symm_group(group: ProcessGroup) -> bool:
    """Whether ``group`` holds a workspace that spans nodes (FABRIC on ROCm)."""
    return any(ws.multinode for ws in _WORKSPACES.get(group.group_name, ()))


def close_multinode_symm_workspaces() -> None:
    """Release the workspaces that span nodes while their groups still exist.

    Graceful shutdown aborts the process groups rather than destroying them, and a
    FABRIC workspace is freed collectively on its group, so this must run first.
    """
    for name in sorted(_WORKSPACES):
        workspaces = _WORKSPACES[name]
        if any(ws.multinode for ws in workspaces):
            for workspace in workspaces:
                workspace.close()
            del _WORKSPACES[name]


def _close_quietly() -> None:
    try:
        close_symm_workspaces()
    except Exception:
        pass


def _is_hip() -> bool:
    from sglang.srt.utils import is_hip

    return is_hip()


def _allocate_symmetric_memory(
    nbytes: int, device: torch.device, group: ProcessGroup, multinode: bool = False
) -> Tuple[object, List[torch.Tensor], int]:
    """``(owner, per-rank uint8 views, multicast VA or 0)``; keep ``owner`` alive."""
    if _is_hip():
        from sglang.srt.utils.hip_vmm_utils import HipSymmetricBuffer

        buffer = HipSymmetricBuffer(nbytes, device, group, require_fabric=multinode)
        return buffer, buffer.buffers, 0

    import torch.distributed._symmetric_memory as symm_mem

    tensor = symm_mem.empty((nbytes,), dtype=torch.uint8, device=device)
    symm_buf = symm_mem.rendezvous(tensor, group=group)
    slabs = [
        symm_buf.get_buffer(i, [nbytes], torch.uint8)
        for i in range(dist.get_world_size(group=group))
    ]
    return (tensor, symm_buf), slabs, int(symm_buf.multicast_ptr)
