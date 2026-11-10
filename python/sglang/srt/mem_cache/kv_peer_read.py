# SPDX-License-Identifier: Apache-2.0
"""Read KV pages out of a peer engine's GPU memory over NVLink (CUDA FABRIC handles).

The owner exports one FABRIC allocation per lease (:class:`KvPeerExporter`); its
control plane hands the descriptor bytes to one reader. The reader
(:class:`KvPeerReader`) imports it read-only and gathers pages into its own,
borrowed, destination buffers with a page-index kernel on a private stream.

Contract:

- A batch is validated as a whole before anything is enqueued: page bounds on
  both sides, destination overlap within the batch (ValueError) and against
  this reader's in-flight batches (PeerReadBusy), and that destinations are
  memory of this reader's device at both ends and not an imported region.
  A rejected batch never touches memory.
- Groups of a batch complete in order, one event each. A group that completed
  stays completed when a later group fails.
- A failed batch is FAILED_DRAINED when the batch's last recorded group event
  synchronizes cleanly, FAILED_UNCERTAIN otherwise. Only SUCCESS and
  FAILED_DRAINED batches can be released; an uncertain batch, including one
  whose enqueue failed without draining, pins its regions and destination
  pages for the life of the process. A sticky CUDA error fail-stops the
  reader; it also leaves the process's CUDA context unusable.
- ``wait(timeout)`` only stops waiting; it never cancels or releases anything.
- The owner releases a lease only after the reader has released every batch on
  the region, unmapped it and acknowledged; never because of a timeout.
- Source pages must not be rewritten while a read is in flight, and borrowed
  destinations must stay allocated until release; both are the callers' job.
"""

from __future__ import annotations

import enum
import functools
import os
import threading
import time
import uuid
from typing import Any, Optional, Sequence

import msgspec
import numpy as np

# Destination bases, source offsets and page sizes must be multiples of this.
PAGE_ALIGN_BYTES = 128

_DESCRIPTOR_VERSION = 1
# Replays of this many most recent leases are rejected; arbitrary.
_MAX_LEASE_TOMBSTONES = 1 << 16
# Arbitrary: covers one buffer tuple per layer group of a few pools.
_MAX_BUFFER_SETS = 256
_FABRIC_HANDLE_BYTES = 64


class ReadState(str, enum.Enum):
    IN_FLIGHT = "IN_FLIGHT"
    SUCCESS = "SUCCESS"
    FAILED_DRAINED = "FAILED_DRAINED"
    FAILED_UNCERTAIN = "FAILED_UNCERTAIN"


class PeerReadError(RuntimeError):
    pass


class PeerReadBusy(PeerReadError):
    """The object is still in use (in-flight, exported, or quarantined)."""


class PeerRegionDescriptor(msgspec.Struct, frozen=True):
    """What a reader needs to map one exported allocation. Opaque to the control plane."""

    version: int
    owner_id: str
    region_id: str
    lease_id: str
    size: int
    fabric_handle: bytes
    # (cluster_uuid, clique_id) of the owner GPU; None when NVML could not tell.
    fabric_clique: Optional[tuple[bytes, int]] = None

    def validate(self) -> None:
        if self.version != _DESCRIPTOR_VERSION:
            raise ValueError(f"unsupported descriptor version {self.version}")
        if not (self.owner_id and self.region_id and self.lease_id):
            raise ValueError("descriptor identities must be non-empty")
        if self.size <= 0:
            raise ValueError("descriptor size must be positive")
        if len(self.fabric_handle) != _FABRIC_HANDLE_BYTES:
            raise ValueError("FABRIC handle must be 64 bytes")

    def to_bytes(self) -> bytes:
        self.validate()
        return msgspec.msgpack.encode(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> PeerRegionDescriptor:
        try:
            desc = msgspec.msgpack.decode(data, type=cls)
        except msgspec.DecodeError as error:
            raise ValueError(f"malformed peer region descriptor: {error}") from None
        desc.validate()
        return desc


class PeerRegion:
    """Reader-side handle of an imported region."""

    __slots__ = ("region_id", "size", "address")

    def __init__(self, region_id: str, size: int, address: int):
        self.region_id = region_id
        self.size = size
        self.address = address


class PeerBuffer(msgspec.Struct, frozen=True):
    """One per-layer buffer: page ``p`` lives at ``src_offset + p * page_bytes`` in
    ``src_region`` and goes to ``dst_base + p * page_bytes`` on the reader."""

    src_region: PeerRegion
    src_offset: int
    dst_base: int
    page_bytes: int
    # Capacity of the destination buffer in pages; bounds dst_pages.
    dst_num_pages: int


class ReadGroup(msgspec.Struct, frozen=True):
    """Buffers that share one page list and complete together (typically a layer group).

    ``src_pages`` / ``dst_pages`` are equal-length int64 page ids on the host
    (numpy array, CPU tensor or sequence); every buffer copies the same pages.
    """

    buffers: tuple[PeerBuffer, ...]
    src_pages: Any
    dst_pages: Any


class ReadStatus(msgspec.Struct, frozen=True):
    state: ReadState
    groups_done: int
    num_groups: int
    error: str = ""


class ReadBatch:
    __slots__ = ("batch_id",)

    def __init__(self, batch_id: str):
        self.batch_id = batch_id


# ===== CUDA access =====
# Everything below talks to the GPU through this object, so the contract tests can
# substitute a fake. One launch per distinct page size of a group.
# Enqueueing a group (launch + record) allocates nothing, never syncs the host and
# never compiles, so it can also run as a deferred, gated chunk (a TransferChain
# ChunkOp) on a stream the reader does not own; drain is judged by the group events.


class _CudaOps:
    def __init__(self, *, num_blocks: int):
        from cuda.bindings import driver

        self._drv = driver
        self._num_blocks = num_blocks
        self.not_ready = int(driver.CUresult.CUDA_ERROR_NOT_READY)
        self._sticky = {
            int(getattr(driver.CUresult, name))
            for name in (
                "CUDA_ERROR_ILLEGAL_ADDRESS",
                "CUDA_ERROR_LAUNCH_FAILED",
                "CUDA_ERROR_ILLEGAL_INSTRUCTION",
                "CUDA_ERROR_MISALIGNED_ADDRESS",
                "CUDA_ERROR_INVALID_ADDRESS_SPACE",
                "CUDA_ERROR_INVALID_PC",
                "CUDA_ERROR_HARDWARE_STACK_ERROR",
                "CUDA_ERROR_ASSERT",
                "CUDA_ERROR_ECC_UNCORRECTABLE",
                "CUDA_ERROR_NVLINK_UNCORRECTABLE",
                "CUDA_ERROR_CONTEXT_IS_DESTROYED",
            )
        }

    def fabric_clique(self, device: int):
        from sglang.srt.utils.cuda_vmm_utils import get_gpu_fabric_clique

        return get_gpu_fabric_clique(device)

    def export_handle(self, handle) -> bytes:
        from sglang.srt.utils.cuda_vmm_utils import export_fabric_handle

        return export_fabric_handle(handle)

    def map_import(self, fabric_handle: bytes, size: int, device: int) -> int:
        from sglang.srt.utils.cuda_vmm_utils import map_fabric_import

        return map_fabric_import(fabric_handle, size, device, read_only=True)

    def unmap_import(self, address: int, size: int) -> None:
        from sglang.srt.utils.cuda_vmm_utils import unmap_fabric_import

        unmap_fabric_import(address, size)

    def pointer_device(self, pointer: int) -> Optional[int]:
        drv = self._drv
        attr = drv.CUpointer_attribute
        err, kind = drv.cuPointerGetAttribute(
            attr.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer
        )
        if err != drv.CUresult.CUDA_SUCCESS or int(kind) != int(
            drv.CUmemorytype.CU_MEMORYTYPE_DEVICE
        ):
            return None
        err, ordinal = drv.cuPointerGetAttribute(
            attr.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, pointer
        )
        return int(ordinal) if err == drv.CUresult.CUDA_SUCCESS else None

    def new_stream(self, device: int):
        import torch

        return torch.cuda.Stream(device=device)

    def wait_stream(self, stream, producer) -> None:
        stream.wait_stream(producer)

    def warmup(self, stream) -> None:
        """Compile and load the gather now: a first launch must not happen inside a group."""
        import torch

        with torch.cuda.stream(stream):
            buf = torch.zeros(
                2 * PAGE_ALIGN_BYTES, dtype=torch.uint8, device=stream.device
            )
            ptrs = torch.tensor(
                [buf.data_ptr(), buf.data_ptr() + PAGE_ALIGN_BYTES],
                dtype=torch.int64,
                device=stream.device,
            ).view(torch.uint64)
            page = torch.zeros(1, dtype=torch.int64, device=stream.device)
            self._gather(ptrs[:1], ptrs[1:], page, page, PAGE_ALIGN_BYTES)
        stream.synchronize()

    def current(self, stream):
        import torch

        return torch.cuda.stream(stream)

    def stage(self, stream, meta: np.ndarray, slices):
        """Upload one batch's metadata (int64) on ``stream``; return what must stay
        alive until release, and per group the launch arguments as device views."""
        import torch

        host = torch.from_numpy(meta).pin_memory()
        with torch.cuda.stream(stream):
            dev = host.to(stream.device, non_blocking=True)
        views = [
            [
                (
                    page_bytes,
                    dev[sb : sb + nb].view(torch.uint64),
                    dev[db : db + nb].view(torch.uint64),
                    dev[sp : sp + n],
                    dev[dp : dp + n],
                )
                for page_bytes, ((sb, nb), (db, _), (sp, n), (dp, _)) in group
            ]
            for group in slices
        ]
        return (host, dev), views

    def launch(self, view) -> None:
        """Enqueue one gather on the current stream."""
        page_bytes, src_ptrs, dst_ptrs, src_pages, dst_pages = view
        self._gather(src_ptrs, dst_ptrs, src_pages, dst_pages, page_bytes)

    def _gather(self, src_ptrs, dst_ptrs, src_pages, dst_pages, page_bytes) -> None:
        from sglang.kernels.ops.kvcache.peer_page_gather import gather_pages

        gather_pages(
            src_ptrs=src_ptrs,
            dst_ptrs=dst_ptrs,
            src_pages=src_pages,
            dst_pages=dst_pages,
            page_bytes=page_bytes,
            num_blocks=self._num_blocks,
        )

    def new_event(self, stream):
        """An event in ``stream``'s context, whatever context is current."""
        drv = self._drv
        err, ctx = drv.cuStreamGetCtx(stream.cuda_stream)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise PeerReadError(f"cuStreamGetCtx: {err}")
        if drv.cuCtxPushCurrent(ctx)[0] != drv.CUresult.CUDA_SUCCESS:
            raise PeerReadError("cuCtxPushCurrent failed")
        try:
            err, event = drv.cuEventCreate(drv.CUevent_flags.CU_EVENT_DISABLE_TIMING)
        finally:
            drv.cuCtxPopCurrent()
        if err != drv.CUresult.CUDA_SUCCESS:
            raise PeerReadError(f"cuEventCreate: {err}")
        return event

    def record(self, event, stream) -> None:
        err = self._drv.cuEventRecord(event, stream.cuda_stream)[0]
        if err != self._drv.CUresult.CUDA_SUCCESS:
            raise PeerReadError(f"cuEventRecord: {err}")

    def query_event(self, event) -> int:
        return int(self._drv.cuEventQuery(event)[0])

    def sync_event(self, event) -> int:
        return int(self._drv.cuEventSynchronize(event)[0])

    def destroy_event(self, event) -> None:
        self._drv.cuEventDestroy(event)

    def is_sticky(self, code: int) -> bool:
        return code in self._sticky


# ===== Owner side =====


def _checked_int(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value < 2**63:
        raise ValueError(f"{name} must be in [{minimum}, 2**63)")
    return value


def _guarded(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        if os.getpid() != self._pid:
            raise PeerReadError("peer KV objects cannot be used across fork")
        with self._lock:
            return method(self, *args, **kwargs)

    return call


class KvPeerExporter:
    """Hands out read leases on the owner's FABRIC allocations.

    The allocation stays the caller's; it must not be freed or remapped while
    :meth:`outstanding` reports a lease on it.
    """

    def __init__(self, device: int, *, _ops=None):
        self.device = _checked_int(device, "device")
        self.owner_id = uuid.uuid4().hex
        self._ops = _ops if _ops is not None else _CudaOps(num_blocks=1)
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._leases: dict[str, str] = {}  # lease_id -> region_id
        self._clique = self._ops.fabric_clique(self.device)

    @_guarded
    def export(self, handle, size: int, *, region_id: str) -> PeerRegionDescriptor:
        """One lease for one reader on the allocation ``handle`` of ``size`` bytes."""
        size = _checked_int(size, "size", 1)
        if not isinstance(region_id, str) or not region_id:
            raise ValueError("region_id must be a non-empty string")
        clique = self._clique
        desc = PeerRegionDescriptor(
            version=_DESCRIPTOR_VERSION,
            owner_id=self.owner_id,
            region_id=region_id,
            lease_id=uuid.uuid4().hex,
            size=size,
            fabric_handle=self._ops.export_handle(handle),
            fabric_clique=None
            if clique is None
            else (bytes(clique[0]), int(clique[1])),
        )
        desc.validate()
        self._leases[desc.lease_id] = region_id
        return desc

    @_guarded
    def release_export(self, lease_id: str) -> None:
        """Call only after the reader acknowledged release and unmap of this lease."""
        if self._leases.pop(lease_id, None) is None:
            raise ValueError("unknown or already released lease")

    @_guarded
    def outstanding(self, region_id: Optional[str] = None) -> int:
        if region_id is None:
            return len(self._leases)
        return sum(r == region_id for r in self._leases.values())


# ===== Reader side =====


class _Region:
    __slots__ = ("public", "desc", "batches")

    def __init__(self, public: PeerRegion, desc: PeerRegionDescriptor):
        self.public = public
        self.desc = desc
        self.batches: set[str] = set()


class _BufferSet:
    """A validated ``ReadGroup.buffers`` tuple.

    Pools keep their buffers for their lifetime, so callers that reuse the tuple
    pay the per-buffer checks (alignment, destination device) once.
    """

    __slots__ = ("buffers", "regions", "launches", "dst_ends", "src_limit", "dst_limit")

    def __init__(self, buffers, regions, launches, dst_ends, src_limit, dst_limit):
        self.buffers = buffers  # the caller's tuple; identity is the cache key
        self.regions = regions  # the source regions the buffers read, by region id
        self.launches = launches  # [(page_bytes, src_bases, dst_bases)]
        # (dst base, page_bytes) -> dst end, per buffer. Two writers can share a
        # destination byte only through the same key (same base and page size).
        self.dst_ends = dst_ends
        self.src_limit = src_limit  # source page ids must be below this
        self.dst_limit = dst_limit  # destination page ids must be below this


class _GroupPlan:
    __slots__ = ("buffer_set", "launches", "src_pages", "dst_pages", "dst_sorted")

    def __init__(self, buffer_set, src_pages, dst_pages, dst_sorted):
        self.buffer_set = buffer_set
        self.launches = buffer_set.launches  # [(page_bytes, src_bases, dst_bases)]
        self.src_pages, self.dst_pages, self.dst_sorted = (
            src_pages,
            dst_pages,
            dst_sorted,
        )


def _layout(plans):
    """Pack every group's tables into one int64 array; return it and per-launch slices.

    A slice set is ((src_bases), (dst_bases), (src_pages), (dst_pages)), each as
    (offset, length) into the array.
    """
    parts, slices, offset, placed = [], [], 0, {}

    def put(array):  # groups usually share their page arrays: upload each once
        nonlocal offset
        hit = placed.get(id(array))
        if hit is not None:
            return hit
        parts.append(array)
        offset += array.size
        placed[id(array)] = (offset - array.size, array.size)
        return placed[id(array)]

    for plan in plans:
        pages = (put(plan.src_pages), put(plan.dst_pages))
        group = []
        for page_bytes, src_bases, dst_bases in plan.launches:
            group.append((page_bytes, (put(src_bases), put(dst_bases)) + pages))
        slices.append(group)
    return np.concatenate(parts, dtype=np.int64), slices


class _Batch:
    __slots__ = (
        "public",
        "regions",
        "events",
        "recorded",
        "unrecorded",
        "keepalive",
        "destinations",
        "done",
        "state",
        "error",
    )

    def __init__(self, public, regions, destinations):
        self.public = public
        self.regions = regions  # the _Regions this batch reads
        self.events = []
        self.recorded = []  # recorded[i]: group i is enqueued and its event recorded
        self.unrecorded = False
        self.keepalive = []
        self.destinations = destinations  # [(buffer set, sorted dst pages)] per group
        self.done = 0
        self.state = ReadState.IN_FLIGHT
        self.error = ""


def _first_overlap(ends: dict):
    """Two (base, page size) keys whose [base, end) ranges overlap, or None."""
    previous = None
    for key in sorted(ends):
        if previous is not None and key[0] < ends[previous]:
            return previous, key
        if previous is None or ends[key] > ends[previous]:
            previous = key
    return None


def _page_info(value, name: str, cache: dict):
    """Validate a page-id array once per read: ``(pages, sorted pages, max)``."""
    hit = cache.get(id(value))
    if hit is not None and hit[0] is value:
        return hit[1]
    pages = _as_pages(value, name)
    ordered = np.sort(pages)
    info = (pages, ordered, int(ordered[-1]))
    cache[id(value)] = (value, info)  # holds value, so its id stays unique
    return info


def _as_pages(value, name: str) -> np.ndarray:
    if hasattr(value, "is_cuda") and value.is_cuda:
        raise ValueError(f"{name} must be on the host; it is validated before launch")
    if hasattr(value, "numpy"):
        value = value.numpy()
    pages = np.asarray(value)
    if pages.ndim != 1 or pages.size == 0 or pages.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a non-empty 1-D integer array")
    pages = pages.astype(np.int64, copy=False)
    if int(pages.min()) < 0:
        raise ValueError(f"{name} has a negative page id")
    return pages


class KvPeerReader:
    """Imports peer regions and reads pages out of them on one device.

    ``num_blocks`` is the gather's CTA budget: more blocks copy faster but take
    SMs from whatever else runs on the device.
    """

    def __init__(self, device: int, *, num_blocks: int = 64, _ops=None):
        self.device = _checked_int(device, "device")
        _checked_int(num_blocks, "num_blocks", 1)
        self._ops = _ops if _ops is not None else _CudaOps(num_blocks=num_blocks)
        self._stream = self._ops.new_stream(self.device)
        self._ops.warmup(self._stream)
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._regions: dict[str, _Region] = {}
        self._batches: dict[str, _Batch] = {}
        self._imported_leases: dict[tuple[str, str], None] = {}  # ordered set
        self._buffer_sets: dict[int, _BufferSet] = {}
        # Destinations of batches not yet known to be done: batch id -> groups, and
        # (dst base, page_bytes) -> [dst end, number of such groups using it].
        self._busy_groups: dict[str, list] = {}
        self._busy_keys: dict[tuple[int, int], list] = {}
        self._clique = self._ops.fabric_clique(self.device)
        self._fail_stop = ""

    # ---- regions

    @_guarded
    def import_region(self, desc: PeerRegionDescriptor) -> PeerRegion:
        self._check_alive()
        if not isinstance(desc, PeerRegionDescriptor):
            raise ValueError("expected a PeerRegionDescriptor")
        desc.validate()
        lease_key = (desc.owner_id, desc.lease_id)
        if lease_key in self._imported_leases:
            raise ValueError("lease already imported; obtain a fresh export")
        mine = self._clique
        if (
            desc.fabric_clique is not None
            and mine is not None
            and tuple(desc.fabric_clique) != (bytes(mine[0]), int(mine[1]))
        ):
            raise PeerReadError("owner GPU is in a different NVLink fabric clique")
        address = self._ops.map_import(desc.fabric_handle, desc.size, self.device)
        public = PeerRegion(uuid.uuid4().hex, desc.size, address)
        self._regions[public.region_id] = _Region(public, desc)
        self._imported_leases[lease_key] = None  # survives unmap: a lease maps once
        if len(self._imported_leases) > _MAX_LEASE_TOMBSTONES:
            del self._imported_leases[next(iter(self._imported_leases))]
        return public

    @_guarded
    def unmap(self, region: PeerRegion) -> None:
        """Unmap after every batch on the region is released; then the control plane may ACK."""
        item = self._region(region)
        if item.batches:
            raise PeerReadBusy("region has unreleased batches")
        self._ops.unmap_import(region.address, region.size)
        del self._regions[region.region_id]
        for key in [
            k for k, v in self._buffer_sets.items() if region.region_id in v.regions
        ]:
            del self._buffer_sets[key]

    # ---- batches

    @_guarded
    def read(self, groups: Sequence[ReadGroup], *, producer_stream=None) -> ReadBatch:
        """Enqueue ``groups`` in order; their buffers may read any imported regions.
        ``producer_stream`` (a torch stream of this device) is waited on first, for
        destinations its last user is still writing."""
        self._check_alive()
        groups = tuple(groups)
        if not groups:
            raise ValueError("empty read batch")
        page_cache: dict = {}
        sets = [self._buffer_set(g, i) for i, g in enumerate(groups)]
        plans = [
            self._plan_group(g, i, bs, page_cache)
            for i, (g, bs) in enumerate(zip(groups, sets))
        ]
        self._check_destinations(plans)
        self._check_alive()  # polling in-flight batches above can fail-stop
        public = ReadBatch(uuid.uuid4().hex)
        regions = {rid: self._regions[rid] for bs in sets for rid in bs.regions}
        batch = _Batch(
            public,
            list(regions.values()),
            [(p.buffer_set, p.dst_sorted) for p in plans],
        )
        try:
            self._enqueue(batch, plans, producer_stream)
        except BaseException:
            if batch.state == ReadState.FAILED_UNCERTAIN:
                self._admit(batch)  # pins its regions and destinations
            raise
        self._admit(batch)
        return public

    @_guarded
    def status(self, batch: ReadBatch) -> ReadStatus:
        item = self._batch(batch)
        self._poll(item)
        return ReadStatus(item.state, item.done, len(item.events), item.error)

    @_guarded
    def group_done(self, batch: ReadBatch, index: int) -> bool:
        item = self._batch(batch)
        if not 0 <= index < len(item.events):
            raise IndexError("group index out of range")
        self._poll(item)
        return index < item.done

    def wait(self, batch: ReadBatch, timeout: Optional[float] = None) -> ReadStatus:
        """Block until the batch leaves IN_FLIGHT. On timeout raise TimeoutError and keep everything."""
        if timeout is not None and not (
            isinstance(timeout, (int, float)) and 0 <= timeout < float("inf")
        ):
            raise ValueError("timeout must be finite and non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        pause = 20e-6
        while True:
            result = self.status(batch)
            if result.state != ReadState.IN_FLIGHT:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("read still in flight; nothing may be released")
            time.sleep(pause)
            pause = min(pause * 2, 1e-3)

    @_guarded
    def release(self, batch: ReadBatch) -> None:
        item = self._batch(batch)
        self._poll(item)
        if item.state not in (ReadState.SUCCESS, ReadState.FAILED_DRAINED):
            raise PeerReadBusy(f"batch is {item.state.value}; it cannot be released")
        for event in item.events:
            self._ops.destroy_event(event)
        self._retire(item)
        for region in item.regions:
            region.batches.discard(batch.batch_id)
        del self._batches[batch.batch_id]

    @_guarded
    def close(self) -> None:
        if self._batches or self._regions:
            raise PeerReadBusy("release batches and unmap regions before close")

    # ---- helpers

    def _check_alive(self) -> None:
        if self._fail_stop:
            raise PeerReadError(f"reader fail-stopped: {self._fail_stop}")

    def _region(self, region) -> _Region:
        item = (
            self._regions.get(region.region_id)
            if isinstance(region, PeerRegion)
            else None
        )
        if item is None or item.public is not region:
            raise ValueError("foreign, forged or unmapped region")
        return item

    def _batch(self, batch) -> _Batch:
        item = (
            self._batches.get(batch.batch_id) if isinstance(batch, ReadBatch) else None
        )
        if item is None or item.public is not batch:
            raise ValueError("foreign, forged or released batch")
        return item

    def _plan_group(
        self, group: ReadGroup, index: int, buffer_set: _BufferSet, page_cache: dict
    ) -> _GroupPlan:
        src_pages, _, src_max = _page_info(group.src_pages, "src_pages", page_cache)
        dst_pages, dst_sorted, dst_max = _page_info(
            group.dst_pages, "dst_pages", page_cache
        )
        if src_pages.size != dst_pages.size:
            raise ValueError(f"group {index}: src_pages and dst_pages differ in length")
        if dst_sorted.size > 1 and bool((dst_sorted[1:] == dst_sorted[:-1]).any()):
            raise ValueError(f"group {index}: overlapping destination pages")
        if src_max >= buffer_set.src_limit:
            raise ValueError(f"group {index}: source page beyond its region")
        if dst_max >= buffer_set.dst_limit:
            raise ValueError(f"group {index}: destination page beyond the buffer")
        return _GroupPlan(buffer_set, src_pages, dst_pages, dst_sorted)

    def _buffer_set(self, group: ReadGroup, index: int) -> _BufferSet:
        if not isinstance(group, ReadGroup) or not group.buffers:
            raise ValueError(f"group {index}: expected a ReadGroup with buffers")
        buffers = group.buffers
        cached = self._buffer_sets.get(id(buffers))
        if cached is not None and cached.buffers is buffers:
            return cached  # unmap evicts sets that read the unmapped region
        by_size: dict[int, tuple[list, list]] = {}
        dst_ends, regions = {}, {}
        src_limit = dst_limit = 2**63
        for buf in buffers:
            if not isinstance(buf, PeerBuffer):
                raise ValueError(f"group {index}: expected PeerBuffer entries")
            region = self._region(buf.src_region)
            page_bytes = _checked_int(buf.page_bytes, "page_bytes", 1)
            src_offset = _checked_int(buf.src_offset, "src_offset")
            dst_base = _checked_int(buf.dst_base, "dst_base", 1)
            capacity = _checked_int(buf.dst_num_pages, "dst_num_pages", 1)
            if (page_bytes | src_offset | dst_base) % PAGE_ALIGN_BYTES:
                raise ValueError(
                    f"group {index}: offsets, bases and page sizes must be "
                    f"{PAGE_ALIGN_BYTES}-byte aligned"
                )
            end = dst_base + capacity * page_bytes
            for pointer in (dst_base, end - 1):
                if self._ops.pointer_device(pointer) != self.device:
                    raise ValueError(
                        f"group {index}: destination is not memory of device {self.device}"
                    )
            if any(
                dst_base < r.public.address + r.public.size and r.public.address < end
                for r in self._regions.values()
            ):
                raise ValueError(f"group {index}: destination is an imported region")
            if (dst_base, page_bytes) in dst_ends:
                raise ValueError(f"group {index}: two buffers share a destination")
            src_limit = min(
                src_limit, max(region.desc.size - src_offset, 0) // page_bytes
            )
            dst_limit = min(dst_limit, capacity)
            regions[region.public.region_id] = None
            srcs, dsts = by_size.setdefault(page_bytes, ([], []))
            srcs.append(region.public.address + src_offset)
            dsts.append(dst_base)
            dst_ends[dst_base, page_bytes] = end
        launches = [
            (size, np.asarray(s, np.int64), np.asarray(d, np.int64))
            for size, (s, d) in by_size.items()
        ]
        if _first_overlap(dst_ends) is not None:
            raise ValueError(f"group {index}: buffers overlap at the destination")
        buffer_set = _BufferSet(
            buffers, tuple(regions), launches, dst_ends, src_limit, dst_limit
        )
        if isinstance(buffers, tuple):  # a list could change under the cache
            if len(self._buffer_sets) >= _MAX_BUFFER_SETS:
                del self._buffer_sets[next(iter(self._buffer_sets))]
            self._buffer_sets[id(buffers)] = buffer_set
        return buffer_set

    def _check_destinations(self, plans) -> None:
        """Reject any destination byte two writers could share.

        Within the batch it is the caller's error (ValueError); against a batch
        still in flight it is PeerReadBusy. Writers can share a byte without a
        partial overlap only through the same (base, page size), where they
        conflict exactly on shared page ids.
        """
        overlap = {}  # (id(pages a), id(pages b)) -> pages intersect

        def shared(a, b) -> bool:
            key = (id(a), id(b))
            if key not in overlap:
                overlap[key] = np.intersect1d(a, b, assume_unique=True).size > 0
            return overlap[key]

        for i, a in enumerate(plans):
            for b in plans[i + 1 :]:
                same = a.buffer_set.dst_ends.keys() & b.buffer_set.dst_ends.keys()
                if same and shared(a.dst_sorted, b.dst_sorted):
                    raise ValueError("overlapping destination pages within the batch")
        self._check_new_geometry(plans)
        for batch_id, groups in list(self._busy_groups.items()):
            for busy_set, busy_pages in groups:
                for plan in plans:
                    if plan.buffer_set.dst_ends.keys().isdisjoint(busy_set.dst_ends):
                        continue
                    if not shared(plan.dst_sorted, busy_pages):
                        continue
                    batch = self._batches[batch_id]
                    self._poll(batch)  # it may have finished since anyone looked
                    if batch_id in self._busy_groups:
                        raise PeerReadBusy("destination pages of a batch in flight")
                    break
                else:
                    continue
                break

    def _check_new_geometry(self, plans, *, polled: bool = False) -> None:
        """Partial overlaps can only come from a (base, page size) not in use yet,
        or one in use whose range this batch extends to more pages."""
        ends: dict[tuple[int, int], int] = {}
        for plan in plans:
            for key, end in plan.buffer_set.dst_ends.items():
                ends[key] = max(end, ends.get(key, 0))
        fresh = {
            k: e
            for k, e in ends.items()
            if k not in self._busy_keys or e > self._busy_keys[k][0]
        }
        if not fresh:
            return
        merged = {k: v[0] for k, v in self._busy_keys.items()}
        for key, end in fresh.items():
            merged[key] = max(end, merged.get(key, 0))
        pair = _first_overlap(merged)
        if pair is None:
            return
        if pair[0] in fresh and pair[1] in fresh:
            raise ValueError("overlapping destination ranges within the batch")
        if not polled:  # batches may have finished since anyone looked
            for batch_id in list(self._busy_groups):
                self._poll(self._batches[batch_id])
            return self._check_new_geometry(plans, polled=True)
        raise PeerReadBusy("destination overlaps one of a batch in flight")

    def _admit(self, batch: _Batch) -> None:
        batch_id = batch.public.batch_id
        self._batches[batch_id] = batch
        for region in batch.regions:
            region.batches.add(batch_id)
        self._busy_groups[batch_id] = batch.destinations
        for buffer_set, _ in batch.destinations:
            for key, end in buffer_set.dst_ends.items():
                entry = self._busy_keys.setdefault(key, [end, 0])
                entry[0] = max(entry[0], end)
                entry[1] += 1

    def _retire(self, batch: _Batch) -> None:
        """Its destinations are no longer written: drop them from the in-flight view."""
        groups = self._busy_groups.pop(batch.public.batch_id, None)
        for buffer_set, _ in groups or ():
            for key in buffer_set.dst_ends:
                entry = self._busy_keys[key]
                entry[1] -= 1
                if entry[1] == 0:
                    del self._busy_keys[key]

    def _enqueue(self, batch: _Batch, plans, producer_stream) -> None:
        """Everything that may allocate or sync happens here, before any group is enqueued."""
        ops, stream = self._ops, self._stream
        meta, slices = _layout(plans)
        batch.recorded = [False] * len(plans)
        with ops.current(stream):
            try:
                for _ in plans:  # kept one by one so a failure destroys the rest
                    batch.events.append(ops.new_event(stream))
                if producer_stream is not None:
                    ops.wait_stream(stream, producer_stream)
                keepalive, views = ops.stage(stream, meta, slices)
                batch.keepalive.append(keepalive)
                for index, group_views in enumerate(views):
                    self._enqueue_group(stream, batch, index, group_views)
            except BaseException as error:
                code = self._drain(batch)
                if code != 0:
                    self._enter_fail_stop(
                        f"enqueue failed ({error!r}), drain returned {code}"
                    )
                    batch.state = ReadState.FAILED_UNCERTAIN
                    batch.error = f"enqueue failed without draining: {error!r}"
                    error.add_note("the read did not drain; the reader is fail-stopped")
                else:
                    for event in batch.events:
                        ops.destroy_event(event)
                raise

    def _enqueue_group(self, stream, batch: _Batch, index: int, views) -> None:
        """Launch one group on ``stream``, which must be current, and record its event,
        even when a launch raised, so the event covers whatever did get enqueued."""
        try:
            for view in views:
                self._ops.launch(view)
        finally:
            try:
                self._ops.record(batch.events[index], stream)
            except BaseException:
                batch.unrecorded = True  # launches no event covers: drain unknowable
                raise
            batch.recorded[index] = True

    def _drain(self, batch: _Batch) -> int:
        """Wait for everything this batch enqueued; 0 means it has stopped touching memory."""
        if batch.unrecorded:
            return -1
        recorded = [i for i, done in enumerate(batch.recorded) if done]
        return self._ops.sync_event(batch.events[recorded[-1]]) if recorded else 0

    def _poll(self, batch: _Batch) -> None:
        while batch.state == ReadState.IN_FLIGHT and batch.done < len(batch.events):
            if not batch.recorded[batch.done]:
                return  # not enqueued yet
            code = self._ops.query_event(batch.events[batch.done])
            if code == self._ops.not_ready:
                return
            if code == 0:
                batch.done += 1
                continue
            drained = self._drain(batch)
            batch.error = f"group {batch.done}: CUDA error {code}"
            if drained != 0 or self._ops.is_sticky(code):
                self._enter_fail_stop(batch.error)
            else:
                batch.state = ReadState.FAILED_DRAINED
                self._retire(batch)
            return
        if batch.state == ReadState.IN_FLIGHT:
            batch.state = ReadState.SUCCESS
            batch.keepalive.clear()
            self._retire(batch)

    def _enter_fail_stop(self, reason: str) -> None:
        self._fail_stop = reason
        for batch in self._batches.values():
            if batch.state == ReadState.IN_FLIGHT:
                batch.state = ReadState.FAILED_UNCERTAIN
                batch.error = batch.error or f"reader fail-stopped: {reason}"
