# SPDX-License-Identifier: Apache-2.0
"""Read KV pages out of a peer engine's GPU memory over NVLink (CUDA FABRIC handles).

The owner describes a FABRIC allocation with :func:`export_region`; its control
plane hands the descriptor bytes to a reader. :class:`KvPeerReader` maps it
read-only and gathers pages into the caller's own destination buffers with a
page-index kernel on a private stream.

Contract:

- A batch is checked as a whole before anything is enqueued: page bounds on both
  sides, alignment, and that every destination is memory of the reader's device
  and not an imported region. A rejected batch never touches memory.
- Until a batch is released, the caller keeps its source pages unchanged and its
  destination pages allocated and free of other writers, including other reads.
- Groups of a batch complete in order, one event each.
- A failed batch is FAILED_DRAINED when the reader's stream then synchronizes
  cleanly, FAILED_UNCERTAIN otherwise. An uncertain batch, including one whose
  enqueue failed without draining, fail-stops the reader and is never released,
  so its regions stay mapped for the life of the process.
- The owner may reuse an exported allocation only after every reader it went to
  has released its batches on it and unmapped it.
"""

from __future__ import annotations

import enum
import functools
import math
import threading
import time
import uuid
from typing import Any, Optional, Sequence

import msgspec
import numpy as np
import torch
from cuda.bindings import driver as drv

from sglang.kernels.ops.kvcache.peer_page_gather import gather_pages
from sglang.srt.utils.cuda_vmm_utils import (
    export_fabric_handle,
    get_gpu_fabric_clique,
    map_fabric_import,
    unmap_fabric_import,
)

# page_row.cuh copies 128-byte rows.
PAGE_ALIGN_BYTES = 128

_DESCRIPTOR_VERSION = 1
_FABRIC_HANDLE_BYTES = 64

# A GPU's fabric clique does not change while the process runs; NVML init is slow.
_fabric_clique = functools.cache(get_gpu_fabric_clique)


class ReadState(str, enum.Enum):
    IN_FLIGHT = "IN_FLIGHT"
    SUCCESS = "SUCCESS"
    FAILED_DRAINED = "FAILED_DRAINED"
    FAILED_UNCERTAIN = "FAILED_UNCERTAIN"


class PeerReadError(RuntimeError):
    pass


class PeerReadBusy(PeerReadError):
    """The batch or region is still in use."""


class PeerRegionDescriptor(msgspec.Struct, frozen=True):
    """What a reader needs to map one exported allocation. Opaque to the control plane."""

    version: int
    size: int
    fabric_handle: bytes
    # (cluster_uuid, clique_id) of the owner GPU; None when NVML could not tell.
    fabric_clique: Optional[tuple[bytes, int]] = None

    def to_bytes(self) -> bytes:
        return msgspec.msgpack.encode(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> PeerRegionDescriptor:
        try:
            return msgspec.msgpack.decode(data, type=cls)
        except msgspec.DecodeError as error:
            raise ValueError(f"malformed peer region descriptor: {error}") from None


class PeerRegion(msgspec.Struct, frozen=True):
    """Reader-side handle of an imported region, valid until the reader unmaps it."""

    region_id: str
    size: int
    address: int


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

    ``src_pages`` / ``dst_pages`` are equal-length integer page ids on the host
    (numpy array, CPU tensor or sequence); every buffer copies the same pages.
    """

    buffers: Sequence[PeerBuffer]
    src_pages: Any
    dst_pages: Any


class ReadStatus(msgspec.Struct, frozen=True):
    state: ReadState
    groups_done: int
    num_groups: int
    error: str = ""


class ReadBatch(msgspec.Struct, frozen=True):
    batch_id: str


class _GroupPlan(msgspec.Struct, frozen=True):
    regions: frozenset
    src_pages: Any  # int64 host arrays, shared by the group's launches
    dst_pages: Any
    # One kernel launch per page size: (page_bytes, src bases, dst bases).
    launches: list


class _Batch(msgspec.Struct):
    regions: frozenset
    num_groups: int
    events: list = []
    groups_done: int = 0
    state: ReadState = ReadState.IN_FLIGHT
    error: str = ""
    # Device launch tables; kept until no kernel of the batch can still read them.
    tables: Any = None


def export_region(handle, size: int, *, device: int) -> PeerRegionDescriptor:
    """Describe the FABRIC allocation ``handle`` of ``size`` bytes on ``device`` for readers.

    The allocation stays the caller's: it must not be freed or reused while any
    reader still maps it.
    """
    return PeerRegionDescriptor(
        version=_DESCRIPTOR_VERSION,
        size=size,
        fabric_handle=export_fabric_handle(handle),
        fabric_clique=_fabric_clique(device),
    )


def _checked_int(value, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not minimum <= value < 2**63
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**63)")
    return int(value)


def _as_pages(value, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            raise ValueError(f"{name} must be on the host; it is checked before launch")
        value = value.numpy()
    pages = np.asarray(value)
    if pages.ndim != 1 or pages.size == 0 or pages.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a non-empty 1-D integer array")
    pages = pages.astype(np.int64, copy=False)
    if int(pages.min()) < 0:
        raise ValueError(f"{name} has a negative page id")
    return pages


def _locked(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return call


class KvPeerReader:
    """Imports peer regions and reads pages out of them on one device.

    ``num_blocks`` is the gather's CTA budget: more blocks copy faster but take
    SMs from whatever else runs on the device.
    """

    def __init__(self, device: int, *, num_blocks: int = 64, _ops=None):
        self.device = _checked_int(device, "device")
        _checked_int(num_blocks, "num_blocks", 1)
        self._ops = _ops if _ops is not None else _CudaOps(self.device, num_blocks)
        self._stream = self._ops.new_stream()
        self._ops.warmup(self._stream)
        self._clique = self._ops.fabric_clique()
        self._lock = threading.Lock()
        self._regions: set[PeerRegion] = set()
        self._batches: dict[ReadBatch, _Batch] = {}
        self._fail_stop = ""

    @_locked
    def import_region(self, desc: PeerRegionDescriptor) -> PeerRegion:
        self._check_alive()
        if not isinstance(desc, PeerRegionDescriptor):
            raise ValueError("expected a PeerRegionDescriptor")
        if desc.version != _DESCRIPTOR_VERSION:
            raise ValueError(f"unsupported descriptor version {desc.version}")
        if len(desc.fabric_handle) != _FABRIC_HANDLE_BYTES:
            raise ValueError("FABRIC handle must be 64 bytes")
        size = _checked_int(desc.size, "region size", 1)
        if (
            desc.fabric_clique is not None
            and self._clique is not None
            and tuple(desc.fabric_clique) != tuple(self._clique)
        ):
            raise PeerReadError("owner GPU is in a different NVLink fabric clique")
        address = self._ops.map_import(desc.fabric_handle, size)
        region = PeerRegion(region_id=uuid.uuid4().hex, size=size, address=address)
        self._regions.add(region)
        return region

    @_locked
    def unmap(self, region: PeerRegion) -> None:
        """Unmap after every batch reading the region is released."""
        if region not in self._regions:
            raise ValueError("region is not mapped by this reader")
        if any(region in batch.regions for batch in self._batches.values()):
            raise PeerReadBusy("region has unreleased batches")
        self._ops.unmap_import(region.address, region.size)
        self._regions.remove(region)

    @_locked
    def read(self, groups: Sequence[ReadGroup], *, producer_stream=None) -> ReadBatch:
        """Check ``groups`` and enqueue them in order; return before they complete.

        ``producer_stream`` (a stream of this device) is waited on first, for
        destinations it is still writing.
        """
        self._check_alive()
        plans = [self._plan(group, index) for index, group in enumerate(groups)]
        if not plans:
            raise ValueError("empty read batch")
        handle = ReadBatch(batch_id=uuid.uuid4().hex)
        batch = _Batch(
            regions=frozenset().union(*(plan.regions for plan in plans)),
            num_groups=len(plans),
        )
        self._batches[handle] = batch  # holds its regions mapped until release
        try:
            self._enqueue(batch, plans, producer_stream)
        except BaseException as error:
            if self._drained():
                del self._batches[handle]
            else:
                self._enter_fail_stop(f"enqueue failed and did not drain: {error!r}")
                error.add_note("the read did not drain; the reader is fail-stopped")
            raise
        return handle

    @_locked
    def status(self, batch: ReadBatch) -> ReadStatus:
        item = self._batch(batch)
        self._poll(item)
        return ReadStatus(
            state=item.state,
            groups_done=item.groups_done,
            num_groups=item.num_groups,
            error=item.error,
        )

    def wait(self, batch: ReadBatch, timeout: Optional[float] = None) -> ReadStatus:
        """Block until the batch leaves IN_FLIGHT. On timeout raise TimeoutError and keep everything."""
        if timeout is not None and not 0 <= timeout < math.inf:
            raise ValueError("timeout must be finite and non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        pause = 20e-6
        while (status := self.status(batch)).state == ReadState.IN_FLIGHT:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("read still in flight; nothing may be released")
            time.sleep(pause)
            pause = min(pause * 2, 1e-3)
        return status

    @_locked
    def release(self, batch: ReadBatch) -> None:
        item = self._batch(batch)
        self._poll(item)
        if item.state not in (ReadState.SUCCESS, ReadState.FAILED_DRAINED):
            raise PeerReadBusy(f"batch is {item.state.value}; it cannot be released")
        del self._batches[batch]

    def _check_alive(self) -> None:
        if self._fail_stop:
            raise PeerReadError(f"reader fail-stopped: {self._fail_stop}")

    def _batch(self, batch) -> _Batch:
        item = self._batches.get(batch) if isinstance(batch, ReadBatch) else None
        if item is None:
            raise ValueError("unknown or released batch")
        return item

    def _plan(self, group: ReadGroup, index: int) -> _GroupPlan:
        if not isinstance(group, ReadGroup) or not group.buffers:
            raise ValueError(f"group {index}: expected a ReadGroup with buffers")
        src_pages = _as_pages(group.src_pages, "src_pages")
        dst_pages = _as_pages(group.dst_pages, "dst_pages")
        if src_pages.size != dst_pages.size:
            raise ValueError(f"group {index}: src_pages and dst_pages differ in length")
        src_end, dst_end = int(src_pages.max()) + 1, int(dst_pages.max()) + 1
        bases: dict[int, tuple[list, list]] = {}
        for buf in group.buffers:
            page_bytes = self._check_buffer(buf, index, src_end, dst_end)
            src_bases, dst_bases = bases.setdefault(page_bytes, ([], []))
            src_bases.append(buf.src_region.address + buf.src_offset)
            dst_bases.append(buf.dst_base)
        return _GroupPlan(
            regions=frozenset(buf.src_region for buf in group.buffers),
            src_pages=src_pages,
            dst_pages=dst_pages,
            launches=[
                (page_bytes, np.array(src, np.int64), np.array(dst, np.int64))
                for page_bytes, (src, dst) in bases.items()
            ],
        )

    def _check_buffer(self, buf, index: int, src_end: int, dst_end: int) -> int:
        """Reject a buffer the gather could fault on or write outside of; return its page size."""
        if not isinstance(buf, PeerBuffer):
            raise ValueError(f"group {index}: expected PeerBuffer entries")
        if buf.src_region not in self._regions:
            raise ValueError(
                f"group {index}: source region is not mapped by this reader"
            )
        src_offset = _checked_int(buf.src_offset, "src_offset")
        dst_base = _checked_int(buf.dst_base, "dst_base", 1)
        page_bytes = _checked_int(buf.page_bytes, "page_bytes", 1)
        capacity = _checked_int(buf.dst_num_pages, "dst_num_pages", 1)
        if (src_offset | dst_base | page_bytes) % PAGE_ALIGN_BYTES:
            raise ValueError(
                f"group {index}: offsets, bases and page sizes must be "
                f"{PAGE_ALIGN_BYTES}-byte aligned"
            )
        if src_offset + src_end * page_bytes > buf.src_region.size:
            raise ValueError(f"group {index}: source page beyond its region")
        if dst_end > capacity:
            raise ValueError(f"group {index}: destination page beyond the buffer")
        dst_limit = dst_base + capacity * page_bytes
        for pointer in (dst_base, dst_limit - 1):
            if self._ops.pointer_device(pointer) != self.device:
                raise ValueError(
                    f"group {index}: destination is not memory of device {self.device}"
                )
        # Imported regions are mapped read-only: a write there faults.
        if any(
            dst_base < region.address + region.size and region.address < dst_limit
            for region in self._regions
        ):
            raise ValueError(f"group {index}: destination is an imported region")
        return page_bytes

    def _enqueue(self, batch: _Batch, plans: list[_GroupPlan], producer_stream) -> None:
        ops, stream = self._ops, self._stream
        arrays = []  # per group: its page lists, then each launch's base tables
        for plan in plans:
            arrays += [plan.src_pages, plan.dst_pages]
            for _, src_bases, dst_bases in plan.launches:
                arrays += [src_bases, dst_bases]
        with ops.current(stream):
            if producer_stream is not None:
                stream.wait_stream(producer_stream)
            batch.tables = ops.upload(arrays)
            tables = iter(batch.tables)
            for plan in plans:
                src_pages, dst_pages = next(tables), next(tables)
                for page_bytes, _, _ in plan.launches:
                    ops.gather(
                        src_bases=next(tables),
                        dst_bases=next(tables),
                        src_pages=src_pages,
                        dst_pages=dst_pages,
                        page_bytes=page_bytes,
                    )
                batch.events.append(ops.record_event(stream))

    def _drained(self) -> bool:
        """Whether nothing this reader enqueued can still touch memory."""
        try:
            self._stream.synchronize()
        except RuntimeError:
            return False
        return True

    def _poll(self, batch: _Batch) -> None:
        if batch.state != ReadState.IN_FLIGHT:
            return
        while batch.groups_done < batch.num_groups:
            try:
                if not batch.events[batch.groups_done].query():
                    return
            except RuntimeError as error:  # CUDA reports a kernel failure here
                batch.error = f"group {batch.groups_done}: {error}"
                if self._drained():
                    batch.state = ReadState.FAILED_DRAINED
                else:
                    self._enter_fail_stop(batch.error)
                return
            batch.groups_done += 1
        batch.state = ReadState.SUCCESS
        batch.tables = None

    def _enter_fail_stop(self, reason: str) -> None:
        self._fail_stop = reason
        for batch in self._batches.values():
            if batch.state == ReadState.IN_FLIGHT:
                batch.state = ReadState.FAILED_UNCERTAIN
                batch.error = batch.error or f"reader fail-stopped: {reason}"


class _CudaOps:
    """The CUDA side of a reader; the contract tests substitute a fake."""

    def __init__(self, device: int, num_blocks: int):
        self._device = torch.device("cuda", device)
        self._num_blocks = num_blocks

    def fabric_clique(self) -> Optional[tuple[bytes, int]]:
        return _fabric_clique(self._device.index)

    def map_import(self, fabric_handle: bytes, size: int) -> int:
        return map_fabric_import(
            fabric_handle, size, self._device.index, read_only=True
        )

    def unmap_import(self, address: int, size: int) -> None:
        unmap_fabric_import(address, size)

    def pointer_device(self, pointer: int) -> Optional[int]:
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

    def new_stream(self) -> torch.cuda.Stream:
        return torch.cuda.Stream(device=self._device)

    def current(self, stream: torch.cuda.Stream):
        return torch.cuda.stream(stream)

    def warmup(self, stream: torch.cuda.Stream) -> None:
        """Compile and load the gather now rather than in the first read."""
        with torch.cuda.stream(stream):
            buf = torch.zeros(
                2 * PAGE_ALIGN_BYTES, dtype=torch.uint8, device=self._device
            )
            bases = torch.tensor(
                [buf.data_ptr(), buf.data_ptr() + PAGE_ALIGN_BYTES],
                dtype=torch.int64,
                device=self._device,
            )
            page = torch.zeros(1, dtype=torch.int64, device=self._device)
            self.gather(
                src_bases=bases[:1],
                dst_bases=bases[1:],
                src_pages=page,
                dst_pages=page,
                page_bytes=PAGE_ALIGN_BYTES,
            )
        stream.synchronize()

    def upload(self, arrays: list[np.ndarray]) -> tuple[torch.Tensor, ...]:
        """Copy int64 host arrays to the device in one transfer, on the current stream."""
        host = torch.from_numpy(np.concatenate(arrays)).pin_memory()
        device = host.to(self._device, non_blocking=True)
        return device.split([array.size for array in arrays])

    def gather(self, *, src_bases, dst_bases, src_pages, dst_pages, page_bytes):
        gather_pages(
            src_ptrs=src_bases.view(torch.uint64),
            dst_ptrs=dst_bases.view(torch.uint64),
            src_pages=src_pages,
            dst_pages=dst_pages,
            page_bytes=page_bytes,
            num_blocks=self._num_blocks,
        )

    def record_event(self, stream: torch.cuda.Stream) -> torch.cuda.Event:
        event = torch.cuda.Event()
        event.record(stream)
        return event
