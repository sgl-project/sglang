from __future__ import annotations

import ctypes
import enum
import errno
import functools
import glob
import logging
import mmap
import os
import re
import time
from contextlib import nullcontext
from typing import Optional, Protocol

import msgspec
import numpy as np
import torch

from sglang.kernels.jit.utils.cuda.vmm import VMMHostMemory, is_vmm_available
from sglang.kernels.ops.attention.dsv4.torch_quant import FP8_BLOCK_SIZE
from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    find_loaded_library,
)
from sglang.srt.distributed.fd_exchange import exchange_fd
from sglang.srt.distributed.symmetric_memory import symmetric_context
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_model
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)


_is_cuda = is_cuda()
_is_hip = is_hip()


class EngramTableLayout(str, enum.Enum):
    """Which rows a rank can read. FULL_SHARED: the whole table, in place (one
    host copy every rank maps, or device column shards in symmetric memory), so
    the lookup needs no collective. ROW_SHARDED: only the rank's own rows; the
    lookup zero-fills the rest and all-reduces."""

    FULL_SHARED = "full_shared"
    ROW_SHARDED = "row_sharded"

    @staticmethod
    def parse(use_host: bool, value: str) -> EngramTableLayout:
        if not value:
            value = "full_shared" if use_host else "row_sharded"
        return EngramTableLayout(value)


def column_shard(
    bounds: tuple[int, ...], tp_size: int, tp_rank: int
) -> tuple[int, int]:
    """This rank's [start, end) rows: on the hash-column boundaries ``bounds``
    when the columns divide evenly, so a rank owns whole per-head subtables,
    else the equal-row split. Exactly one rank owns each row either way."""
    num_cols, num_rows = len(bounds) - 1, bounds[-1]
    if num_cols % tp_size == 0:
        per_rank = num_cols // tp_size
        return bounds[per_rank * tp_rank], bounds[per_rank * (tp_rank + 1)]
    return num_rows * tp_rank // tp_size, num_rows * (tp_rank + 1) // tp_size


class EngramGatherView(msgspec.Struct, frozen=True):
    """How a kernel reads every row from this rank without a collective: shard
    ``r`` holds rows from ``starts[r]`` at ``w_ptrs[r]`` (fp8) / ``s_ptrs[r]``
    (e8m0), and owns the columns ``r * num_cols / world`` onward."""

    w_ptrs: tuple[int, ...]
    s_ptrs: tuple[int, ...]
    starts: tuple[int, ...]


def _split_views(
    raw: torch.Tensor, rows: int, dim: int, stride_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``raw`` as [stride_rows * dim] fp8 rows, then [stride_rows * dim / 32]
    e8m0 scales; the first ``rows`` of each."""
    sdim = dim // FP8_BLOCK_SIZE
    weight = raw[: rows * dim].view(torch.float8_e4m3fn).view(rows, dim)
    base = stride_rows * dim
    scale = raw[base : base + rows * sdim].view(torch.float8_e8m0fnu)
    return weight, scale.view(rows, sdim)


def _table_nbytes(rows: int, dim: int) -> int:
    return rows * (dim + dim // FP8_BLOCK_SIZE)


_THP_DIR = "/sys/kernel/mm/transparent_hugepage"


def _thp_mode(knob: str) -> str:
    """Active mode of a transparent_hugepage sysfs knob ("" if unreadable)."""
    try:
        with open(f"{_THP_DIR}/{knob}") as f:
            m = re.search(r"\[(\w+)\]", f.read())
        return m.group(1) if m else ""
    except OSError:
        return ""


def _huge_pages_backing(addr: int) -> tuple[int, int]:
    """(mapped_kB, huge_kB) of the VMA holding addr, from /proc/self/smaps.
    The only evidence that the kernel really handed out huge pages."""
    mapped = huge = 0
    inside = False
    try:
        with open("/proc/self/smaps") as f:
            for line in f:
                m = re.match(r"^([0-9a-f]+)-([0-9a-f]+) ", line)
                if m:
                    if inside:
                        break
                    inside = int(m.group(1), 16) <= addr < int(m.group(2), 16)
                elif inside:
                    key, _, rest = line.partition(":")
                    if key == "Rss":
                        mapped = int(rest.split()[0])
                    elif key in ("AnonHugePages", "ShmemPmdMapped", "FilePmdMapped"):
                        huge += int(rest.split()[0])
    except OSError:
        pass
    return mapped, huge


_page_cache_dropped = False


def _drop_checkpoint_page_cache() -> tuple[int, int]:
    """posix_fadvise(DONTNEED) on the checkpoint files; returns (files, bytes)."""
    try:
        model_path = get_model().model_path
    except (ValueError, AttributeError):
        # No published runtime context (unit tests, offline tools): nothing to drop.
        return 0, 0
    files, nbytes = 0, 0
    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        try:
            fd = os.open(f, os.O_RDONLY)
        except OSError:
            continue
        try:
            nbytes += os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            files += 1
        finally:
            os.close(fd)
    return files, nbytes


def _drop_page_cache_once(reason: str) -> None:
    global _page_cache_dropped
    if _page_cache_dropped:
        return
    _page_cache_dropped = True
    files, nbytes = _drop_checkpoint_page_cache()
    logger.info(
        "engram host table: dropped the page cache of %d checkpoint files (%.0f GiB) %s",
        files,
        nbytes / 2**30,
        reason,
    )


@functools.cache
def _hip_runtime() -> ctypes.CDLL:
    """
    torch.cuda.cudart() does not expose hipHostGetDevicePointer, so call it
    through ctypes to map a registered host address to its device address.
    """
    path = find_loaded_library("libamdhip64")
    if path is None:
        raise RuntimeError("libamdhip64 is not loaded in the current process")
    lib = ctypes.CDLL(path)
    lib.hipHostGetDevicePointer.restype = ctypes.c_int
    lib.hipHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    return lib


def _registered_device_ptr(host_ptr: int) -> int:
    """
    Address kernels must use for registered host memory.
    UVA makes it the host address on CUDA; HIP may map it elsewhere.
    """
    if not _is_hip:
        return host_ptr
    device_ptr = ctypes.c_void_p()
    err = _hip_runtime().hipHostGetDevicePointer(ctypes.byref(device_ptr), host_ptr, 0)
    if err != 0 or not device_ptr.value:
        raise RuntimeError(f"hipHostGetDevicePointer failed: {err}")
    return device_ptr.value


class _HostTablePosix:
    """Host memory for one engram table: a memfd every rank maps (full_shared) or
    a private anonymous mapping of the rank's rows (row_sharded).

    Lives for the whole process: the mapping, the memfd and the cudaHostRegister
    pin are never released because the table is read by every forward.
    """

    def __init__(
        self,
        layout: EngramTableLayout,
        rows: int,
        row_start: int,
        dim: int,
        name: str,
        group,
    ):
        self.layout = layout
        self.row_start = row_start
        # mmap requires storage even for an empty shard.
        nbytes = max(1, _table_nbytes(rows, dim))
        self.nbytes = nbytes
        self.group = group
        self.dirty = False
        if layout == EngramTableLayout.FULL_SHARED:
            self.fd = self._open_shared_fd(nbytes, name)
            self.mm = mmap.mmap(
                self.fd,
                nbytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        else:
            self.fd = None
            self.mm = mmap.mmap(
                -1,
                nbytes,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        # Advisory before the first touch: pages are allocated huge at fault time.
        self.mm.madvise(mmap.MADV_HUGEPAGE)
        self.bytes = torch.frombuffer(self.mm, dtype=torch.uint8)
        if layout == EngramTableLayout.ROW_SHARDED:
            # Cached checkpoint pages, left by a previous server or by the loader,
            # make the 512 MiB huge-page faults fall back, so empty them first.
            _drop_page_cache_once("before pre-faulting the per-rank shard")
            np.frombuffer(self.mm, dtype=np.uint8)[:: mmap.PAGESIZE] = 0
        if layout == EngramTableLayout.FULL_SHARED:
            # Every rank holds the fd before rank 0 continues; the /proc path only
            # resolves while rank 0 keeps its descriptor.
            group.barrier()
        err = torch.cuda.cudart().cudaHostRegister(self.bytes.data_ptr(), nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({nbytes} bytes) failed: {err}")
        device_ptr = _registered_device_ptr(self.bytes.data_ptr())
        self.weight, self.scale = _split_views(self.bytes, rows, dim, rows)
        self.ptrs = (device_ptr, device_ptr + rows * dim)
        self._view = _host_gather_view(layout, self.ptrs, group)

    def gather_view(self) -> Optional[EngramGatherView]:
        return self._view

    def _open_shared_fd(self, nbytes: int, name: str) -> int:
        owner = None
        if self.group.rank_in_group == 0:
            fd = os.memfd_create(name, 0)
            os.ftruncate(fd, nbytes)
            owner = (os.getpid(), fd)
        pid, owner_fd = self.group.broadcast_object(owner, src=0)
        if self.group.rank_in_group == 0:
            return fd
        try:
            return os.open(f"/proc/{pid}/fd/{owner_fd}", os.O_RDWR)
        except OSError as e:
            raise RuntimeError(
                "engram host table: cannot open rank 0's memfd through /proc; the "
                "TP ranks must share a PID namespace"
            ) from e

    def _collapse(self, tries: int = 3) -> None:
        """Synchronously fold whatever is still on base pages into huge pages.
        Anonymous memory only; shmem obeys shmem_enabled and refuses."""
        MADV_COLLAPSE = 25  # Linux >= 6.1; not in Python's mmap module
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        for attempt in range(tries):
            rc = libc.madvise(
                ctypes.c_void_p(self.bytes.data_ptr()),
                ctypes.c_size_t(self.nbytes),
                MADV_COLLAPSE,
            )
            if rc == 0:
                return
            err = ctypes.get_errno()
            if (
                err != errno.EAGAIN or attempt == tries - 1
            ):  # EAGAIN is the only one worth retrying
                logger.info("engram host table: MADV_COLLAPSE errno %d", err)
                return
            time.sleep(1.0)

    def mark_loaded(self):
        self.dirty = True

    def finish_load(self, label: str):
        if not self.dirty:
            return
        self.dirty = False
        if self.layout == EngramTableLayout.FULL_SHARED:
            self.group.barrier()
        mapped_kb, huge_kb = _huge_pages_backing(self.bytes.data_ptr())
        if self.layout == EngramTableLayout.ROW_SHARDED and huge_kb < mapped_kb * 0.98:
            # The loader's own reads refilled the page cache; empty it again so the
            # collapse can find contiguous memory.
            _drop_checkpoint_page_cache()
            self._collapse()
            mapped_kb, huge_kb = _huge_pages_backing(self.bytes.data_ptr())
        pct = 100.0 * huge_kb / mapped_kb if mapped_kb else 0.0
        msg = (
            f"engram host table {label}: layout={self.layout}, "
            f"{mapped_kb / 2**10:.0f} MiB resident, {huge_kb / 2**10:.0f} MiB in huge pages "
            f"({pct:.0f}%)"
        )
        if huge_kb == 0:
            knob = (
                "shmem_enabled"
                if self.layout == EngramTableLayout.FULL_SHARED
                else "enabled"
            )
            logger.warning(
                "%s. No huge pages: expect ~10x slower lookups (one TLB miss per row); "
                "transparent_hugepage/%s is '%s'",
                msg,
                knob,
                _thp_mode(knob) or "unreadable",
            )
        else:
            logger.info(msg)


class _HostTableVMM:
    """Host memory for one engram table through CUDA VMM: pinned, 2MB-granular,
    and for full_shared exported by fd so every rank maps one physical copy."""

    def __init__(
        self,
        layout: EngramTableLayout,
        rows: int,
        row_start: int,
        dim: int,
        name: str,
        group,
    ):
        self.group = group
        self.layout = layout
        self.row_start = row_start
        nbytes = max(1, _table_nbytes(rows, dim))
        if layout == EngramTableLayout.FULL_SHARED:
            # Both sides close their fd once the mapping exists: SCM_RIGHTS copies
            # are independent descriptors, so there is no keep-alive requirement.
            if group.rank_in_group == 0:
                self.vmm = VMMHostMemory(nbytes)
                size = group.broadcast_object(self.vmm.size(), src=0)
                fd = exchange_fd(group, self.vmm.export_fd(), name=name)
            else:
                size = group.broadcast_object(None, src=0)
                fd = exchange_fd(group, None, name=name)
                self.vmm = VMMHostMemory(size, fd)
            os.close(fd)
        else:
            self.vmm = VMMHostMemory(nbytes)
        # The mapping is host- and device-accessible at one address.
        device_ptr = self.vmm.ptr()
        raw = torch.frombuffer(
            ctypes.cast(device_ptr, ctypes.POINTER(ctypes.c_uint8 * nbytes)).contents,
            dtype=torch.uint8,
        )
        self.weight, self.scale = _split_views(raw, rows, dim, rows)
        self.ptrs = (device_ptr, device_ptr + rows * dim)
        self._view = _host_gather_view(layout, self.ptrs, group)
        self.dirty = False

    def gather_view(self) -> Optional[EngramGatherView]:
        return self._view

    def mark_loaded(self):
        self.dirty = True

    def finish_load(self, label: str):
        if not self.dirty:
            return
        self.dirty = False
        self.group.barrier()


def _host_gather_view(
    layout: EngramTableLayout, ptrs: tuple[int, int], group
) -> Optional[EngramGatherView]:
    """A host table every rank maps whole is a world of one; a private shard is
    readable only by its own rank."""
    if layout == EngramTableLayout.FULL_SHARED or group.world_size == 1:
        return EngramGatherView((ptrs[0],), (ptrs[1],), (0,))
    return None


class _DeviceTable:
    """HBM for the rank's column shard. full_shared puts the shard in symmetric
    memory so peers read it in place; row_sharded keeps it private."""

    def __init__(
        self,
        layout: EngramTableLayout,
        spans: list[tuple[int, int]],
        rank: int,
        dim: int,
        group,
    ):
        self.layout = layout
        self.group = group
        self.row_start, row_end = spans[rank]
        rows = row_end - self.row_start
        self._spans = spans
        self._symmetric = layout == EngramTableLayout.FULL_SHARED and len(spans) > 1
        # The symmetric pool wants the same shapes on every rank, and the column
        # shards are ragged, so every rank lays out the widest one.
        stride_rows = (
            max(end - start for start, end in spans) if self._symmetric else rows
        )
        device = torch.empty(0).device
        ctx = symmetric_context(device) if self._symmetric else nullcontext()
        with ctx:
            self._buf = torch.empty(_table_nbytes(stride_rows, dim), dtype=torch.uint8)
        self.weight, self.scale = _split_views(self._buf, rows, dim, stride_rows)
        self._scale_offset = stride_rows * dim
        self.ptrs = (self._buf.data_ptr(), self._buf.data_ptr() + self._scale_offset)
        self._view: Optional[EngramGatherView] = None
        if len(spans) == 1:
            self._view = EngramGatherView((self.ptrs[0],), (self.ptrs[1],), (0,))

    def gather_view(self) -> Optional[EngramGatherView]:
        return self._view

    def mark_loaded(self):
        pass

    def finish_load(self, label: str):
        """Collective for a symmetric shard: exchanges the peers' addresses, once,
        before any CUDA graph capture."""
        if not self._symmetric or self._view is not None:
            return
        import torch.distributed._symmetric_memory as symm_mem

        handle = symm_mem.rendezvous(self._buf, self.group.device_group)
        self._view = EngramGatherView(
            tuple(handle.buffer_ptrs),
            tuple(p + self._scale_offset for p in handle.buffer_ptrs),
            tuple(start for start, _ in self._spans),
        )


class Table(Protocol):
    layout: EngramTableLayout
    row_start: int  # the first global row this rank stores
    weight: torch.Tensor  # [rows, dim] fp8 e4m3, where the loader writes
    scale: torch.Tensor  # [rows, dim / 32] e8m0
    ptrs: tuple[int, int]  # device addresses of weight and scale

    def gather_view(self) -> Optional[EngramGatherView]:
        """Every shard's addresses when this rank can read the whole table in
        place, else None; final after `finish_load`."""
        ...

    def mark_loaded(self): ...

    def finish_load(self, label: str):
        """Collective: every rank calls it once its rows are loaded."""
        ...


def _check_device_full_shared(bounds: tuple[int, ...], dim: int, tp_size: int):
    num_cols = len(bounds) - 1
    reasons = [
        (not _is_cuda, "it needs CUDA symmetric memory"),
        (
            num_cols % tp_size != 0,
            f"{num_cols} hash columns do not split over TP {tp_size}",
        ),
        (
            num_cols != 24 or dim != 256,
            "the gather kernel is built for 24 columns of 256",
        ),
        (bounds[-1] >= 2**31, "row ids must fit int32"),
    ]
    for failed, reason in reasons:
        if failed:
            raise ValueError(f"engram device table full_shared: {reason}")


def create_engram_table(
    *,
    use_host: bool,
    layout: EngramTableLayout,
    bounds: tuple[int, ...],
    dim: int,
    name: str,
    group,
) -> Table:
    """One layer's table; ``bounds`` are the prefix sums of its hash columns."""
    tp_size, rank = group.world_size, group.rank_in_group
    spans = [column_shard(bounds, tp_size, r) for r in range(tp_size)]
    if not use_host:
        if layout == EngramTableLayout.FULL_SHARED and tp_size > 1:
            _check_device_full_shared(bounds, dim, tp_size)
        return _DeviceTable(layout, spans, rank, dim, group)
    if layout == EngramTableLayout.FULL_SHARED:
        rows, row_start = bounds[-1], 0
    else:
        row_start, row_end = spans[rank]
        rows = row_end - row_start
    use_vmm = (
        envs.SGLANG_OPT_DSV41_ENGRAM_HOST_TABLE_VMM.get()
        and _is_cuda
        and is_vmm_available()
    )
    cls = _HostTableVMM if use_vmm else _HostTablePosix
    return cls(layout, rows, row_start, dim, name, group)
