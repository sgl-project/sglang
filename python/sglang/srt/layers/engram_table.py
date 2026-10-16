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
    """What a rank stores and who reads it.

    PRIVATE: its own shard, readable only by itself; the lookup zero-fills the
    other rows and all-reduces. SHARED: its own shard, readable in place by every
    rank (symmetric memory, or one host mapping holding all shards), so the
    lookup needs no collective. REPLICATED: the whole table, read locally."""

    PRIVATE = "private"
    SHARED = "shared"
    REPLICATED = "replicated"

    @staticmethod
    def parse(value: str) -> Optional[EngramTableLayout]:
        """None for an empty value: create_engram_table picks the layout."""
        return EngramTableLayout(value) if value else None


class EngramGatherView(msgspec.Struct, frozen=True):
    """How a kernel reads every row from this rank without a collective: shard
    ``r`` holds rows from ``starts[r]`` at ``w_ptrs[r]`` (fp8) / ``s_ptrs[r]``
    (e8m0), and owns the columns ``r * num_cols / world`` onward."""

    w_ptrs: tuple[int, ...]
    s_ptrs: tuple[int, ...]
    starts: tuple[int, ...]


class EngramStorage(msgspec.Struct, frozen=True, kw_only=True):
    """The rows this rank stores and how a kernel reaches the rest: everything the
    loader and the lookup read, nothing about how the memory was obtained."""

    weight: torch.Tensor  # [num_rows, dim] fp8 e4m3
    scale: torch.Tensor  # [num_rows, dim / 32] e8m0
    row_start: int  # the first global row of weight / scale
    ptrs: tuple[int, int]  # device addresses of weight / scale
    load_rows: tuple[int, int]  # the global [start, end) the loader writes
    # Every shard's addresses when the whole table is readable in place, else
    # None; the rows behind them are valid once every rank's finish_load returns.
    view: Optional[EngramGatherView] = None

    @property
    def num_rows(self) -> int:
        return self.weight.size(0)


def _make_storage(
    raw: torch.Tensor,
    dim: int,
    *,
    row_start: int,
    num_rows: int,
    load_rows: tuple[int, int],
    is_full: bool,
    device_ptr: Optional[int] = None,
    device_peer_pairs: Optional[tuple[list[int], list[int]]] = None,
    num_padded_rows: Optional[int] = None,
) -> EngramStorage:
    """``raw`` (device address ``device_ptr``) as [num_padded_rows * dim] fp8
    values, then [num_padded_rows * dim / 32] e8m0 scales, the first ``num_rows``
    of each. A buffer holding every row is readable in place as a world of one;
    ``device_peer_pairs`` (every rank's buffer address and first row) makes the
    peers' shards readable, each laid out alike so its scales sit at the same
    offset."""
    if device_ptr is None:
        device_ptr = raw.data_ptr()
    num_padded_rows = num_rows if num_padded_rows is None else num_padded_rows
    sdim = dim // FP8_BLOCK_SIZE
    weight = raw[: num_rows * dim].view(torch.float8_e4m3fn).view(num_rows, dim)
    base = num_padded_rows * dim
    scale = raw[base : base + num_rows * sdim].view(torch.float8_e8m0fnu)
    ptrs = (device_ptr, device_ptr + base)
    if is_full:
        assert device_peer_pairs is None
        view = EngramGatherView((ptrs[0],), (ptrs[1],), (0,))
    elif device_peer_pairs is not None:
        w_ptrs, starts = device_peer_pairs
        s_ptrs = [p + base for p in w_ptrs]
        view = EngramGatherView(tuple(w_ptrs), tuple(s_ptrs), tuple(starts))
    else:
        view = None
    return EngramStorage(
        weight=weight,
        scale=scale.view(num_rows, sdim),
        row_start=row_start,
        ptrs=ptrs,
        load_rows=load_rows,
        view=view,
    )


def _shard_hash_columns_at(bounds: tuple[int, ...], *, rank: int, world: int):
    """This rank's [start, end) rows on the hash-column boundaries ``bounds``:
    the columns split evenly over the ranks, each owning whole per-head
    subtables."""
    num_hash_columns = len(bounds) - 1
    assert num_hash_columns % world == 0
    per_rank = num_hash_columns // world
    return bounds[per_rank * rank], bounds[per_rank * (rank + 1)]


def _shard_hash_columns(layout: EngramTableLayout, bounds: tuple[int, ...], group):
    if layout == EngramTableLayout.REPLICATED:
        return 0, bounds[-1]
    assert layout in (EngramTableLayout.PRIVATE, EngramTableLayout.SHARED)
    rank, world = group.rank_in_group, group.world_size
    return _shard_hash_columns_at(bounds, rank=rank, world=world)


def _table_nbytes(rows: int, dim: int) -> int:
    assert dim % FP8_BLOCK_SIZE == 0, "dim must be a multiple of the fp8 block size"
    nbytes = rows * (dim + dim // FP8_BLOCK_SIZE)
    assert nbytes > 0, "at least one row is needed to hold the table"
    return nbytes


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
        model = get_model()
        model_path, revision = model.model_path, model.revision
    except (ValueError, AttributeError):
        # No published runtime context (unit tests, offline tools): nothing to drop.
        return 0, 0
    if not os.path.isdir(model_path):
        # A hub id: the files live in the local snapshot the loader read.
        try:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                model_path, revision=revision, local_files_only=True
            )
        except Exception:
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
    """Host memory for one engram table: for the shared layout a memfd every rank
    maps, else a private anonymous mapping collapsed onto huge pages.

    Lives for the whole process: the mapping, the memfd and the cudaHostRegister
    pin are never released because the table is read by every forward.
    """

    def __init__(
        self,
        layout: EngramTableLayout,
        bounds: tuple[int, ...],
        dim: int,
        name: str,
        group,
    ) -> None:
        load_rows = _shard_hash_columns(layout, bounds, group)
        if layout == EngramTableLayout.SHARED:
            start, end = 0, bounds[-1]  # each rank still hold full view
            nbytes = _table_nbytes(end - start, dim)
            fd = self._open_shared_fd(nbytes, name, group)
            mm = mmap.mmap(
                fd,
                nbytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            mm.madvise(mmap.MADV_HUGEPAGE)
            # Every rank holds the fd before rank 0 continues; the /proc path only
            # resolves while rank 0 keeps its descriptor.
            group.barrier()
        else:
            start, end = load_rows
            nbytes = _table_nbytes(end - start, dim)
            fd = None
            mm = mmap.mmap(
                -1,
                nbytes,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            mm.madvise(mmap.MADV_HUGEPAGE)
            # Cached checkpoint pages, left by a previous server or by the loader,
            # make the 512 MiB huge-page faults fall back, so empty them first.
            _drop_page_cache_once("before pre-faulting the per-rank shard")
            np.frombuffer(mm, dtype=np.uint8)[:: mmap.PAGESIZE] = 0

        self.mm = mm
        self.fd = fd
        self.is_shared = layout == EngramTableLayout.SHARED
        self.group = group
        self.dirty = False
        self.nbytes = nbytes
        self.bytes = torch.frombuffer(self.mm, dtype=torch.uint8)
        err = torch.cuda.cudart().cudaHostRegister(self.bytes.data_ptr(), nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister({nbytes} bytes) failed: {err}")
        device_ptr = _registered_device_ptr(self.bytes.data_ptr())
        self.storage = _make_storage(
            self.bytes,
            dim,
            row_start=start,
            num_rows=end - start,
            load_rows=load_rows,
            is_full=layout != EngramTableLayout.PRIVATE,
            device_ptr=device_ptr,
        )

    def _open_shared_fd(self, nbytes: int, name: str, group) -> int:
        owner = None
        if group.rank_in_group == 0:
            fd = os.memfd_create(name, 0)
            os.ftruncate(fd, nbytes)
            owner = (os.getpid(), fd)
        pid, owner_fd = group.broadcast_object(owner, src=0)
        if group.rank_in_group == 0:
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
        if self.is_shared:
            self.group.barrier()
        mapped_kb, huge_kb = _huge_pages_backing(self.bytes.data_ptr())
        if not self.is_shared and huge_kb < mapped_kb * 0.98:
            # The loader's own reads refilled the page cache; empty it again so the
            # collapse can find contiguous memory.
            _drop_checkpoint_page_cache()
            self._collapse()
            mapped_kb, huge_kb = _huge_pages_backing(self.bytes.data_ptr())
        pct = 100.0 * huge_kb / mapped_kb if mapped_kb else 0.0
        msg = (
            f"engram host table {label}: {'shared' if self.is_shared else 'private'}, "
            f"{mapped_kb / 2**10:.0f} MiB resident, {huge_kb / 2**10:.0f} MiB in huge pages "
            f"({pct:.0f}%)"
        )
        if huge_kb == 0:
            knob = "shmem_enabled" if self.is_shared else "enabled"
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
    and for the shared layout exported by fd so every rank maps one physical copy."""

    def __init__(
        self,
        layout: EngramTableLayout,
        bounds: tuple[int, ...],
        dim: int,
        name: str,
        group,
    ) -> None:
        load_rows = _shard_hash_columns(layout, bounds, group)
        if layout == EngramTableLayout.SHARED:
            # One allocation holds the whole table; each rank writes its own shard.
            start, end = 0, bounds[-1]
            nbytes = _table_nbytes(end - start, dim)
            if group.rank_in_group == 0:
                vmm = self._allocate(nbytes, group)
                size = group.broadcast_object(vmm.size(), src=0)
                fd = exchange_fd(group, vmm.export_fd(), name=name)
            else:
                size = group.broadcast_object(None, src=0)
                if size is None:
                    raise RuntimeError(
                        "engram host table: rank 0 could not allocate the VMM table"
                    )
                fd = exchange_fd(group, None, name=name)
                vmm = VMMHostMemory(size, fd)
            os.close(fd)
        else:
            start, end = load_rows
            nbytes = _table_nbytes(end - start, dim)
            vmm = self._allocate(nbytes)
        device_ptr = vmm.ptr()
        buffer = (ctypes.c_uint8 * nbytes).from_address(device_ptr)
        self.vmm = vmm
        self.group = group
        self.dirty = False
        self.is_shared = layout == EngramTableLayout.SHARED
        self.storage = _make_storage(
            torch.frombuffer(buffer, dtype=torch.uint8),
            dim,
            row_start=start,
            num_rows=end - start,
            load_rows=load_rows,
            is_full=layout != EngramTableLayout.PRIVATE,
            device_ptr=device_ptr,
        )

    @staticmethod
    def _allocate(nbytes: int, peers=None) -> VMMHostMemory:
        # cuMemCreate on a host NUMA node does not reclaim page cache, and the
        # checkpoint's cached pages can take most of the node.
        _drop_page_cache_once("before allocating the VMM host table")
        try:
            return VMMHostMemory(nbytes)
        except RuntimeError:
            if peers is not None:
                # The peers wait for the size; None makes them raise too.
                peers.broadcast_object(None, src=0)
            raise

    def mark_loaded(self):
        self.dirty = True

    def finish_load(self, label: str):
        if not self.dirty:
            return
        self.dirty = False
        if self.is_shared:
            self.group.barrier()


class _DeviceTableSymmMem:
    """The shared layout on the device: the rank's column shard in symmetric
    memory, which every peer reads in place. Every rank lays out the widest
    shard, as the symmetric pool needs the same shapes on every rank."""

    def __init__(
        self, layout: EngramTableLayout, bounds: tuple[int, ...], dim: int, group
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        assert layout == EngramTableLayout.SHARED
        start, end = load_rows = _shard_hash_columns(layout, bounds, group)
        w = group.world_size
        spans = [_shard_hash_columns_at(bounds, rank=r, world=w) for r in range(w)]
        num_padded_rows = max(e - s for s, e in spans)
        nbytes = _table_nbytes(num_padded_rows, dim)
        # Built under the model's default-device context.
        with symmetric_context(torch.get_default_device()):
            self.buf = torch.empty(nbytes, dtype=torch.uint8)
        handle = symm_mem.rendezvous(self.buf, group.device_group)
        self.group = group
        self.storage = _make_storage(
            self.buf,
            dim,
            row_start=start,
            num_rows=end - start,
            load_rows=load_rows,
            is_full=False,
            num_padded_rows=num_padded_rows,
            device_peer_pairs=(handle.buffer_ptrs, [start for start, _ in spans]),
        )
        self.dirty = False

    def mark_loaded(self):
        self.dirty = True

    def finish_load(self, label: str):
        if not self.dirty:
            return
        self.dirty = False
        # Peers read this shard in place: the loader's copies must land first.
        torch.cuda.current_stream().synchronize()
        self.group.barrier()


class _DeviceTable:
    """The private or replicated layout on the device: the rank's column shard or
    the whole table, in ordinary HBM."""

    def __init__(
        self, layout: EngramTableLayout, bounds: tuple[int, ...], dim: int, group
    ) -> None:
        assert layout != EngramTableLayout.SHARED
        start, end = _shard_hash_columns(layout, bounds, group)
        num_rows = end - start
        self.buf = torch.empty(_table_nbytes(num_rows, dim), dtype=torch.uint8)
        self.storage = _make_storage(
            self.buf,
            dim,
            row_start=start,
            num_rows=end - start,
            load_rows=(start, end),
            is_full=layout == EngramTableLayout.REPLICATED,
        )

    def mark_loaded(self):
        return

    def finish_load(self, label: str):
        return


class Table(Protocol):
    storage: EngramStorage

    def mark_loaded(self):
        """The loader wrote to the table."""

    def finish_load(self, label: str):
        """Collective: every rank calls it once its rows are loaded."""


def _device_share_blocker(bounds: tuple[int, ...], dim: int, world: int):
    """Why the device table cannot take the shared layout, or None if it can."""
    num_cols = len(bounds) - 1
    if not _is_cuda:
        return "it needs CUDA symmetric memory"
    if num_cols != 24 or dim != 256:
        return "the peer gather kernel is built for 24 hash columns of 256"
    if world > 8:
        return f"the peer gather kernel reads at most 8 shards, not {world}"
    if bounds[-1] >= 2**31:
        return "row ids must fit int32"
    return None


def create_engram_table(
    *,
    use_host: bool,
    layout: Optional[EngramTableLayout],
    bounds: tuple[int, ...],
    dim: int,
    name: str,
    group,
) -> Table:
    """One layer's table; ``bounds`` are the prefix sums of its hash columns.
    ``layout`` None picks one: replicated at TP 1, else shared on the host, and
    on the device shared where the peer gather supports it and custom all-reduce
    has verified peer access, private otherwise."""
    world = group.world_size
    if world == 1:
        # The only rank's shard is the whole table.
        layout = EngramTableLayout.REPLICATED
    if use_host:
        use_vmm = (
            envs.SGLANG_OPT_DSV41_ENGRAM_HOST_TABLE_VMM.get()
            and _is_cuda
            and is_vmm_available()
        )
        cls = _HostTableVMM if use_vmm else _HostTablePosix
        return cls(layout or EngramTableLayout.SHARED, bounds, dim, name, group)
    # device table:
    blocker = _device_share_blocker(bounds, dim, world)
    if layout is None:
        # Custom all-reduce comes up only once a real peer read works.
        can_p2p = group.ca_comm is not None and not group.ca_comm.disabled
        use_shared = blocker is None and can_p2p
        layout = EngramTableLayout.SHARED if use_shared else EngramTableLayout.PRIVATE
    if layout != EngramTableLayout.SHARED:
        return _DeviceTable(layout, bounds, dim, group)
    if blocker is not None:
        raise ValueError(f"engram device table, shared layout: {blocker}")
    return _DeviceTableSymmMem(layout, bounds, dim, group)
