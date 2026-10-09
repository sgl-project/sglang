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
from typing import Protocol

import numpy as np
import torch

from sglang.kernels.jit.utils.cuda.vmm import VMMHostMemory, is_vmm_available
from sglang.srt.distributed.device_communicators.cuda_wrapper import (
    find_loaded_library,
)
from sglang.srt.distributed.fd_exchange import exchange_fd
from sglang.srt.runtime_context import get_model
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()


class EngramTableLayout(str, enum.Enum):
    FULL_SHARED = "full_shared"
    ROW_SHARDED = "row_sharded"

    @staticmethod
    def parse(use_host: bool, value: str) -> EngramTableLayout:
        if not value:
            value = "full_shared" if use_host else "row_sharded"
        return EngramTableLayout(value)

    def shard(self, bounds: tuple[int, ...], tp_size: int, tp_rank: int):
        assert (len(bounds) - 1) % tp_size == 0
        avg = (len(bounds) - 1) // tp_size
        if self == EngramTableLayout.ROW_SHARDED:
            return bounds[avg * tp_rank], bounds[avg * (tp_rank + 1)]
        elif self == EngramTableLayout.FULL_SHARED:
            return 0, bounds[-1]
        raise ValueError(f"Unknown EngramTableLayout {self}")


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
    """Host-memory backing for one engram table ('shared' or 'per_rank' layout).

    Lives for the whole process: the mapping, the memfd and the cudaHostRegister
    pin are never released because the table is read by every forward.
    """

    def __init__(self, layout: EngramTableLayout, nbytes: int, name: str, group):
        self.layout = layout
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
        self.device_ptr = _registered_device_ptr(self.bytes.data_ptr())

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
    def __init__(self, layout: EngramTableLayout, nbytes: int, name: str, group):
        self.group = group
        self.layout = layout
        rank = group.rank_in_group
        # Both sides close their fd once the mapping exists: SCM_RIGHTS copies
        # are independent descriptors, so there is no keep-alive requirement.
        raw_nbytes = nbytes
        if layout == EngramTableLayout.FULL_SHARED:
            if rank == 0:
                self.vmm = VMMHostMemory(nbytes)
                nbytes = group.broadcast_object(self.vmm.size(), src=0)
                fd = exchange_fd(group, self.vmm.export_fd(), name=name)
            else:
                nbytes = group.broadcast_object(None, src=0)
                fd = exchange_fd(group, None, name=name)
                self.vmm = VMMHostMemory(nbytes, fd)
            os.close(fd)
        else:
            self.vmm = VMMHostMemory(nbytes)
        # create a torch tensor that shares the VMMHostMemory, so that it can be used in the model
        self.device_ptr = self.vmm.ptr()
        self.bytes = torch.frombuffer(
            ctypes.cast(
                self.device_ptr,
                ctypes.POINTER(ctypes.c_uint8 * raw_nbytes),
            ).contents,
            dtype=torch.uint8,
        )
        self.dirty = False

    def mark_loaded(self):
        self.dirty = True

    def finish_load(self, label: str):
        if not self.dirty:
            return
        self.dirty = False
        self.group.barrier()


class _DeviceTable:
    def __init__(self, layout: EngramTableLayout, nbytes: int):
        assert layout == EngramTableLayout.ROW_SHARDED
        self.layout = layout
        self.bytes = torch.empty(nbytes, dtype=torch.uint8)
        self.device_ptr = self.bytes.data_ptr()

    def mark_loaded(self):
        pass

    def finish_load(self, label: str):
        pass


class Table(Protocol):
    layout: EngramTableLayout
    bytes: torch.Tensor
    device_ptr: int

    def mark_loaded(self): ...

    def finish_load(self, label: str): ...


def create_engram_table(
    *,
    use_host: bool,
    layout: EngramTableLayout,
    nbytes: int,
    name: str,
    group,
) -> Table:
    if use_host:
        if _is_cuda and not _is_hip and is_vmm_available():
            return _HostTableVMM(layout, nbytes, name, group)
        return _HostTablePosix(layout, nbytes, name, group)
    else:
        # TODO: support symmetric memory allocation for device tables on CUDA
        return _DeviceTable(layout, nbytes)
