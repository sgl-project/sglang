"""The host copy of every expert, the transfer of experts into GPU slots, and the loads that
fill it."""

from __future__ import annotations

import concurrent.futures
import ctypes
import functools
import logging
import math
import mmap
import os
import sys
import threading
import weakref
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

logger = logging.getLogger(__name__)


def _pin_until_freed(shape, dtype: torch.dtype) -> torch.Tensor:
    """Page-locked host memory of exactly the tensor's size (torch's pinned allocator rounds
    every allocation up to a power of two): shared memory of that size, registered with CUDA
    in place, as multimodal_gen's layerwise offload does.

    The registration lasts as long as the storage, so as long as any tensor or view on it.
    When the storage is freed, the device first finishes its queued reads (page-in copies,
    gather), then the pages are unregistered, before torch releases them. Stores live as long
    as their model, so that sync happens only at exit or in tests.
    """
    nbytes = math.prod(shape) * dtype.itemsize
    fd = os.memfd_create("sglang-paged-experts", os.MFD_CLOEXEC)
    try:
        os.ftruncate(fd, nbytes)
        storage = torch.UntypedStorage.from_file(
            f"/proc/self/fd/{fd}", shared=True, nbytes=nbytes
        )
    finally:
        os.close(fd)
    rc = int(torch.cuda.cudart().cudaHostRegister(storage.data_ptr(), nbytes, 0))
    if rc != 0:
        raise RuntimeError(
            f"Paged experts: cudaHostRegister of {nbytes} bytes failed (error {rc})"
        )
    weakref.finalize(storage, _unregister, storage.data_ptr())
    # On the CPU, as its storage, also under a default CUDA device (model construction).
    return torch.empty(0, dtype=dtype, device="cpu").set_(storage, 0, tuple(shape))


def _unregister(data_ptr: int) -> None:
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    rc = int(torch.cuda.cudart().cudaHostUnregister(data_ptr))
    if rc != 0:
        logger.warning("Paged experts: cudaHostUnregister failed (error %d)", rc)


RowsLike = Dict[str, Tuple[Tuple[int, ...], torch.dtype]]


def rows_like(layer, names: Sequence[str]) -> RowsLike:
    """Each tensor's row shape and dtype, as the layer has it."""
    return {
        name: (tuple(getattr(layer, name).shape[1:]), getattr(layer, name).dtype)
        for name in names
    }


class HostExpertStore:
    """Holds all E experts of each paged tensor on the host, in the layer's GPU layout,
    page-locked. ``page_in`` is planned on the host; ``gather`` reads a plan on the GPU.

    The GPU tensors are looked up on the layer at every transfer, because a quantization
    method may rebind them after loading.
    """

    def __init__(self, rows: RowsLike, num_experts: int):
        self.host: Dict[str, torch.Tensor] = {
            name: self._alloc((num_experts, *shape), dtype)
            for name, (shape, dtype) in rows.items()
        }

    def _alloc(self, shape, dtype: torch.dtype) -> torch.Tensor:
        if math.prod(shape) == 0:
            return torch.empty(shape, dtype=dtype, device="cpu")
        return _pin_until_freed(shape, dtype)

    def page_in(self, layer, loads: Sequence[Tuple[int, int]]) -> None:
        """Copy each ``(expert, slot)`` pair's expert, of every paged tensor, into its slot."""
        for name, host in self.host.items():
            gpu = getattr(layer, name).data
            for expert, slot in loads:
                gpu[slot].copy_(host[expert], non_blocking=True)

    def bind_device_gather(self, layer) -> None:
        """Prepare ``gather``; called once the layer's tensors are final. The gather kernel
        reads the pinned store directly, through its UVA address."""
        from sglang.kernels.ops.moe.paged_experts import (
            paged_experts_host_device_pointer,
        )

        hosts, gpus, sizes = [], [], []
        for name, host in self.host.items():
            gpu = getattr(layer, name).data
            row_bytes = host[0].numel() * host.element_size()
            if row_bytes == 0:
                continue  # e.g. GPTQ without act-order keeps empty g_idx tensors
            if not (gpu.is_contiguous() and row_bytes % 2 == 0):
                raise RuntimeError(
                    f"Paged experts: CUDA graphs need contiguous expert rows of an even "
                    f"size; {name} has {row_bytes} bytes per expert, "
                    f"contiguous={gpu.is_contiguous()}"
                )
            hosts.append(paged_experts_host_device_pointer(host))
            gpus.append(gpu.data_ptr())
            sizes.append(row_bytes)
        self._gather_args = [
            torch.tensor(v, dtype=torch.int64, device=gpu.device)
            for v in (hosts, gpus, sizes)
        ]

    def gather(self, src: torch.Tensor, dst: torch.Tensor, count: torch.Tensor) -> None:
        """Copy expert ``src[i]`` into GPU slot ``dst[i]`` for ``i < count[0]``, with every
        index read on the GPU (a device-planned step): capturable in a CUDA graph."""
        from sglang.kernels.ops.moe.paged_experts import paged_experts_gather

        paged_experts_gather(*self._gather_args, src, dst, count)


class ExpertLoads:
    """A layer's expert loads until the post-load step. A load waits (``defer``) as a write
    on a view of the mapped checkpoint, and ``rows`` replays a few experts' loads into K rows,
    so no table of all E experts is filled; a load that cannot wait (a stacked tensor, a
    copy) sends the waiting loads and every later one to ``table``, as without waiting."""

    def __init__(
        self,
        layer,
        names: Sequence[str],
        num_experts: int,
        num_slots: int,
        wait: bool = True,
    ):
        self.rows_like = rows_like(layer, names)
        self.num_experts = num_experts
        self.num_slots = num_slots
        # What a row holds where no load writes (e.g. padding a kernel adds), as on the GPU.
        self._created = {name: getattr(layer, name).data[:1].cpu() for name in names}
        self.waiting = wait
        # Experts whose checkpoint pages stay after a replay: a caller replays them again
        # soon and gives them back itself. Every other replayed expert's are given back.
        self.keep: Set[int] = set()
        self._writes: Dict[int, List[Tuple[str, Callable, torch.Tensor]]] = {}
        self._table: Optional[Dict[str, torch.Tensor]] = None
        self._buffer: Optional[Dict[str, torch.Tensor]] = None
        self._lock = threading.Lock()  # a model's loader may run in threads
        # Gives replayed pages back off the loading thread (``page_out``); ``drop`` ends it.
        self._pager: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def defer(
        self, expert: int, name: str, write: Callable, view: torch.Tensor
    ) -> bool:
        """Keep ``write`` (of ``name``'s row of ``expert`` into the one-row tensor it is
        given, reading ``view``) for later; False once the loads stopped waiting."""
        with self._lock:
            if not self.waiting:
                return False
            self._writes.setdefault(expert, []).append((name, write, view))
            return True

    @property
    def table(self) -> Dict[str, torch.Tensor]:
        """Every expert's rows; the waiting loads are written, and loads stop waiting."""
        with self._lock:  # a load in another thread waits until the rows are filled
            if self._table is None:
                self._table = {
                    name: c.expand(self.num_experts, *c.shape[1:]).clone()
                    for name, c in self._created.items()
                }
                writes, self._writes, self.waiting = self._writes, {}, False
                for expert, items in writes.items():
                    for name, write, _ in items:
                        write(self._table[name][expert : expert + 1])
        return self._table

    def rows(
        self,
        experts: Sequence[int],
        out: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Each tensor's rows of ``experts`` (at most K; repeats allowed), in ``out`` or in a
        buffer the next call reuses; their checkpoint pages are then given back, unless
        ``keep`` holds them."""
        if out is None:
            if self._buffer is None:
                self._buffer = {
                    name: torch.empty((self.num_slots, *shape), dtype=dtype)
                    for name, (shape, dtype) in self.rows_like.items()
                }
            out = {name: b[: len(experts)] for name, b in self._buffer.items()}
        if self._table is not None:
            for name, r in out.items():
                r.copy_(self._table[name][list(experts)])
            return out
        first = {}
        for i, expert in enumerate(experts):
            if expert in first:  # a repeat (padding): copy the row written already
                for r in out.values():
                    r[i].copy_(r[first[expert]])
                continue
            first[expert] = i
            for name, r in out.items():
                r[i].copy_(self._created[name][0])
            for name, write, _ in self._writes.get(expert, ()):
                write(out[name][i : i + 1])
        self.page_out(e for e in first if e not in self.keep)
        return out

    def read_ahead(self, experts: Iterable[int]) -> None:
        """Read the checkpoint pages of ``experts``' waiting loads ahead
        (``MADV_WILLNEED``)."""
        self._advise(experts, _MADV_WILLNEED)

    def page_out(self, experts: Iterable[int]) -> None:
        """Give the checkpoint pages of ``experts``' waiting loads back (``MADV_PAGEOUT``), once
        replayed, so that they push no other memory to swap. The kernel reclaims them
        synchronously, so this runs on a thread of its own, next to the next chunk's replay."""
        views = [v for e in experts for _, _, v in self._writes.get(e, ())]
        if not views or _MADV_PAGEOUT is None:
            return
        if self._pager is None:
            self._pager = concurrent.futures.ThreadPoolExecutor(
                1, thread_name_prefix="paged-experts-pageout"
            )
        self._pager.submit(_madvise_all, views, _MADV_PAGEOUT)

    def _advise(self, experts: Iterable[int], advice: Optional[int]) -> None:
        _madvise_all(
            [v for e in experts for _, _, v in self._writes.get(e, ())], advice
        )

    def drop(self) -> None:
        """Release the loads and the checkpoint views they hold, once the pages handed to
        ``page_out`` are given back."""
        if self._pager is not None:
            pager, self._pager = self._pager, None
            pager.shutdown(wait=True)
        with self._lock:
            self._writes, self._table, self._buffer = {}, None, None
            self.waiting = False
            self.keep = set()


# Linux's values; Python's mmap module exports MADV_PAGEOUT only where it was built with it.
_LINUX = sys.platform.startswith("linux")
_MADV_WILLNEED = getattr(mmap, "MADV_WILLNEED", None)
_MADV_PAGEOUT = getattr(mmap, "MADV_PAGEOUT", 21 if _LINUX else None)


@functools.lru_cache(maxsize=None)
def _libc():
    """The C library's ``madvise``, loaded on first use; None where there is none."""
    if not _LINUX:
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    except (OSError, AttributeError):
        return None
    return libc


def _madvise_all(views: Sequence[torch.Tensor], advice: Optional[int]) -> None:
    for view in views:
        _madvise(view, advice)


def _madvise(view: torch.Tensor, advice: Optional[int]) -> None:
    """``madvise`` the pages a tensor (a view of a mapped file) spans. Advice the kernel does
    not know (``MADV_PAGEOUT`` before Linux 5.4) is ignored: never ``MADV_DONTNEED``, which
    would zero memory that is not backed by a file."""
    nbytes = view.numel() * view.element_size()
    libc = _libc()
    if not nbytes or advice is None or libc is None:
        return
    start = view.data_ptr() // mmap.PAGESIZE * mmap.PAGESIZE
    length = view.data_ptr() + nbytes - start
    libc.madvise(start, length, advice)


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal as bytes: NaN equals itself and -0.0 differs from 0.0, unlike ``torch.equal``."""
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
        )
    )
