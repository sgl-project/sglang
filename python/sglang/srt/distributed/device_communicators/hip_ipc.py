# SPDX-License-Identifier: Apache-2.0
"""HIP IPC shared buffers: the ROCm counterpart of
``CustomAllreduce.create_shared_buffer``.

``CustomAllreduce.create_shared_buffer`` hands Python a list of peer device
pointers, which is what a Triton/Gluon collective needs in order to address
other ranks' memory directly. It is CUDA-only: it goes through
``CudaRTLibrary``, which loads ``libcudart``.

On ROCm the existing collectives keep their peer pointers inside C++
(``quick_all_reduce`` exchanges handles via ``ops.qr_{get,open}_handles``; the
HIP branch of ``custom_all_reduce`` opens them inside ``init_custom_ar``), so
there is no in-tree way for a Python-level kernel to obtain them. This module
fills that gap with the same API shape as the CUDA path.

Pointers are raw ``hipMalloc`` allocations, not torch tensors, so they are
stable for the process lifetime and are never moved by the caching allocator —
a requirement for CUDA-graph capture, where the peer pointer table is baked
into the captured launch.
"""

from __future__ import annotations

import ctypes
import logging
from typing import List, Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# sizeof(hipIpcMemHandle_t); matches CUDA's cudaIpcMemHandle_t.
_HANDLE_BYTES = 64
# hipIpcMemLazyEnablePeerAccess
_LAZY_ENABLE_PEER_ACCESS = 1


class HipIpcMemHandle(ctypes.Structure):
    # c_ubyte, not c_char: a c_char array is treated as a NUL-terminated string
    # by ctypes, so both reading and assigning would truncate the 64-byte handle
    # at its first zero byte.
    _fields_ = [("reserved", ctypes.c_ubyte * _HANDLE_BYTES)]


class HipRTLibrary:
    """Minimal ctypes binding for the HIP runtime calls we need."""

    _instance: Optional[HipRTLibrary] = None

    def __new__(cls) -> HipRTLibrary:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_lib()
        return cls._instance

    def _init_lib(self) -> None:
        self.lib = ctypes.CDLL("libamdhip64.so")
        self.lib.hipMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.hipMalloc.restype = ctypes.c_int
        self.lib.hipFree.argtypes = [ctypes.c_void_p]
        self.lib.hipFree.restype = ctypes.c_int
        self.lib.hipMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        self.lib.hipMemset.restype = ctypes.c_int
        self.lib.hipIpcGetMemHandle.argtypes = [
            ctypes.POINTER(HipIpcMemHandle),
            ctypes.c_void_p,
        ]
        self.lib.hipIpcGetMemHandle.restype = ctypes.c_int
        self.lib.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            HipIpcMemHandle,
            ctypes.c_uint,
        ]
        self.lib.hipIpcOpenMemHandle.restype = ctypes.c_int
        self.lib.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self.lib.hipIpcCloseMemHandle.restype = ctypes.c_int
        self.lib.hipMemGetAddressRange.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        self.lib.hipMemGetAddressRange.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, op: str) -> None:
        if status != 0:
            raise RuntimeError(f"{op} failed with HIP status {status}")

    def malloc(self, size_in_bytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self.lib.hipMalloc(ctypes.byref(ptr), size_in_bytes), "hipMalloc")
        return ptr

    def free(self, ptr: int) -> None:
        self._check(self.lib.hipFree(ctypes.c_void_p(ptr)), "hipFree")

    def memset(self, ptr: ctypes.c_void_p, value: int, size_in_bytes: int) -> None:
        self._check(self.lib.hipMemset(ptr, value, size_in_bytes), "hipMemset")

    def get_ipc_handle(self, ptr: ctypes.c_void_p) -> bytes:
        handle = HipIpcMemHandle()
        self._check(
            self.lib.hipIpcGetMemHandle(ctypes.byref(handle), ptr),
            "hipIpcGetMemHandle",
        )
        return ctypes.string_at(ctypes.addressof(handle), _HANDLE_BYTES)

    def open_ipc_handle(self, raw: bytes) -> int:
        if len(raw) != _HANDLE_BYTES:
            raise ValueError(
                f"IPC handle must be {_HANDLE_BYTES} bytes, got {len(raw)}"
            )
        handle = HipIpcMemHandle()
        ctypes.memmove(ctypes.addressof(handle), raw, _HANDLE_BYTES)
        opened = ctypes.c_void_p()
        self._check(
            self.lib.hipIpcOpenMemHandle(
                ctypes.byref(opened), handle, _LAZY_ENABLE_PEER_ACCESS
            ),
            "hipIpcOpenMemHandle",
        )
        if not opened.value:
            raise RuntimeError("hipIpcOpenMemHandle returned NULL")
        return opened.value

    def close_ipc_handle(self, ptr: int) -> None:
        self._check(
            self.lib.hipIpcCloseMemHandle(ctypes.c_void_p(ptr)),
            "hipIpcCloseMemHandle",
        )


def create_shared_buffer(
    size_in_bytes: int,
    group: Optional[ProcessGroup] = None,
    zero_fill: bool = False,
) -> List[int]:
    """Allocate ``size_in_bytes`` on every rank and return all peer pointers.

    The returned list is in **global rank order**: entry ``i`` addresses rank
    ``i``'s allocation, and entry ``dist.get_rank(group)`` is this rank's own.
    Mirrors ``CustomAllreduce.create_shared_buffer``.

    ``zero_fill`` is required for synchronization words, whose protocols assume
    counters start at zero.
    """
    lib = HipRTLibrary()
    pointer = lib.malloc(size_in_bytes)
    if zero_fill:
        lib.memset(pointer, 0, size_in_bytes)

    handle = lib.get_ipc_handle(pointer)
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    handles: List[Optional[bytes]] = [None] * world_size
    dist.all_gather_object(handles, handle, group=group)

    pointers: List[int] = []
    for i, h in enumerate(handles):
        pointers.append(pointer.value if i == rank else lib.open_ipc_handle(h))
    return pointers


def create_shared_tensor(
    shape,
    dtype,
    device,
    group: Optional[ProcessGroup] = None,
    zero_fill: bool = True,
):
    """Allocate a torch tensor and return ``(tensor, peer_pointers)``.

    Unlike :func:`create_shared_buffer`, the local allocation is an ordinary
    torch tensor, so callers can ``copy_()`` into it and pass it to kernels.
    Because torch's caching allocator hands out offsets inside larger segments,
    the IPC handle is taken on the *segment base* (via ``hipMemGetAddressRange``)
    and the intra-segment offset is exchanged alongside it; peers reopen the base
    and re-apply the offset.

    ``peer_pointers`` is in global rank order; entry ``rank`` aliases
    ``tensor.data_ptr()``.
    """
    import torch

    lib = HipRTLibrary()
    # Always zero-initialised: the synchronisation rows require counters to
    # start at zero, and a zeroed staging buffer is harmless.
    tensor = torch.zeros(shape, dtype=dtype, device=device)

    base = ctypes.c_void_p()
    size = ctypes.c_size_t()
    data_ptr = tensor.data_ptr()
    lib._check(
        lib.lib.hipMemGetAddressRange(
            ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(data_ptr)
        ),
        "hipMemGetAddressRange",
    )
    offset = data_ptr - base.value
    handle = lib.get_ipc_handle(base)

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    payload = [None] * world_size
    dist.all_gather_object(payload, (handle, offset), group=group)

    pointers: List[int] = []
    opened_bases: List[int] = []
    for i, (h, off) in enumerate(payload):
        if i == rank:
            pointers.append(data_ptr)
        else:
            peer_base = lib.open_ipc_handle(h)
            opened_bases.append(peer_base)
            pointers.append(peer_base + off)
    if pointers[rank] != data_ptr:
        raise RuntimeError("local IPC pointer does not alias the source tensor")
    # `opened_bases` must be handed to close_shared_tensor(); closing
    # base+offset is invalid, and the local side is owned by torch.
    return tensor, pointers, opened_bases


def register_peer_pointers(tensors, group: Optional[ProcessGroup] = None):
    """Publish existing tensors over IPC and return their peer pointers.

    Unlike :func:`create_shared_tensor`, the tensors already exist and are owned
    by someone else (here: activations captured inside a CUDA graph). One
    batched ``all_gather_object`` covers the whole list, so this costs a single
    collective no matter how many call sites were captured.

    Returns ``(rows, opened_bases)`` where ``rows[i]`` is the global-rank-ordered
    peer pointer list for ``tensors[i]``.

    Every rank must pass the same number of tensors in the same order.
    """
    lib = HipRTLibrary()
    local = []
    for t in tensors:
        base = ctypes.c_void_p()
        size = ctypes.c_size_t()
        data_ptr = t.data_ptr()
        lib._check(
            lib.lib.hipMemGetAddressRange(
                ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(data_ptr)
            ),
            "hipMemGetAddressRange",
        )
        local.append((lib.get_ipc_handle(base), data_ptr - base.value))

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local, group=group)
    if any(len(g) != len(local) for g in gathered):
        raise RuntimeError("ranks captured different numbers of collective sites")

    rows, opened_bases = [], []
    # Reopening the same segment handle repeatedly is wasteful and can exhaust
    # the mapping table, so cache per (rank, handle).
    cache = {}
    for i in range(len(local)):
        row = []
        for r in range(world_size):
            handle, offset = gathered[r][i]
            if r == rank:
                row.append(tensors[i].data_ptr())
                continue
            key = (r, handle)
            if key not in cache:
                peer_base = lib.open_ipc_handle(handle)
                cache[key] = peer_base
                opened_bases.append(peer_base)
            row.append(cache[key] + offset)
        rows.append(row)
    return rows, opened_bases


def close_shared_tensor(opened_bases: List[int]) -> None:
    """Close peer mappings created by :func:`create_shared_tensor`."""
    lib = HipRTLibrary()
    for base in opened_bases:
        try:
            lib.close_ipc_handle(base)
        except RuntimeError as exc:  # teardown must not mask the real error
            logger.debug("hipIpcCloseMemHandle failed during teardown: %s", exc)


def free_shared_buffer(
    pointers: List[int], group: Optional[ProcessGroup] = None
) -> None:
    """Close peer mappings and free this rank's own allocation."""
    lib = HipRTLibrary()
    rank = dist.get_rank(group=group)
    for i, ptr in enumerate(pointers):
        if ptr and i != rank:
            try:
                lib.close_ipc_handle(ptr)
            except RuntimeError as exc:  # teardown must not mask the real error
                logger.debug("hipIpcCloseMemHandle failed during teardown: %s", exc)
    if pointers and pointers[rank]:
        lib.free(pointers[rank])
