"""Symmetric peer memory for in-kernel TP all-reduces, through the HIP runtime
this process loaded (``create_shared_buffer`` supports only CUDA). Construction is
collective, and all ranks refuse together.

Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6 (MIT License,
Copyright (C) 2026, Advanced Micro Devices, Inc.): atom/mono/runtime/peer_memory.py.
"""

from __future__ import annotations

import ctypes
import functools

import torch
import torch.distributed as dist

_IPC_HANDLE_BYTES = 64
_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
_DEVICE_MALLOC_UNCACHED = 0x3
_PAGE = 4096  # hipIpcGetMemHandle on uncached allocations under 2 MiB needs whole pages


class PeerMemoryError(RuntimeError):
    """The peer handshake failed on this or another rank."""


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * _IPC_HANDLE_BYTES)]


@functools.cache
def _hip():
    """The HIP runtime this process loaded (torch's), not a second copy."""
    path = None
    with open("/proc/self/maps") as maps:
        for line in maps:
            if "libamdhip64" in line:
                path = line.split()[-1]
                break
    lib = ctypes.CDLL(path or "libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
    sig = {
        "hipExtMallocWithFlags": [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ],
        "hipMemset": [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
        "hipFree": [ctypes.c_void_p],
        "hipIpcGetMemHandle": [ctypes.POINTER(_IpcHandle), ctypes.c_void_p],
        "hipIpcOpenMemHandle": [
            ctypes.POINTER(ctypes.c_void_p),
            _IpcHandle,
            ctypes.c_uint,
        ],
        "hipIpcCloseMemHandle": [ctypes.c_void_p],
        "hipGetErrorString": [ctypes.c_int],
    }
    for name, args in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_char_p if name == "hipGetErrorString" else ctypes.c_int
    return lib


def _check(err: int, what: str) -> None:
    if err:
        msg = _hip().hipGetErrorString(err)
        raise RuntimeError(f"{what}: {msg.decode() if msg else err}")


def _alloc_uncached(nbytes: int) -> int:
    size = -(-nbytes // _PAGE) * _PAGE
    ptr = ctypes.c_void_p()
    _check(
        _hip().hipExtMallocWithFlags(ctypes.byref(ptr), size, _DEVICE_MALLOC_UNCACHED),
        "hipExtMallocWithFlags",
    )
    _check(_hip().hipMemset(ptr, 0, size), "hipMemset")
    return int(ptr.value)


def _handle(ptr: int) -> bytes:
    h = _IpcHandle()
    _check(
        _hip().hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(ptr)),
        "hipIpcGetMemHandle",
    )
    return bytes(ctypes.string_at(ctypes.byref(h), _IPC_HANDLE_BYTES))


def _open(handle: bytes) -> int:
    h = _IpcHandle()
    ctypes.memmove(ctypes.byref(h), handle, _IPC_HANDLE_BYTES)
    out = ctypes.c_void_p()
    _check(
        _hip().hipIpcOpenMemHandle(
            ctypes.byref(out), h, _IPC_MEM_LAZY_ENABLE_PEER_ACCESS
        ),
        "hipIpcOpenMemHandle",
    )
    return int(out.value)


class _DeviceBytes:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class PeerBuffer:
    """One uncached buffer a rank and every peer's address (``addresses``, int64
    on the device). ``group`` is a CPU (gloo) group of the ``npes`` ranks."""

    def __init__(self, nbytes: int, group, rank: int, npes: int, device: torch.device):
        self.rank = rank
        self.local = _alloc_uncached(nbytes)
        # a view, not an owner: close() frees the memory under it
        self.bytes = torch.as_tensor(_DeviceBytes(self.local, nbytes), device=device)
        self._opened: list[int] = []
        addresses = [self.local]
        if npes > 1:
            handles: list = [None] * npes
            dist.all_gather_object(handles, _handle(self.local), group=group)
            addresses, failure = [], None
            try:
                for peer, handle in enumerate(handles):
                    if peer == rank:
                        addresses.append(self.local)
                    else:
                        base = _open(handle)
                        self._opened.append(base)
                        addresses.append(base)
            except RuntimeError as err:  # every rank must still reach the agreement
                failure = err
            ok = torch.tensor([int(failure is None)], dtype=torch.int32)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
            if not ok.item():
                self.close()
                raise PeerMemoryError(
                    f"peer memory handshake failed on {'this' if failure else 'another'} rank"
                ) from failure
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self) -> None:
        """Unmap the peers and free the local buffer (idempotent)."""
        opened, self._opened = self._opened, []
        for base in opened:
            _hip().hipIpcCloseMemHandle(ctypes.c_void_p(base))
        if self.local:
            self.bytes = None
            _hip().hipFree(ctypes.c_void_p(self.local))
            self.local = 0
