"""HIP virtual-memory (VMM) buffers shared across one host or fabric clique.

Every rank allocates one VMM allocation with ``hipMemCreate``, exports it as a
shareable handle, and maps every peer's allocation into its own address space.
This is the ROCm counterpart of torch symmetric memory for the JIT custom
all-reduce: it needs only the HIP runtime already loaded by torch and no
multicast object.

Two handle types are supported. A FABRIC handle is a 64-byte value exchanged
with ``all_gather_object``; a POSIX fd is exchanged over UNIX sockets
(``SCM_RIGHTS``). ``SGLANG_HIP_VMM_HANDLE_TYPE`` picks one (``auto``, the
default, uses FABRIC when the device supports it).
"""

from __future__ import annotations

import ctypes
import logging
import os
from datetime import timedelta
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.environ import envs
from sglang.srt.utils.cuda_vmm_utils import exchange_posix_fds

logger = logging.getLogger(__name__)

hipError_t = ctypes.c_int

# hip_runtime_api.h / driver_types.h
_HIP_MEM_ALLOCATION_TYPE_PINNED = 0x1
_HIP_MEM_ALLOCATION_TYPE_UNCACHED = 0x40000000
_HIP_MEM_HANDLE_TYPE_POSIX_FD = 0x1
_HIP_MEM_HANDLE_TYPE_FABRIC = 0x8
_HIP_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 95
_HIP_FABRIC_HANDLE_BYTES = 64
_HIP_MEM_LOCATION_TYPE_DEVICE = 1
_HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE = 3
_HIP_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
_HIP_ERROR_OUT_OF_MEMORY = 2
_MIN_CHUNK_BYTES = 2 << 20
_TEARDOWN_BARRIER_TIMEOUT = timedelta(seconds=60)


def teardown_barrier(group: ProcessGroup) -> bool:
    """Barrier that returns False instead of hanging when a peer never arrives."""
    try:
        if dist.get_backend(group) == dist.Backend.GLOO:
            dist.monitored_barrier(group=group, timeout=_TEARDOWN_BARRIER_TIMEOUT)
        else:
            dist.barrier(group=group)
        return True
    except Exception as e:
        logger.warning("VMM teardown barrier failed, continuing without it: %s", e)
        return False


class _HipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _HipMemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
    ]


class _HipMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _HipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _HipMemAllocFlags),
    ]


class _HipMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _HipMemLocation), ("flags", ctypes.c_int)]


class _HipMemFabricHandle(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * _HIP_FABRIC_HANDLE_BYTES)]


_PROTOTYPES = {
    "hipGetErrorString": (ctypes.c_char_p, [hipError_t]),
    "hipGetLastError": (hipError_t, []),
    "hipRuntimeGetVersion": (hipError_t, [ctypes.POINTER(ctypes.c_int)]),
    "hipDeviceGetAttribute": (
        hipError_t,
        [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
    ),
    "hipMemGetAllocationGranularity": (
        hipError_t,
        [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(_HipMemAllocationProp),
            ctypes.c_int,
        ],
    ),
    "hipMemCreate": (
        hipError_t,
        [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.POINTER(_HipMemAllocationProp),
            ctypes.c_ulonglong,
        ],
    ),
    "hipMemExportToShareableHandle": (
        hipError_t,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong],
    ),
    "hipMemImportFromShareableHandle": (
        hipError_t,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int],
    ),
    "hipMemAddressReserve": (
        hipError_t,
        [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ],
    ),
    "hipMemMap": (
        hipError_t,
        [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ],
    ),
    "hipMemSetAccess": (
        hipError_t,
        [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(_HipMemAccessDesc),
            ctypes.c_size_t,
        ],
    ),
    "hipMemUnmap": (hipError_t, [ctypes.c_void_p, ctypes.c_size_t]),
    "hipMemAddressFree": (hipError_t, [ctypes.c_void_p, ctypes.c_size_t]),
    "hipMemRelease": (hipError_t, [ctypes.c_void_p]),
}

_LIB: Optional[ctypes.CDLL] = None


def _hip() -> ctypes.CDLL:
    """The HIP runtime torch already loaded, with the VMM entry points typed."""
    global _LIB
    if _LIB is None:
        from sglang.srt.distributed.device_communicators.cuda_wrapper import (
            find_loaded_library,
        )

        torch.cuda.init()
        path = find_loaded_library("libamdhip64")
        if path is None:
            raise RuntimeError("libamdhip64 is not loaded in this process")
        lib = ctypes.CDLL(path)
        for name, (restype, argtypes) in _PROTOTYPES.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = restype, argtypes
        _LIB = lib
    return _LIB


def _check(err: int, what: str) -> None:
    if err != 0:
        msg = _hip().hipGetErrorString(err)
        raise RuntimeError(f"{what} failed: {msg.decode() if msg else err}")


# Oldest HIP runtime the ROCm custom all-reduce (v2) path is validated on. It is
# the HIP version, not the ROCm release, that matters: ROCm 10.x ships HIP 7.16+.
MIN_HIP_VERSION = (7, 2)


def hip_runtime_version() -> Tuple[int, int, int]:
    """``(major, minor, patch)`` of the HIP runtime this process has loaded."""
    version = ctypes.c_int()
    _check(_hip().hipRuntimeGetVersion(ctypes.byref(version)), "hipRuntimeGetVersion")
    v = version.value  # HIP_VERSION: major * 10^7 + minor * 10^5 + patch
    return v // 10_000_000, v // 100_000 % 100, v % 100_000


def is_hip_runtime_supported() -> bool:
    return hip_runtime_version()[:2] >= MIN_HIP_VERSION


def is_fabric_handle_supported(device_index: int) -> bool:
    supported = ctypes.c_int(0)
    err = _hip().hipDeviceGetAttribute(
        ctypes.byref(supported),
        _HIP_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        device_index,
    )
    if err != 0:
        _hip().hipGetLastError()
        return False
    return supported.value != 0


def _resolve_handle_type(device_index: int) -> int:
    choice = envs.SGLANG_HIP_VMM_HANDLE_TYPE.get().lower()
    if choice == "posix_fd":
        return _HIP_MEM_HANDLE_TYPE_POSIX_FD
    if choice == "fabric":
        return _HIP_MEM_HANDLE_TYPE_FABRIC
    if choice != "auto":
        raise ValueError(
            f"SGLANG_HIP_VMM_HANDLE_TYPE={choice!r}; expected auto, fabric or posix_fd"
        )
    if is_fabric_handle_supported(device_index):
        return _HIP_MEM_HANDLE_TYPE_FABRIC
    return _HIP_MEM_HANDLE_TYPE_POSIX_FD


class _DeviceArray:
    """`__cuda_array_interface__` over a mapped range; keeps its buffer alive."""

    def __init__(self, owner: HipSymmetricBuffer, ptr: int, nbytes: int):
        self.owner = owner
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class HipSymmetricBuffer:
    """One ``nbytes`` VMM allocation per rank, mapped into every rank's address space.

    ``buffers[i]`` is a uint8 view of rank ``i``'s allocation, addressable from
    this rank's device. The mappings live until every view is gone.

    Uncached (``hipMemAllocationTypeUncached``) memory is the default because
    kernels poll it for peer writes within a launch, which the non-coherent L2
    would otherwise hide; RCCL allocates its VMM transport buffers the same way.

    ``require_fabric`` forces FABRIC handles regardless of
    ``SGLANG_HIP_VMM_HANDLE_TYPE``: a group spanning hosts cannot pass POSIX fds.
    """

    def __init__(
        self,
        nbytes: int,
        device: torch.device,
        group: ProcessGroup,
        *,
        uncached: bool = True,
        require_fabric: bool = False,
    ) -> None:
        if not is_hip_runtime_supported():
            raise RuntimeError(
                f"HIP runtime {hip_runtime_version()} is older than {MIN_HIP_VERSION}"
            )
        lib = _hip()
        self.device = torch.device(device)
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self._handles: List[ctypes.c_void_p] = []  # this rank's first, then imports
        self._num_local_handles = 0
        # (reserved VA, [(mapped address, bytes), ...]) per rank
        self._mapped: List[Tuple[int, List[Tuple[int, int]]]] = []
        self._closed = False

        prop = _HipMemAllocationProp()
        prop.type = (
            _HIP_MEM_ALLOCATION_TYPE_UNCACHED
            if uncached
            else _HIP_MEM_ALLOCATION_TYPE_PINNED
        )
        self.handle_type = (
            _HIP_MEM_HANDLE_TYPE_FABRIC
            if require_fabric
            else _resolve_handle_type(self.device.index)
        )
        use_fabric = self.handle_type == _HIP_MEM_HANDLE_TYPE_FABRIC
        prop.requestedHandleTypes = self.handle_type
        prop.location.type = _HIP_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device.index
        granularity = ctypes.c_size_t()
        _check(
            lib.hipMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(prop),
                _HIP_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            ),
            "hipMemGetAllocationGranularity",
        )
        gran = granularity.value
        self.size = (nbytes + gran - 1) // gran * gran

        # Every rank learns every rank's outcome before the handle exchange, so
        # a failure raises everywhere instead of leaving the peers blocked in it.
        local_fds: List[int] = []
        local_fabric: List[bytes] = []
        try:
            local, chunk_sizes = self._create(prop, gran)
            for handle in local:
                if use_fabric:
                    fabric = _HipMemFabricHandle()
                    _check(
                        lib.hipMemExportToShareableHandle(
                            ctypes.byref(fabric), handle, self.handle_type, 0
                        ),
                        "hipMemExportToShareableHandle(FABRIC)",
                    )
                    local_fabric.append(bytes(fabric.data))
                else:
                    fd = ctypes.c_int(-1)
                    _check(
                        lib.hipMemExportToShareableHandle(
                            ctypes.byref(fd), handle, self.handle_type, 0
                        ),
                        "hipMemExportToShareableHandle(POSIX_FD)",
                    )
                    local_fds.append(fd.value)
            outcome = (self.handle_type, chunk_sizes, local_fabric)
        except RuntimeError as e:
            outcome = str(e)
        outcomes: List[object] = [None] * self.world_size
        dist.all_gather_object(outcomes, outcome, group=group)
        errors = {r: o for r, o in enumerate(outcomes) if isinstance(o, str)}
        ok = [o for o in outcomes if not isinstance(o, str)]
        if errors or len({sum(o[1]) for o in ok}) != 1 or len({o[0] for o in ok}) != 1:
            for fd in local_fds:
                os.close(fd)
            self.close()
            raise RuntimeError(
                f"VMM allocation of {self.size} bytes failed: "
                f"{errors or [o[:2] for o in ok]}"
            )
        peer_sizes = [o[1] for o in outcomes]

        peer_fds = {}
        if not use_fabric:
            try:
                peer_fds = exchange_posix_fds(
                    group,
                    self.rank,
                    self.world_size,
                    local_fds,
                    [len(s) for s in peer_sizes],
                )
            finally:
                for fd in local_fds:
                    os.close(fd)

        ptrs: List[int] = []
        map_error: Optional[str] = None
        try:
            for peer in range(self.world_size):
                if peer == self.rank:
                    ptrs.append(self._map(local, chunk_sizes))
                    continue
                handles = []
                for idx in range(len(peer_sizes[peer])):
                    handle = ctypes.c_void_p()
                    if use_fabric:
                        fabric = _HipMemFabricHandle.from_buffer_copy(
                            outcomes[peer][2][idx]
                        )
                        _check(
                            lib.hipMemImportFromShareableHandle(
                                ctypes.byref(handle),
                                ctypes.cast(ctypes.byref(fabric), ctypes.c_void_p),
                                self.handle_type,
                            ),
                            f"hipMemImportFromShareableHandle(FABRIC, rank={peer})",
                        )
                    else:
                        peer_fd = peer_fds.pop((peer, idx))
                        try:
                            _check(
                                lib.hipMemImportFromShareableHandle(
                                    ctypes.byref(handle),
                                    ctypes.c_void_p(peer_fd),
                                    self.handle_type,
                                ),
                                f"hipMemImportFromShareableHandle(rank={peer})",
                            )
                        finally:
                            os.close(peer_fd)
                    self._handles.append(handle)
                    handles.append(handle)
                ptrs.append(self._map(handles, peer_sizes[peer]))
        except BaseException as e:
            for leftover in peer_fds.values():
                os.close(leftover)
            if not (require_fabric and isinstance(e, RuntimeError)):
                self.close()
                raise
            map_error = str(e)
        if require_fabric:
            # An import can fail on some ranks only (a peer outside this rank's
            # fabric clique); agree so no rank waits on a torn-down peer.
            map_errors: List[Optional[str]] = [None] * self.world_size
            dist.all_gather_object(map_errors, map_error, group=group)
            if any(error is not None for error in map_errors):
                self.close(group=group)
                failed = {r: e for r, e in enumerate(map_errors) if e is not None}
                raise RuntimeError(f"VMM peer mapping failed: {failed}")

        self.buffers: List[torch.Tensor] = []
        for ptr in ptrs:
            view = torch.as_tensor(
                _DeviceArray(self, ptr, self.size), device=self.device
            )
            if view.device != self.device or view.data_ptr() != ptr:
                raise RuntimeError(
                    f"VMM view landed on {view.device} @ {view.data_ptr():#x}, "
                    f"expected {self.device} @ {ptr:#x}"
                )
            self.buffers.append(view[:nbytes])
        if self.rank == 0:
            logger.info(
                "HIP VMM symmetric buffer: %d bytes x %d ranks, %s handles",
                self.size,
                self.world_size,
                "FABRIC" if use_fabric else "POSIX_FD",
            )

    def _create(
        self, prop: _HipMemAllocationProp, gran: int
    ) -> Tuple[List[ctypes.c_void_p], List[int]]:
        """``hipMemCreate`` ``self.size`` bytes, as several handles if need be.

        Some amdgpu drivers check a VA-less VRAM allocation's ``[0, size)``
        against the process's SVM ranges, so once something registers SVM
        memory low in the address space (e.g. Python's brk heap during RCCL
        init) every allocation reaching that address fails as out of memory.
        Smaller chunks, mapped back to back, stay under it.
        """
        lib = _hip()
        chunk = self.size
        while True:
            handles: List[ctypes.c_void_p] = []
            sizes: List[int] = []
            err = 0
            while sum(sizes) < self.size:
                n = min(chunk, self.size - sum(sizes))
                handle = ctypes.c_void_p()
                err = lib.hipMemCreate(ctypes.byref(handle), n, ctypes.byref(prop), 0)
                if err != 0:
                    break
                handles.append(handle)
                sizes.append(n)
            if err == 0:
                self._handles.extend(handles)
                self._num_local_handles = len(handles)
                if len(handles) > 1:
                    logger.info(
                        "VMM buffer of %d bytes split into %d chunks of <= %d bytes",
                        self.size,
                        len(handles),
                        chunk,
                    )
                return handles, sizes
            for handle in handles:
                lib.hipMemRelease(handle)
            if err != _HIP_ERROR_OUT_OF_MEMORY or chunk <= _MIN_CHUNK_BYTES:
                _check(err, f"hipMemCreate({chunk} of {self.size} bytes)")
            # HIP keeps the failure as the thread's last error, which torch's
            # next kernel-launch check would report as its own
            lib.hipGetLastError()
            chunk = max(_MIN_CHUNK_BYTES, chunk // 2 // gran * gran)

    def _map(self, handles: List[ctypes.c_void_p], sizes: List[int]) -> int:
        lib = _hip()
        va = ctypes.c_void_p()
        _check(
            lib.hipMemAddressReserve(ctypes.byref(va), self.size, 0, None, 0),
            "hipMemAddressReserve",
        )
        mapped: List[Tuple[int, int]] = []
        self._mapped.append((va.value, mapped))
        desc = _HipMemAccessDesc()
        desc.location.type = _HIP_MEM_LOCATION_TYPE_DEVICE
        desc.location.id = self.device.index
        desc.flags = _HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE
        offset = 0
        for handle, n in zip(handles, sizes):
            _check(lib.hipMemMap(va.value + offset, n, 0, handle, 0), "hipMemMap")
            mapped.append((va.value + offset, n))
            _check(
                lib.hipMemSetAccess(va.value + offset, n, ctypes.byref(desc), 1),
                "hipMemSetAccess",
            )
            offset += n
        return va.value

    def close(self, group: Optional[ProcessGroup] = None) -> None:
        """Unmap every view and release every handle.

        With ``group``, every rank releases its imports before any rank releases
        its own allocation, so no exporter revokes a FABRIC handle a peer still
        maps. If a peer never arrives, the own allocation is left to process exit.
        """
        if self._closed:
            return
        self._closed = True
        lib = _hip()
        for va, mapped in self._mapped:
            for ptr, n in mapped:
                err = lib.hipMemUnmap(ptr, n)
                if err != 0:
                    logger.warning("hipMemUnmap(%#x) -> %d", ptr, err)
            err = lib.hipMemAddressFree(va, self.size)
            if err != 0:
                logger.warning("hipMemAddressFree(%#x) -> %d", va, err)
        local = self._handles[: self._num_local_handles]
        for handle in self._handles[self._num_local_handles :]:
            err = lib.hipMemRelease(handle)
            if err != 0:
                logger.warning("hipMemRelease -> %d", err)
        if group is not None and not teardown_barrier(group):
            return
        for handle in local:
            err = lib.hipMemRelease(handle)
            if err != 0:
                logger.warning("hipMemRelease -> %d", err)
        self._mapped.clear()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
