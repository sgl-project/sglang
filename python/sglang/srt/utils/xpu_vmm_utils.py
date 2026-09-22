"""Level Zero VMM + IPC for Intel XPU, mirroring the cuda_vmm_utils surface.

Driven through ctypes against libze_loader; there is no compiled shim. The three
primitives DWDP needs map onto L0 as:

  cuMemAddressReserve/Free  -> zeVirtualMemReserve/Free
  cuMemCreate/Release       -> zePhysicalMemCreate/Destroy
  cuMemMap/SetAccess/Unmap  -> zeVirtualMemMap/SetAccessAttribute/Unmap

Handle sharing is POSIX-fd only, and therefore single-node: zeMemGetIpcHandle
rejects physical-memory handles and L0 has no FABRIC analogue.

Reservations live in a private L0 context, since SYCL exposes the handles behind
torch's queue only to C++. Intel's address space is per-device and shared across
contexts in a process, so torch kernels still read these mappings and the
allocator never hands back a reserved range; test_vmm_backend pins both.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
from functools import cache
from typing import List, Optional, Tuple

import torch
from torch.distributed import ProcessGroup

from sglang.srt.environ import envs
from sglang.srt.utils.vmm_common import (
    all_ranks_ok,
)

logger = logging.getLogger(__name__)

# Constants and struct layouts below are transcribed from the oneAPI Level Zero
# headers, github.com/oneapi-src/level-zero include/{ze,zes}_api.h, at ZE_API_VERSION
# 1.14; each is named for the enum or #define it was copied from.

_LOADER_SONAME = "libze_loader.so.1"

# zePhysicalMemGetProperties is the version gate: absent from loader 1.26.2, present
# in 1.28.2, and the only route to an exportable fd for a physical object.
_REQUIRED_SYMBOLS = (
    "zePhysicalMemCreate",
    "zePhysicalMemGetProperties",
    "zeVirtualMemMap",
    "zeVirtualMemQueryPageSize",
)

# Probed apart from the VMM symbols so missing telemetry degrades a memory query
# instead of failing a launch.
_SYSMAN_SYMBOLS = (
    "zesInit",
    "zesDriverGet",
    "zesDeviceGet",
    "zesDeviceGetProperties",
    "zesDeviceProcessesGetState",
)

# ze_result_t
_ZE_RESULT_SUCCESS = 0

# ze_structure_type_t
_ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3
_ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0xD
_ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC = 0x18
_ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD = 0x19
_ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD = 0x1A
_ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC = 0x20
_ZE_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES = 0x00020040  # added in 1.15

_ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD = 1  # ZE_BIT(0); L0's only physical-mem type
_ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE = 1

_ZE_MAX_DEVICE_NAME = 256  # #define; sizes a _DeviceProperties array

# zes_structure_type_t
_ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x1
_ZES_STRUCTURE_TYPE_PROCESS_STATE = 0x16

_ZES_STRING_PROPERTY_SIZE = 64  # #define; sizes _ZesDeviceProperties arrays

# ===== Chosen here, not read from the spec =====

# zesDeviceProcessesGetState counts then fills, and a rank starting up in between
# turns the second call into INVALID_SIZE; ask for room the count did not ask for.
_SYSMAN_PROCESS_SLACK = 8

# zeVirtualMemQueryPageSize grows with the request: measured 64 KiB below 2 MiB,
# 2 MiB from there to 64 GiB on Arc B60. Expert shards always sit past that step.
_PAGE_SIZE_QUERY_BYTES = 2 << 20

# Arbitrary; ldconfig -p only reads a cache, and it must not be what hangs a launch.
_LDCONFIG_TIMEOUT_S = 10.0


class _ContextDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class _PhysicalMemDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_size_t),
    ]


class _PhysicalMemProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("id", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]


class _ExternalMemoryExportDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class _ExternalMemoryExportFd(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("fd", ctypes.c_int),
    ]


class _ExternalMemoryImportFd(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("fd", ctypes.c_int),
    ]


class _DeviceProperties(ctypes.Structure):
    """Field order and widths must match ze_device_properties_t exactly; the
    driver writes ``uuid`` at a fixed offset from the struct base."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("vendorId", ctypes.c_uint32),
        ("deviceId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subdeviceId", ctypes.c_uint32),
        ("coreClockRate", ctypes.c_uint32),
        ("maxMemAllocSize", ctypes.c_uint64),
        ("maxHardwareContexts", ctypes.c_uint32),
        ("maxCommandQueuePriority", ctypes.c_uint32),
        ("numThreadsPerEU", ctypes.c_uint32),
        ("physicalEUSimdWidth", ctypes.c_uint32),
        ("numEUsPerSubslice", ctypes.c_uint32),
        ("numSubslicesPerSlice", ctypes.c_uint32),
        ("numSlices", ctypes.c_uint32),
        ("timerResolution", ctypes.c_uint64),
        ("timestampValidBits", ctypes.c_uint32),
        ("kernelTimestampValidBits", ctypes.c_uint32),
        ("uuid", ctypes.c_ubyte * 16),
        ("name", ctypes.c_char * _ZE_MAX_DEVICE_NAME),
    ]


def _missing_symbol(lib: ctypes.CDLL, symbols: Tuple[str, ...]) -> Optional[str]:
    for symbol in symbols:
        try:
            lib[symbol]
        except AttributeError:
            return symbol
    return None


def _loader_candidates() -> Tuple[str, ...]:
    # find_loaded_library reads /proc/self/maps, so it is device-neutral despite the
    # module it lives in; it names the copy torch mapped, which we reuse when it works.
    from sglang.srt.distributed.device_communicators.cuda_wrapper import (
        find_loaded_library,
    )

    candidates = (
        envs.SGLANG_XPU_ZE_LOADER_SO_PATH.get(),
        find_loaded_library("libze_loader"),
        *_ldconfig_paths(_LOADER_SONAME),
        _LOADER_SONAME,
    )
    return tuple(dict.fromkeys(path for path in candidates if path))


def _ldconfig_paths(soname: str) -> Tuple[str, ...]:
    # dlopen answers a soname with whatever copy is already mapped and find_library
    # reports a soname, not a path, so neither reaches a second install; the cache is
    # where the paths are. Invoked as CPython's ctypes.util._findSoname_ldconfig does.
    try:
        listing = subprocess.run(
            ["/sbin/ldconfig", "-p"],
            capture_output=True,
            text=True,
            env={"LC_ALL": "C", "LANG": "C", "PATH": "/sbin:/usr/sbin:/bin:/usr/bin"},
            timeout=_LDCONFIG_TIMEOUT_S,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()

    return tuple(
        line.rpartition("=>")[2].strip()
        for line in listing.splitlines()
        if "=>" in line and line.split("=>")[0].strip().startswith(soname)
    )


def _load_level_zero() -> ctypes.CDLL:
    return _load_level_zero_exporting(_REQUIRED_SYMBOLS)


@cache
def _load_level_zero_exporting(symbols: Tuple[str, ...]) -> ctypes.CDLL:
    # Load the first loader exporting `symbols`. torch maps whichever copy the distro
    # ships, and dlopen then answers the soname with that one, so probe absolute paths.
    rejected: List[str] = []
    for candidate in _loader_candidates():
        try:
            lib = ctypes.CDLL(candidate)
        except OSError as error:
            rejected.append(f"{candidate}: {error}")
            continue
        missing = _missing_symbol(lib, symbols)
        if missing is not None:
            rejected.append(f"{candidate}: no {missing}")
            continue
        check_ze(lib.zeInit(0), "zeInit")
        return lib

    raise ImportError(
        "XPU needs a Level Zero loader (>= 1.28) exporting "
        + ", ".join(symbols)
        + "; set SGLANG_XPU_ZE_LOADER_SO_PATH to one. Tried "
        + "; ".join(rejected)
    )


def check_ze(result: int, label: str) -> None:
    """Raise on a non-success ze_result_t."""
    if result != _ZE_RESULT_SUCCESS:
        raise RuntimeError(f"{label}: ze_result_t=0x{result & 0xFFFFFFFF:x}")


@cache
def _driver_and_devices():
    """Every (driver, device) pair keyed by the device's L0 UUID."""
    lib = _load_level_zero()
    count = ctypes.c_uint32(0)
    check_ze(lib.zeDriverGet(ctypes.byref(count), None), "zeDriverGet(count)")
    if count.value == 0:
        raise RuntimeError("no Level Zero drivers found")
    drivers = (ctypes.c_void_p * count.value)()
    check_ze(lib.zeDriverGet(ctypes.byref(count), drivers), "zeDriverGet")

    by_uuid = {}
    for driver in drivers[: count.value]:
        device_count = ctypes.c_uint32(0)
        check_ze(
            lib.zeDeviceGet(ctypes.c_void_p(driver), ctypes.byref(device_count), None),
            "zeDeviceGet(count)",
        )
        if device_count.value == 0:
            continue
        devices = (ctypes.c_void_p * device_count.value)()
        check_ze(
            lib.zeDeviceGet(
                ctypes.c_void_p(driver), ctypes.byref(device_count), devices
            ),
            "zeDeviceGet",
        )
        for device in devices[: device_count.value]:
            props = _DeviceProperties()
            props.stype = _ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
            check_ze(
                lib.zeDeviceGetProperties(ctypes.c_void_p(device), ctypes.byref(props)),
                "zeDeviceGetProperties",
            )
            by_uuid[bytes(props.uuid)] = (int(driver), int(device))
    return by_uuid


@cache
def _device_handle(device_id: int) -> int:
    """L0 device handle for a torch XPU ordinal, matched by UUID: ZE_AFFINITY_MASK
    and multi-driver systems permute index order, the UUID is what both agree on."""
    device_id = int(device_id)
    torch_uuid = bytes(torch.xpu.get_device_properties(device_id).uuid.bytes)
    entry = _driver_and_devices().get(torch_uuid)
    if entry is None:
        raise RuntimeError(
            f"torch XPU device {device_id} (uuid={torch_uuid.hex()}) has no "
            f"matching Level Zero device"
        )
    return entry[1]


@cache
def _driver_handle(device_id: int) -> int:
    torch_uuid = bytes(torch.xpu.get_device_properties(int(device_id)).uuid.bytes)
    return _driver_and_devices()[torch_uuid][0]


@cache
def _context_handle(device_id: int) -> int:
    """One L0 context per device, owning all reservations on it. Cached for the
    process lifetime; destroying it invalidates everything created through it."""
    lib = _load_level_zero()
    desc = _ContextDesc(_ZE_STRUCTURE_TYPE_CONTEXT_DESC, None, 0)
    context = ctypes.c_void_p()
    check_ze(
        lib.zeContextCreate(
            ctypes.c_void_p(_driver_handle(device_id)),
            ctypes.byref(desc),
            ctypes.byref(context),
        ),
        "zeContextCreate",
    )
    return int(context.value)


def tensor_from_pointer(
    pointer: int,
    nbytes: int,
    *,
    shape=None,
    dtype: torch.dtype = torch.uint8,
    device_id: int,
) -> torch.Tensor:
    """Use non-owning storage; the caller controls the underlying pages' lifetime."""
    device = torch.device("xpu", device_id)
    storage = torch._C._construct_storage_from_data_pointer(pointer, device, nbytes)
    if shape is None:
        shape = (nbytes,)
    return torch.empty(0, dtype=dtype, device=device).set_(storage, 0, shape)


class XpuAllocationProp:
    """Allocation policy for a device's physical objects.

    Only exportability varies; L0 expresses it by chaining
    ze_external_memory_export_desc_t onto the physical-memory descriptor.
    """

    def __init__(self, device_id: int, *, exportable: bool) -> None:
        self.device_id = int(device_id)
        self.exportable = bool(exportable)


def make_device_allocation_prop(
    device_id: int,
    *,
    handle_types: int | str | None = "auto",
    gpu_direct_rdma: bool = False,
):
    """Build an allocation prop; ``handle_types=None`` means non-exportable.
    ``gpu_direct_rdma`` has no L0 equivalent and is rejected, not dropped."""
    if gpu_direct_rdma:
        raise ValueError("gpu_direct_rdma is not supported for XPU physical memory")
    if handle_types not in ("auto", None):
        raise ValueError(
            "XPU physical memory only supports opaque-fd export; handle_types "
            "must be 'auto' or None"
        )
    return XpuAllocationProp(device_id, exportable=handle_types == "auto")


@cache
def query_page_size(device_id: int, size: int) -> int:
    """Page size Level Zero requires of a physical object of ``size`` bytes.

    Not a device constant: larger allocations report a coarser page, and a size
    that is not a multiple of its own reported page is rejected.
    """
    lib = _load_level_zero()
    device_id = int(device_id)
    page_size = ctypes.c_size_t(0)
    check_ze(
        lib.zeVirtualMemQueryPageSize(
            ctypes.c_void_p(_context_handle(device_id)),
            ctypes.c_void_p(_device_handle(device_id)),
            ctypes.c_size_t(int(size)),
            ctypes.byref(page_size),
        ),
        "zeVirtualMemQueryPageSize",
    )
    return int(page_size.value)


def get_device_granularity(device_id: int) -> int:
    """Page every reservation, offset and mapping size aligns to: the coarser of
    the two this driver reports, so one alignment holds at any object size."""
    return query_page_size(int(device_id), _PAGE_SIZE_QUERY_BYTES)


def create_physical_mem(size: int, prop: XpuAllocationProp) -> int:
    """Create a physical memory object, exportable per ``prop``."""
    lib = _load_level_zero()
    device_id = prop.device_id
    required_page = query_page_size(device_id, int(size))
    if int(size) % required_page:
        raise ValueError(
            f"physical size {size} is not a multiple of the {required_page} B page "
            f"Level Zero requires at that size; layouts align to "
            f"{get_device_granularity(device_id)} B"
        )
    desc = _PhysicalMemDesc(_ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, None, 0, int(size))
    export_desc = _ExternalMemoryExportDesc(
        _ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC,
        None,
        _ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
    )
    if prop.exportable:
        desc.pNext = ctypes.cast(ctypes.byref(export_desc), ctypes.c_void_p)
    handle = ctypes.c_void_p()
    check_ze(
        lib.zePhysicalMemCreate(
            ctypes.c_void_p(_context_handle(device_id)),
            ctypes.c_void_p(_device_handle(device_id)),
            ctypes.byref(desc),
            ctypes.byref(handle),
        ),
        "zePhysicalMemCreate",
    )
    return int(handle.value)


def release_physical_mem(handle: int, device_id: int) -> None:
    """Destroy a physical memory object. All mappings of it must be unmapped."""
    lib = _load_level_zero()
    check_ze(
        lib.zePhysicalMemDestroy(
            ctypes.c_void_p(_context_handle(int(device_id))), ctypes.c_void_p(handle)
        ),
        "zePhysicalMemDestroy",
    )


class VmmReservation:
    """Own a VA reservation, its mappings, and their teardown order."""

    def __init__(
        self,
        size: int,
        prop: XpuAllocationProp,
        device_id: int,
        *,
        alignment: int = 0,
        requested_address: int = 0,
    ) -> None:
        lib = _load_level_zero()
        self.size = int(size)
        self._prop = prop
        self._device_id = int(device_id)
        self._context = _context_handle(self._device_id)
        # zeVirtualMemReserve has no alignment argument and always returns a
        # page-aligned start; anything coarser would need over-reserving.
        granularity = get_device_granularity(self._device_id)
        if int(alignment) > granularity:
            raise ValueError(
                f"alignment {alignment} exceeds the XPU VMM page size {granularity}"
            )
        base = ctypes.c_void_p()
        check_ze(
            lib.zeVirtualMemReserve(
                ctypes.c_void_p(self._context),
                ctypes.c_void_p(int(requested_address)) if requested_address else None,
                ctypes.c_size_t(self.size),
                ctypes.byref(base),
            ),
            "zeVirtualMemReserve(local)",
        )
        self.base = int(base.value)
        self._mappings = []
        self._closed = False

    def map(self, offset: int, size: int, *, retain_handle: bool):
        """Create and map local memory at ``base + offset``."""
        if self._closed:
            raise RuntimeError("VmmReservation.map after close")
        offset, size = int(offset), int(size)
        if offset < 0 or size <= 0 or offset + size > self.size:
            raise ValueError(
                f"mapping [{offset}, {offset + size}) is outside reservation "
                f"[0, {self.size})"
            )

        address = self.base + offset
        handle = create_physical_mem(size, self._prop)
        try:
            self._map_physical(address, size, handle)
        except BaseException as error:
            try:
                release_physical_mem(handle, self._device_id)
            except BaseException as cleanup_error:
                error.add_note(f"XPU VMM rollback also failed: {cleanup_error!r}")
            raise

        # L0 has no cuMemRelease-style refcount: destroying a still-mapped object
        # tears down its pages, so a non-retained handle is freed only at close.
        self._mappings.append((address, size, handle, not retain_handle))
        return handle

    def map_existing(self, offset: int, size: int, handle) -> None:
        """Map a caller-owned physical allocation into this reservation."""
        if self._closed:
            raise RuntimeError("VmmReservation.map_existing after close")
        offset, size = int(offset), int(size)
        address = self.base + offset
        # zeVirtualMemMap maps the whole range or nothing, so a failure leaves
        # no mapping to roll back.
        self._map_physical(address, size, int(handle))
        self._mappings.append((address, size, None, False))

    def unmap_existing(self, offset: int, size: int) -> None:
        """Undo one ``map_existing`` so its handle can be mapped somewhere else; a
        physical object may sit in only one live mapping on this backend."""
        if self._closed:
            raise RuntimeError("VmmReservation.unmap_existing after close")
        offset, size = int(offset), int(size)
        record = (self.base + offset, size, None, False)
        if record not in self._mappings:
            raise ValueError(
                f"[{offset}, {offset + size}) is not an aliased mapping of this "
                f"reservation"
            )
        self._mappings.remove(record)
        self._unmap(record[0], size)

    def _map_physical(self, address: int, size: int, handle: int) -> None:
        lib = _load_level_zero()
        check_ze(
            lib.zeVirtualMemMap(
                ctypes.c_void_p(self._context),
                ctypes.c_void_p(address),
                ctypes.c_size_t(size),
                ctypes.c_void_p(handle),
                ctypes.c_size_t(0),
                ctypes.c_uint32(_ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE),
            ),
            "zeVirtualMemMap",
        )

    def _unmap(self, address: int, size: int) -> None:
        lib = _load_level_zero()
        check_ze(
            lib.zeVirtualMemUnmap(
                ctypes.c_void_p(self._context),
                ctypes.c_void_p(address),
                ctypes.c_size_t(size),
            ),
            "zeVirtualMemUnmap",
        )

    def close(self, *, release_handles: bool = True) -> None:
        """Unmap allocations, optionally release retained handles, and free VA.
        ``release_handles=False`` spares only handles ``map`` handed back."""
        if self._closed:
            return
        self._closed = True
        lib = _load_level_zero()
        while self._mappings:
            address, size, handle, privately_owned = self._mappings.pop()
            try:
                self._unmap(address, size)
            except RuntimeError as error:
                logger.warning("%s", error)
            if handle is not None and (release_handles or privately_owned):
                try:
                    release_physical_mem(handle, self._device_id)
                except RuntimeError as error:
                    logger.warning("%s", error)
        result = lib.zeVirtualMemFree(
            ctypes.c_void_p(self._context),
            ctypes.c_void_p(self.base),
            ctypes.c_size_t(self.size),
        )
        if result != _ZE_RESULT_SUCCESS:
            logger.warning("zeVirtualMemFree(local) -> 0x%x", result & 0xFFFFFFFF)


def export_shareable_handles(
    retained_handles, group: ProcessGroup, rank: int, *, device_id: int
):
    """Export retained physical-memory handles as POSIX fds.

    Returns the CUDA backend's ``(fabric_handles, posix_fds, use_fabric)`` shape;
    ``use_fabric`` is always False, as L0 has no fabric handle type. The fds come
    from zePhysicalMemGetProperties and are the driver's -- do not close them.
    """
    lib = _load_level_zero()
    context = _context_handle(int(device_id))
    posix_fds: List[int] = []
    error: Optional[Exception] = None
    try:
        for handle in retained_handles:
            export_fd = _ExternalMemoryExportFd(
                _ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD,
                None,
                _ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
                -1,
            )
            props = _PhysicalMemProperties()
            props.stype = _ZE_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES
            props.pNext = ctypes.cast(ctypes.byref(export_fd), ctypes.c_void_p)
            check_ze(
                lib.zePhysicalMemGetProperties(
                    ctypes.c_void_p(context),
                    ctypes.c_void_p(int(handle)),
                    ctypes.byref(props),
                ),
                "zePhysicalMemGetProperties(EXPORT_FD)",
            )
            if export_fd.fd < 0:
                raise RuntimeError(
                    "zePhysicalMemGetProperties returned no exportable fd; the "
                    "physical object was not created with an export descriptor"
                )
            posix_fds.append(int(export_fd.fd))
        ok = True
    except Exception as e:
        error = e
        ok = False
        posix_fds = []

    if not all_ranks_ok(group, ok):
        message = (
            "XPU VMM handle export failed: opaque-fd export failed on at least one rank"
        )
        if error is not None:
            message += f"; local rank {rank} error: {error}"
        raise RuntimeError(message) from error

    return [], posix_fds, False


def import_peer_handle(
    fabric_handle, fd, *, use_fabric: bool, peer_rank: int, device_id: int, size: int
) -> int:
    """Import a peer physical-memory object from a POSIX fd, duping it so the
    caller keeps the original.

    An opaque fd does not carry the object's size, so ``size`` must be the
    exporter's page-aligned size; a mismatch comes back as UNSUPPORTED_SIZE.
    """
    if use_fabric:
        raise ValueError("XPU has no FABRIC handle type")
    lib = _load_level_zero()
    device_id = int(device_id)
    dup_fd = os.dup(fd)
    import_fd = _ExternalMemoryImportFd(
        _ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
        None,
        _ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
        dup_fd,
    )
    desc = _PhysicalMemDesc(_ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, None, 0, int(size))
    desc.pNext = ctypes.cast(ctypes.byref(import_fd), ctypes.c_void_p)
    handle = ctypes.c_void_p()
    try:
        check_ze(
            lib.zePhysicalMemCreate(
                ctypes.c_void_p(_context_handle(device_id)),
                ctypes.c_void_p(_device_handle(device_id)),
                ctypes.byref(desc),
                ctypes.byref(handle),
            ),
            f"zePhysicalMemCreate(IMPORT_FD, rank={peer_rank})",
        )
    finally:
        try:
            os.close(dup_fd)
        except OSError:
            pass
    return int(handle.value)


def can_access_peer(device_id: int, peer_device_id: int) -> bool:
    """Whether ``device_id`` can read allocations owned by ``peer_device_id``."""
    lib = _load_level_zero()
    value = ctypes.c_uint8(0)
    check_ze(
        lib.zeDeviceCanAccessPeer(
            ctypes.c_void_p(_device_handle(int(device_id))),
            ctypes.c_void_p(_device_handle(int(peer_device_id))),
            ctypes.byref(value),
        ),
        "zeDeviceCanAccessPeer",
    )
    return bool(value.value)


class _ZesProcessState(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("processId", ctypes.c_uint32),
        ("memSize", ctypes.c_uint64),
        ("sharedSize", ctypes.c_uint64),
        ("engines", ctypes.c_uint32),
    ]


class _ZesDeviceProperties(ctypes.Structure):
    """``core`` is a whole ze_device_properties_t by value, so the trailing fields
    only land where the driver writes them if that nested layout is exact too."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("core", _DeviceProperties),
        ("numSubdevices", ctypes.c_uint32),
        ("serialNumber", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("boardNumber", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("brandName", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("modelName", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("vendorName", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("driverVersion", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
    ]


@cache
def _sysman_device_handle(device_id: int) -> int:
    """Sysman handle for a torch XPU ordinal, matched by UUID: sysman enumerates
    independently of the core API, so index order is no contract here either."""
    lib = _load_level_zero_exporting(_SYSMAN_SYMBOLS)
    check_ze(lib.zesInit(0), "zesInit")
    torch_uuid = bytes(torch.xpu.get_device_properties(int(device_id)).uuid.bytes)

    driver_count = ctypes.c_uint32(0)
    check_ze(lib.zesDriverGet(ctypes.byref(driver_count), None), "zesDriverGet(count)")
    drivers = (ctypes.c_void_p * driver_count.value)()
    check_ze(lib.zesDriverGet(ctypes.byref(driver_count), drivers), "zesDriverGet")

    for driver in drivers[: driver_count.value]:
        device_count = ctypes.c_uint32(0)
        check_ze(
            lib.zesDeviceGet(ctypes.c_void_p(driver), ctypes.byref(device_count), None),
            "zesDeviceGet(count)",
        )
        devices = (ctypes.c_void_p * device_count.value)()
        check_ze(
            lib.zesDeviceGet(
                ctypes.c_void_p(driver), ctypes.byref(device_count), devices
            ),
            "zesDeviceGet",
        )
        for device in devices[: device_count.value]:
            props = _ZesDeviceProperties()
            props.stype = _ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES
            props.core.stype = _ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
            check_ze(
                lib.zesDeviceGetProperties(
                    ctypes.c_void_p(device), ctypes.byref(props)
                ),
                "zesDeviceGetProperties",
            )
            if bytes(props.core.uuid) == torch_uuid:
                return int(device)

    raise RuntimeError(
        f"torch XPU device {device_id} (uuid={torch_uuid.hex()}) has no matching "
        f"Level Zero sysman device"
    )


def get_device_memory_in_use(device_id: int) -> int:
    """Device memory held by every process on ``device_id``, in bytes.

    The only query on this driver that sees L0 physical memory: on Arc B60
    zesMemoryGetState reports free == total, currUsableMemSize reports 0, and
    torch.xpu.mem_get_info forwards the former. ~35 ms (the driver walks /proc),
    so use it on startup and sizing paths, never per step.
    """
    lib = _load_level_zero_exporting(_SYSMAN_SYMBOLS)
    device = _sysman_device_handle(int(device_id))

    count = ctypes.c_uint32(0)
    check_ze(
        lib.zesDeviceProcessesGetState(
            ctypes.c_void_p(device), ctypes.byref(count), None
        ),
        "zesDeviceProcessesGetState(count)",
    )
    if count.value == 0:
        return 0

    capacity = ctypes.c_uint32(count.value + _SYSMAN_PROCESS_SLACK)
    states = (_ZesProcessState * capacity.value)()
    for state in states:
        state.stype = _ZES_STRUCTURE_TYPE_PROCESS_STATE
    check_ze(
        lib.zesDeviceProcessesGetState(
            ctypes.c_void_p(device), ctypes.byref(capacity), states
        ),
        "zesDeviceProcessesGetState",
    )
    return sum(int(state.memSize) for state in states[: capacity.value])
