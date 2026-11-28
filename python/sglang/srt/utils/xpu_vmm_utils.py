# SPDX-License-Identifier: Apache-2.0
"""Level Zero virtual memory management for Intel XPU.

The XPU counterpart of ``cuda_vmm_utils`` and ``hip_vmm_utils``: it splits a
device allocation into a physical part and a virtual part, so one physical
allocation can be mapped at an address each process chooses for itself. That is
what lets a weight cache hand the same device memory to an engine without
copying it, and without the engine having to accept the daemon's addresses.

Three objects, in the order a caller uses them:

- ``PhysicalMemory.create`` makes a ``zePhysicalMemCreate`` allocation. Declaring
  it exportable happens here, at creation, so no later call can make the memory
  unexportable.
- ``VirtualReservation`` reserves a virtual range and maps physical allocations
  into it, then sets page access attributes.
- ``PhysicalMemory.export_dma_buf`` hands out a dma-buf fd that ``SCM_RIGHTS``
  carries over a unix socket; ``PhysicalMemory.from_dma_buf`` rebuilds a physical
  allocation from an fd a peer received. No ``pidfd_getfd``, so no ptrace
  permission is needed.

ctypes over ``libze_loader.so.1`` rather than a build-time extension, matching
``cuda_vmm_utils`` and ``hip_vmm_utils``: no new compiled dependency and no
sgl-kernel-xpu dependency, and the loader is present wherever a torch-xpu build
runs.

Measured on 8x Intel Arc Pro B60 (device 0xE211, driver 1.15.39122+11, L0 loader
1.28.0), the whole path works, including dma-buf export and import of a
``zePhysicalMemCreate`` allocation. Five facts from that hardware, none of them
promised by the spec, shape the code:

- ``zePhysicalMemGetProperties`` reports ``id == 0`` for every allocation, so the
  id cannot identify an allocation. ``dma_buf_identity`` reads the dma-buf's
  ``(st_dev, st_ino)`` instead, which is the kernel's own identity for the
  buffer and equal in two processes exactly when they hold the same one.
- Device page size is 64 KiB, not the 4 KiB host page size, and reserve, map and
  unmap all reject an unaligned size. Callers go through ``query_page_size``.
- ``zePhysicalMemDestroy`` succeeds on an allocation that is still mapped,
  rather than answering ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE as the spec
  permits. The stale mapping then reads changed data and faults, so
  ``PhysicalMemory.close`` counts live mappings and refuses instead.
- ``zePhysicalMemCreate`` succeeds well past the card's capacity -- 256 GiB on a
  24 GiB B60 -- so a successful create is not a capacity check. It answers
  ERROR_OUT_OF_DEVICE_MEMORY from 512 GiB up to 16 TiB, and above 16 TiB calls
  abort(); ``create`` refuses that range rather than letting the process die.
- One physical handle carries one live mapping: a second ``zeVirtualMemMap`` of
  the same handle, to any address, answers ZE_RESULT_ERROR_INVALID_ARGUMENT
  until the first range is unmapped. A non-zero ``physical_offset`` is fine, and
  so is remapping after an unmap. Sharing is unaffected, because exporter and
  importer each hold their own handle for the one buffer; packing several
  tensors behind one allocation must map it once and sub-slice the address.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from typing import List, Optional, Tuple

import msgspec

logger = logging.getLogger(__name__)

_LEVEL_ZERO_LOADER = "libze_loader.so.1"

ZE_RESULT_SUCCESS = 0

ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0x0D
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0x0E
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC = 0x18
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD = 0x19
ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD = 0x1A
ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC = 0x20
# Extension-range stype, unlike the others here; zePhysicalMemGetProperties
# arrived later than the rest of the VMM API.
ZE_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES = 0x00020040

ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF = 1 << 1

ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE = 1 << 0

ACCESS_NONE = 0
ACCESS_READWRITE = 1
ACCESS_READONLY = 2

ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1

_UINT64_MAX = (1 << 64) - 1

_VP = ctypes.c_void_p
_SZ = ctypes.c_size_t
_SIZE_T_MAX = (1 << (8 * ctypes.sizeof(_SZ))) - 1

# Largest size zePhysicalMemCreate answers rather than aborting. Bisected on Arc
# Pro B60: exactly 16 TiB returns ERROR_OUT_OF_DEVICE_MEMORY, one page more
# calls abort() from gmm_helper/resource_info.cpp and takes the process down.
_PHYSICAL_MEM_MAX_BYTES = 1 << 44

_RESULT_NAMES = {
    0x70000001: "ERROR_DEVICE_LOST",
    0x70000002: "ERROR_OUT_OF_HOST_MEMORY",
    0x70000003: "ERROR_OUT_OF_DEVICE_MEMORY",
    0x70010000: "ERROR_INSUFFICIENT_PERMISSIONS",
    0x70010001: "ERROR_NOT_AVAILABLE",
    0x70020000: "ERROR_DEPENDENCY_UNAVAILABLE",
    0x78000001: "ERROR_UNINITIALIZED",
    0x78000002: "ERROR_UNSUPPORTED_VERSION",
    0x78000003: "ERROR_UNSUPPORTED_FEATURE",
    0x78000004: "ERROR_INVALID_ARGUMENT",
    0x78000005: "ERROR_INVALID_NULL_HANDLE",
    0x78000006: "ERROR_HANDLE_OBJECT_IN_USE",
    0x78000007: "ERROR_INVALID_NULL_POINTER",
    0x78000008: "ERROR_INVALID_SIZE",
    0x78000009: "ERROR_UNSUPPORTED_SIZE",
    0x7800000A: "ERROR_UNSUPPORTED_ALIGNMENT",
    0x7800000C: "ERROR_INVALID_ENUMERATION",
    0x7800000D: "ERROR_UNSUPPORTED_ENUMERATION",
    0x7FFFFFFE: "ERROR_UNKNOWN",
}


class ZeVmmError(RuntimeError):
    """A Level Zero call returned a non-success ``ze_result_t``."""


def result_name(rc: int) -> str:
    return _RESULT_NAMES.get(rc & 0xFFFFFFFF, "UNRECOGNISED_ZE_RESULT")


def check(name: str, rc: int) -> None:
    if rc != ZE_RESULT_SUCCESS:
        raise ZeVmmError(
            f"{name} failed with ze_result_t 0x{rc & 0xFFFFFFFF:08x} "
            f"({result_name(rc)})"
        )


# ------------------------------------------------------------------ ABI structs


class _ContextDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int), ("pNext", _VP), ("flags", ctypes.c_uint32)]


class _CommandQueueDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", _VP),
        ("ordinal", ctypes.c_uint32),
        ("index", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("mode", ctypes.c_int),
        ("priority", ctypes.c_int),
    ]


class _PhysicalMemDesc(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", _VP),
        ("flags", ctypes.c_uint32),
        ("size", _SZ),
    ]


class _PhysicalMemProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", _VP),
        ("id", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]


class _ExternalMemoryExportDesc(ctypes.Structure):
    _fields_ = [("stype", ctypes.c_int), ("pNext", _VP), ("flags", ctypes.c_uint32)]


class _ExternalMemoryExportFd(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", _VP),
        ("flags", ctypes.c_uint32),
        ("fd", ctypes.c_int),
    ]


class _ExternalMemoryImportFd(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", _VP),
        ("flags", ctypes.c_uint32),
        ("fd", ctypes.c_int),
    ]


# ------------------------------------------------------------------- the driver


def _bind(lib: ctypes.CDLL) -> None:
    lib.zeInit.argtypes = [ctypes.c_uint32]
    lib.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(_VP)]
    lib.zeDeviceGet.argtypes = [
        _VP,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(_VP),
    ]
    lib.zeContextCreate.argtypes = [
        _VP,
        ctypes.POINTER(_ContextDesc),
        ctypes.POINTER(_VP),
    ]
    lib.zeContextDestroy.argtypes = [_VP]

    lib.zeVirtualMemQueryPageSize.argtypes = [_VP, _VP, _SZ, ctypes.POINTER(_SZ)]
    lib.zeVirtualMemReserve.argtypes = [_VP, _VP, _SZ, ctypes.POINTER(_VP)]
    lib.zeVirtualMemFree.argtypes = [_VP, _VP, _SZ]
    lib.zeVirtualMemMap.argtypes = [_VP, _VP, _SZ, _VP, _SZ, ctypes.c_int]
    lib.zeVirtualMemUnmap.argtypes = [_VP, _VP, _SZ]
    lib.zeVirtualMemSetAccessAttribute.argtypes = [_VP, _VP, _SZ, ctypes.c_int]
    lib.zeVirtualMemGetAccessAttribute.argtypes = [
        _VP,
        _VP,
        _SZ,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(_SZ),
    ]

    # desc is passed as a plain void* because the export and import variants
    # chain different structures onto pNext.
    lib.zePhysicalMemCreate.argtypes = [_VP, _VP, _VP, ctypes.POINTER(_VP)]
    lib.zePhysicalMemDestroy.argtypes = [_VP, _VP]
    lib.zePhysicalMemGetProperties.argtypes = [
        _VP,
        _VP,
        ctypes.POINTER(_PhysicalMemProperties),
    ]

    lib.zeCommandListCreateImmediate.argtypes = [
        _VP,
        _VP,
        ctypes.POINTER(_CommandQueueDesc),
        ctypes.POINTER(_VP),
    ]
    lib.zeCommandListAppendMemoryCopy.argtypes = [
        _VP,
        _VP,
        _VP,
        _SZ,
        _VP,
        ctypes.c_uint32,
        ctypes.POINTER(_VP),
    ]
    lib.zeCommandListHostSynchronize.argtypes = [_VP, ctypes.c_uint64]
    lib.zeCommandListDestroy.argtypes = [_VP]


class _Driver:
    """Process-wide Level Zero driver, device list and context.

    One context for the process, because a physical allocation and the virtual
    range it is mapped into must belong to the same context. When #45742 lands
    this bootstrap should be shared with ``xpu_ipc_utils`` rather than repeated.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lib: Optional[ctypes.CDLL] = None
        self._devices: Tuple[int, ...] = ()
        self._context: Optional[_VP] = None

    def load(self) -> ctypes.CDLL:
        with self._lock:
            if self._lib is not None:
                return self._lib

            lib = ctypes.CDLL(_LEVEL_ZERO_LOADER)
            _bind(lib)
            check("zeInit", lib.zeInit(0))

            count = ctypes.c_uint32(0)
            check("zeDriverGet", lib.zeDriverGet(ctypes.byref(count), None))
            if count.value == 0:
                raise ZeVmmError("no Level Zero driver found")
            drivers = (_VP * count.value)()
            check("zeDriverGet", lib.zeDriverGet(ctypes.byref(count), drivers))
            driver = drivers[0]

            devices = self._enumerate_devices(lib, driver)

            desc = _ContextDesc(ZE_STRUCTURE_TYPE_CONTEXT_DESC, None, 0)
            context = _VP()
            check(
                "zeContextCreate",
                lib.zeContextCreate(
                    _VP(driver), ctypes.byref(desc), ctypes.byref(context)
                ),
            )

            self._devices = devices
            self._context = context
            self._lib = lib
            return lib

    @staticmethod
    def _enumerate_devices(lib: ctypes.CDLL, driver) -> Tuple[int, ...]:
        count = ctypes.c_uint32(0)
        check("zeDeviceGet", lib.zeDeviceGet(_VP(driver), ctypes.byref(count), None))
        if count.value == 0:
            raise ZeVmmError("Level Zero driver reports no devices")
        handles = (_VP * count.value)()
        check("zeDeviceGet", lib.zeDeviceGet(_VP(driver), ctypes.byref(count), handles))
        return tuple(handles[i] for i in range(count.value))

    @property
    def lib(self) -> ctypes.CDLL:
        return self.load()

    @property
    def context(self):
        self.load()
        return self._context

    def device(self, index: int):
        self.load()
        if not 0 <= index < len(self._devices):
            raise ZeVmmError(
                f"device_index {index} out of range; the Level Zero driver "
                f"reports {len(self._devices)} device(s). ZE_AFFINITY_MASK "
                f"restricts this list."
            )
        return _VP(self._devices[index])

    def device_count(self) -> int:
        self.load()
        return len(self._devices)


_driver = _Driver()


def device_count() -> int:
    return _driver.device_count()


# -------------------------------------------------------------------- page size


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def _as_size_t(name: str, value: int) -> int:
    """Reject a byte count that does not fit in ``size_t``.

    ctypes truncates an out-of-range ``c_size_t`` modulo 2**64 without
    complaining, and the truncation is not safely absorbed downstream: a
    reservation of 2**64 bytes arrives as 0, ``zeVirtualMemReserve`` returns
    success, and the driver then aborts the process from
    ``gmm_helper/resource_info.cpp``.
    """
    if not 0 <= value <= _SIZE_T_MAX:
        raise ValueError(f"{name} {value} does not fit in size_t (0..{_SIZE_T_MAX})")
    return value


def query_page_size(size: int, *, device_index: int = 0) -> int:
    """Page size to align a reservation and a physical allocation of ``size`` to.

    64 KiB on Arc Pro B60, against a 4 KiB host page; reserve, map and unmap all
    reject a size that is not a multiple of it.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    _as_size_t("size", size)
    lib = _driver.lib
    pagesize = _SZ(0)
    check(
        "zeVirtualMemQueryPageSize",
        lib.zeVirtualMemQueryPageSize(
            _driver.context,
            _driver.device(device_index),
            _SZ(size),
            ctypes.byref(pagesize),
        ),
    )
    if pagesize.value == 0:
        raise ZeVmmError("zeVirtualMemQueryPageSize reported a page size of 0")
    return pagesize.value


def aligned_size(size: int, *, device_index: int = 0) -> int:
    """``size`` rounded up to the device page size."""
    return align_up(size, query_page_size(size, device_index=device_index))


# -------------------------------------------------------------- dma-buf identity


class DmaBufIdentity(msgspec.Struct, frozen=True):
    """The kernel's identity for a dma-buf, as seen through one fd.

    Equal in exporter and importer exactly when both hold the same buffer, which
    is what makes a zero-copy claim checkable rather than inferred from matching
    bytes. ``zePhysicalMemGetProperties`` cannot serve here: on Arc Pro B60 it
    reports ``id == 0`` for every allocation.
    """

    st_dev: int
    st_ino: int
    size: int


def dma_buf_identity(fd: int) -> DmaBufIdentity:
    if fd < 0:
        raise ValueError(f"fd must be non-negative, got {fd}")
    st = os.fstat(fd)
    return DmaBufIdentity(st_dev=st.st_dev, st_ino=st.st_ino, size=st.st_size)


# ----------------------------------------------------------- physical memory


class PhysicalMemory:
    """One ``zePhysicalMemCreate`` allocation.

    Owns the handle and destroys it in ``close``. Three fd ownership rules,
    each measured on Arc Pro B60 rather than taken from the spec, because the
    spec does not state any of them:

    - ``zePhysicalMemGetProperties(+export_fd)`` returns the *same* fd number on
      every call, so this class queries it once and keeps it. Handing that fd
      out directly would make two callers share one fd and close it twice.
      ``export_dma_buf`` returns a ``dup`` instead, which each caller owns.
    - ``zePhysicalMemDestroy`` does not close the exported fd. Left alone it
      leaks one fd per allocation, so ``close`` closes the retained one.
    - ``zePhysicalMemCreate(+import_fd)`` takes ownership of the fd it is given
      and closes it at destroy. ``from_dma_buf`` therefore imports a ``dup`` and
      leaves the caller's fd alone.
    """

    def __init__(
        self,
        handle,
        *,
        size: int,
        device_index: int,
        page_size: int,
        exportable: bool = False,
    ):
        self._handle = handle
        self._size = size
        self._device_index = device_index
        self._page_size = page_size
        self._exportable = exportable
        # The driver's dma-buf fd, queried on first export and closed by us.
        self._export_fd: Optional[int] = None
        self._mapped_refs = 0
        self._closed = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def exportable(self) -> bool:
        return self._exportable

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def mapped_count(self) -> int:
        """How many live mappings refer to this allocation."""
        return self._mapped_refs

    def _acquire_mapping(self) -> None:
        self._mapped_refs += 1

    def _release_mapping(self) -> None:
        self._mapped_refs -= 1

    @property
    def handle(self):
        if self._closed:
            raise ZeVmmError("physical allocation is closed")
        return self._handle

    @classmethod
    def create(
        cls, size: int, *, device_index: int = 0, exportable: bool = True
    ) -> PhysicalMemory:
        """Allocate physical device memory, page-aligning ``size`` upward.

        ``exportable`` is declared here because the Level Zero contract requires
        the export type to be requested at creation; a plain allocation cannot be
        exported afterwards.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        lib = _driver.lib
        page_size = query_page_size(size, device_index=device_index)
        padded = _as_size_t("page-aligned size", align_up(size, page_size))
        if padded > _PHYSICAL_MEM_MAX_BYTES:
            raise ValueError(
                f"size {padded} exceeds {_PHYSICAL_MEM_MAX_BYTES}, above which "
                f"this driver aborts the process instead of returning an error"
            )

        export = _ExternalMemoryExportDesc(
            ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC,
            None,
            ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
        )
        desc = _PhysicalMemDesc(
            ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC,
            ctypes.cast(ctypes.byref(export), _VP) if exportable else None,
            ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE,
            _SZ(padded),
        )
        handle = _VP()
        check(
            "zePhysicalMemCreate",
            lib.zePhysicalMemCreate(
                _driver.context,
                _driver.device(device_index),
                ctypes.cast(ctypes.byref(desc), _VP),
                ctypes.byref(handle),
            ),
        )
        return cls(
            handle,
            size=padded,
            device_index=device_index,
            page_size=page_size,
            exportable=exportable,
        )

    @classmethod
    def from_dma_buf(
        cls, fd: int, size: int, *, device_index: int = 0
    ) -> PhysicalMemory:
        """Rebuild a peer's physical allocation from a dma-buf fd.

        ``fd`` stays the caller's to close. The driver takes ownership of the fd
        handed to ``zePhysicalMemCreate``, so what it receives is a ``dup``.
        """
        if fd < 0:
            raise ValueError(f"fd must be non-negative, got {fd}")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        lib = _driver.lib
        page_size = query_page_size(size, device_index=device_index)
        _as_size_t("size", size)
        if size % page_size:
            raise ValueError(
                f"size {size} is not a multiple of the device page size "
                f"{page_size}; the exporter padded its allocation and must send "
                f"the padded size"
            )

        owned_fd = os.dup(fd)
        try:
            import_fd = _ExternalMemoryImportFd(
                ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
                None,
                ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
                owned_fd,
            )
            desc = _PhysicalMemDesc(
                ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC,
                ctypes.cast(ctypes.byref(import_fd), _VP),
                ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE,
                _SZ(size),
            )
            handle = _VP()
            check(
                "zePhysicalMemCreate(import_fd)",
                lib.zePhysicalMemCreate(
                    _driver.context,
                    _driver.device(device_index),
                    ctypes.cast(ctypes.byref(desc), _VP),
                    ctypes.byref(handle),
                ),
            )
        except BaseException:
            # The import failed, so the driver never took the dup over.
            os.close(owned_fd)
            raise
        return cls(handle, size=size, device_index=device_index, page_size=page_size)

    def export_dma_buf(self) -> int:
        """Return a dma-buf fd for this allocation; the caller closes it.

        Every call returns an independent fd onto the same buffer, so one
        allocation can be served to several peers.
        """
        if not self._exportable:
            raise ZeVmmError(
                "this allocation was created with exportable=False; Level Zero "
                "requires the dma-buf export type at creation time"
            )
        if self._export_fd is None:
            self._export_fd = self._query_export_fd()
        return os.dup(self._export_fd)

    def _query_export_fd(self) -> int:
        lib = _driver.lib
        export_fd = _ExternalMemoryExportFd(
            ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD,
            None,
            ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
            -1,
        )
        props = _PhysicalMemProperties(
            ZE_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES,
            ctypes.cast(ctypes.byref(export_fd), _VP),
            0,
            0,
        )
        check(
            "zePhysicalMemGetProperties(export_fd)",
            lib.zePhysicalMemGetProperties(
                _driver.context, self.handle, ctypes.byref(props)
            ),
        )
        if export_fd.fd < 0:
            raise ZeVmmError(
                "zePhysicalMemGetProperties succeeded but handed out no dma-buf "
                "fd; this driver cannot export a zePhysicalMemCreate allocation, "
                "so VMM sharing is unavailable"
            )
        return int(export_fd.fd)

    def reported_size(self) -> int:
        """``size`` as the driver reports it, which should match ``self.size``."""
        lib = _driver.lib
        props = _PhysicalMemProperties(
            ZE_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES, None, 0, 0
        )
        check(
            "zePhysicalMemGetProperties",
            lib.zePhysicalMemGetProperties(
                _driver.context, self.handle, ctypes.byref(props)
            ),
        )
        return int(props.size)

    def close(self) -> None:
        """Destroy the allocation and reclaim the fd the driver would leak.

        Refuses while a mapping still refers to it. Level Zero makes that the
        caller's responsibility and allows the driver to answer
        ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE, but on Arc Pro B60 the destroy
        succeeds instead: a read through the stale mapping then returns changed
        data and the next one faults. Checking here turns that into an error.
        """
        if self._closed:
            return
        if self._mapped_refs:
            raise ZeVmmError(
                f"refusing to destroy a physical allocation with "
                f"{self._mapped_refs} live mapping(s); close the "
                f"VirtualReservation first"
            )
        # Marked closed before the call, so a failure cannot leave a handle that
        # a later close would destroy twice.
        self._closed = True
        try:
            check(
                "zePhysicalMemDestroy",
                _driver.lib.zePhysicalMemDestroy(_driver.context, self._handle),
            )
        finally:
            if self._export_fd is not None:
                os.close(self._export_fd)
                self._export_fd = None

    def __enter__(self) -> PhysicalMemory:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ------------------------------------------------------------ virtual memory


class _Mapping(msgspec.Struct, frozen=True):
    offset: int
    size: int
    # Keeps the allocation reachable and counted for as long as this mapping
    # refers to it; see PhysicalMemory.close.
    physical: PhysicalMemory


class VirtualReservation:
    """A reserved virtual address range and the mappings made into it.

    Owns the reservation and unmaps every range it mapped before freeing it.
    It does not own the physical allocations: ``close`` leaves them alive, which
    is what the Level Zero contract requires and what lets an importer drop its
    view while the exporter keeps the memory.
    """

    def __init__(self, size: int, *, device_index: int = 0):
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        page_size = query_page_size(size, device_index=device_index)
        padded = _as_size_t("page-aligned size", align_up(size, page_size))

        ptr = _VP()
        check(
            "zeVirtualMemReserve",
            _driver.lib.zeVirtualMemReserve(
                _driver.context, None, _SZ(padded), ctypes.byref(ptr)
            ),
        )
        self._ptr = ctypes.cast(ptr, _VP).value
        self._size = padded
        self._page_size = page_size
        self._device_index = device_index
        self._mappings: List[_Mapping] = []
        self._closed = False

    @property
    def ptr(self) -> int:
        if self._closed:
            raise ZeVmmError("virtual reservation is closed")
        return self._ptr

    @property
    def size(self) -> int:
        return self._size

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def num_mappings(self) -> int:
        return len(self._mappings)

    def map(
        self,
        physical: PhysicalMemory,
        *,
        offset: int = 0,
        size: Optional[int] = None,
        physical_offset: int = 0,
        access: int = ACCESS_READWRITE,
    ) -> int:
        """Map ``physical`` into this reservation at ``offset``; returns the address."""
        span = physical.size if size is None else size
        self._validate_span(offset=offset, size=span, physical_offset=physical_offset)
        if physical_offset + span > physical.size:
            raise ValueError(
                f"physical_offset {physical_offset} + size {span} exceeds the "
                f"physical allocation size {physical.size}"
            )

        address = self._ptr + offset
        check(
            "zeVirtualMemMap",
            _driver.lib.zeVirtualMemMap(
                _driver.context,
                _VP(address),
                _SZ(span),
                physical.handle,
                _SZ(physical_offset),
                access,
            ),
        )
        self._mappings.append(_Mapping(offset=offset, size=span, physical=physical))
        physical._acquire_mapping()
        return address

    def _validate_span(self, *, offset: int, size: int, physical_offset: int) -> None:
        if self._closed:
            raise ZeVmmError("virtual reservation is closed")
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        for name, value in (
            ("offset", offset),
            ("size", size),
            ("physical_offset", physical_offset),
        ):
            if value % self._page_size:
                raise ValueError(
                    f"{name} {value} is not a multiple of the device page size "
                    f"{self._page_size}"
                )
        if offset < 0 or offset + size > self._size:
            raise ValueError(
                f"offset {offset} + size {size} exceeds the reservation size "
                f"{self._size}"
            )

    def set_access(
        self, access: int, *, offset: int = 0, size: Optional[int] = None
    ) -> None:
        span = self._size - offset if size is None else size
        self._validate_span(offset=offset, size=span, physical_offset=0)
        check(
            "zeVirtualMemSetAccessAttribute",
            _driver.lib.zeVirtualMemSetAccessAttribute(
                _driver.context, _VP(self._ptr + offset), _SZ(span), access
            ),
        )

    def access_at(
        self, *, offset: int = 0, size: Optional[int] = None
    ) -> Tuple[int, int]:
        """``(access, span)`` -- the attribute at ``offset`` and how far it holds."""
        span = self._size - offset if size is None else size
        self._validate_span(offset=offset, size=span, physical_offset=0)
        access = ctypes.c_int(-1)
        out_size = _SZ(0)
        check(
            "zeVirtualMemGetAccessAttribute",
            _driver.lib.zeVirtualMemGetAccessAttribute(
                _driver.context,
                _VP(self._ptr + offset),
                _SZ(span),
                ctypes.byref(access),
                ctypes.byref(out_size),
            ),
        )
        return int(access.value), int(out_size.value)

    def close(self) -> None:
        """Unmap every mapping, then free the reservation.

        Every step is attempted even when an earlier one fails, so one bad range
        cannot leak the rest; the first error is raised at the end.
        """
        if self._closed:
            return
        self._closed = True
        lib = _driver.lib
        first_error: Optional[BaseException] = None

        for mapping in reversed(self._mappings):
            try:
                check(
                    "zeVirtualMemUnmap",
                    lib.zeVirtualMemUnmap(
                        _driver.context,
                        _VP(self._ptr + mapping.offset),
                        _SZ(mapping.size),
                    ),
                )
            except ZeVmmError as exc:
                logger.warning("zeVirtualMemUnmap failed during close: %s", exc)
                first_error = first_error or exc
                # Left counted on purpose: the range may still be live, and
                # refusing to destroy the allocation beats faulting on it.
                continue
            mapping.physical._release_mapping()
        self._mappings = []

        try:
            check(
                "zeVirtualMemFree",
                lib.zeVirtualMemFree(_driver.context, _VP(self._ptr), _SZ(self._size)),
            )
        except ZeVmmError as exc:
            logger.warning("zeVirtualMemFree failed during close: %s", exc)
            first_error = first_error or exc

        if first_error is not None:
            raise first_error

    def __enter__(self) -> VirtualReservation:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ----------------------------------------------------------------- capability


def is_vmm_supported(*, device_index: int = 0) -> bool:
    """Whether this driver can create, map and dma-buf export physical memory.

    Probes rather than inspecting versions, because the loader exports every VMM
    symbol whether or not the driver behind it implements them.
    """
    return vmm_unsupported_reason(device_index=device_index) is None


def vmm_unsupported_reason(*, device_index: int = 0) -> Optional[str]:
    """``None`` when VMM works, otherwise why it does not."""
    try:
        page_size = query_page_size(1 << 20, device_index=device_index)
        physical = PhysicalMemory.create(
            page_size, device_index=device_index, exportable=True
        )
    except (OSError, ZeVmmError, ValueError) as exc:
        return str(exc)

    try:
        fd = physical.export_dma_buf()
        os.close(fd)
        with VirtualReservation(page_size, device_index=device_index) as reservation:
            reservation.map(physical)
        return None
    except (OSError, ZeVmmError, ValueError) as exc:
        return str(exc)
    finally:
        try:
            physical.close()
        except ZeVmmError as exc:
            logger.debug("probe cleanup failed: %s", exc)


# -------------------------------------------------------- verification copies


def copy_to_device(ptr: int, data: bytes, *, device_index: int = 0) -> None:
    """Blocking host-to-device copy, for checking a mapping end to end."""
    _copy(ptr, bytearray(data), to_device=True, device_index=device_index)


def copy_from_device(ptr: int, size: int, *, device_index: int = 0) -> bytes:
    """Blocking device-to-host copy, for checking a mapping end to end."""
    buf = bytearray(size)
    _copy(ptr, buf, to_device=False, device_index=device_index)
    return bytes(buf)


def _copy(ptr: int, host: bytearray, *, to_device: bool, device_index: int) -> None:
    if ptr == 0:
        raise ValueError("ptr must be non-zero")
    if not host:
        raise ValueError("copy length must be positive")
    lib = _driver.lib
    desc = _CommandQueueDesc(
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        None,
        0,
        0,
        0,
        ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS,
        0,
    )
    cmdlist = _VP()
    check(
        "zeCommandListCreateImmediate",
        lib.zeCommandListCreateImmediate(
            _driver.context,
            _driver.device(device_index),
            ctypes.byref(desc),
            ctypes.byref(cmdlist),
        ),
    )
    try:
        staging = (ctypes.c_ubyte * len(host)).from_buffer_copy(bytes(host))
        source, target = (
            (ctypes.cast(staging, _VP), _VP(ptr))
            if to_device
            else (_VP(ptr), ctypes.cast(staging, _VP))
        )
        check(
            "zeCommandListAppendMemoryCopy",
            lib.zeCommandListAppendMemoryCopy(
                cmdlist, target, source, _SZ(len(host)), None, 0, None
            ),
        )
        check(
            "zeCommandListHostSynchronize",
            lib.zeCommandListHostSynchronize(cmdlist, ctypes.c_uint64(_UINT64_MAX)),
        )
        if not to_device:
            host[:] = bytes(bytearray(staging))
    finally:
        lib.zeCommandListDestroy(cmdlist)
