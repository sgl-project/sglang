"""ROCm HIP virtual-memory primitives used by the SharedEP VMM facade.

The HIP runtime has no supported Python bindings in SGLang's ROCm dependency
set.  These wrappers bind the small, C-compatible VMM surface directly instead
of adding an unpinned package or requiring a compiler on serving nodes.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from functools import lru_cache

_HIP_SUCCESS = 0
_HIP_MEM_ALLOCATION_TYPE_UNCACHED = 0x40000000
_HIP_MEM_HANDLE_TYPE_POSIX_FD = 0x1
_HIP_MEM_LOCATION_TYPE_DEVICE = 0x1
_HIP_MEM_ACCESS_PROT_READ_WRITE = 0x3
_HIP_MEM_GRANULARITY_MINIMUM = 0x0
_MIN_UNCACHED_VMM_RUNTIME_VERSION = 7 * 10_000_000 + 2 * 100_000


class _HipMemLocation(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_int),
    ]


class _HipMemAllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compression_type", ctypes.c_ubyte),
        ("gpu_direct_rdma_capable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
    ]


class _HipMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requested_handle_type", ctypes.c_int),
        ("location", _HipMemLocation),
        ("win32_handle_metadata", ctypes.c_void_p),
        ("alloc_flags", _HipMemAllocationFlags),
    ]


class _HipMemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", _HipMemLocation),
        ("flags", ctypes.c_int),
    ]


class HipVmmError(RuntimeError):
    def __init__(self, label: str, code: int, detail: str):
        self.label = label
        self.code = int(code)
        super().__init__(f"{label}: HIP error {code} ({detail})")


class _HipRuntime:
    """Typed ctypes binding for the HIP VMM ABI."""

    def __init__(self, library: ctypes.CDLL):
        self._library = library
        self._bind_functions()

    def _bind(self, name: str, argtypes, restype=ctypes.c_int):
        try:
            function = getattr(self._library, name)
        except AttributeError as error:
            raise RuntimeError(
                f"HIP runtime does not export required VMM symbol {name}"
            ) from error
        function.argtypes = argtypes
        function.restype = restype
        return function

    def _bind_functions(self) -> None:
        void_p = ctypes.c_void_p
        size_t = ctypes.c_size_t
        uint64 = ctypes.c_ulonglong
        self.hipGetErrorString = self._bind(
            "hipGetErrorString",
            [ctypes.c_int],
            ctypes.c_char_p,
        )
        self.hipRuntimeGetVersion = self._bind(
            "hipRuntimeGetVersion",
            [ctypes.POINTER(ctypes.c_int)],
        )
        self.hipSetDevice = self._bind("hipSetDevice", [ctypes.c_int])
        self.hipMemGetAllocationGranularity = self._bind(
            "hipMemGetAllocationGranularity",
            [
                ctypes.POINTER(size_t),
                ctypes.POINTER(_HipMemAllocationProp),
                ctypes.c_int,
            ],
        )
        self.hipMemCreate = self._bind(
            "hipMemCreate",
            [
                ctypes.POINTER(void_p),
                size_t,
                ctypes.POINTER(_HipMemAllocationProp),
                uint64,
            ],
        )
        self.hipMemExportToShareableHandle = self._bind(
            "hipMemExportToShareableHandle",
            [void_p, void_p, ctypes.c_int, uint64],
        )
        self.hipMemImportFromShareableHandle = self._bind(
            "hipMemImportFromShareableHandle",
            [ctypes.POINTER(void_p), void_p, ctypes.c_int],
        )
        self.hipMemAddressReserve = self._bind(
            "hipMemAddressReserve",
            [ctypes.POINTER(void_p), size_t, size_t, void_p, uint64],
        )
        self.hipMemMap = self._bind(
            "hipMemMap",
            [void_p, size_t, size_t, void_p, uint64],
        )
        self.hipMemSetAccess = self._bind(
            "hipMemSetAccess",
            [
                void_p,
                size_t,
                ctypes.POINTER(_HipMemAccessDesc),
                size_t,
            ],
        )
        self.hipMemUnmap = self._bind("hipMemUnmap", [void_p, size_t])
        self.hipMemRelease = self._bind("hipMemRelease", [void_p])
        self.hipMemAddressFree = self._bind(
            "hipMemAddressFree",
            [void_p, size_t],
        )

    def _error_string(self, code: int) -> str:
        message = self.hipGetErrorString(int(code))
        return (
            message.decode("utf-8", errors="replace")
            if message is not None
            else "unknown error"
        )

    def check(self, code: int, label: str) -> None:
        if int(code) != _HIP_SUCCESS:
            raise HipVmmError(label, int(code), self._error_string(int(code)))

    def runtime_version(self) -> int:
        version = ctypes.c_int()
        self.check(
            self.hipRuntimeGetVersion(ctypes.byref(version)),
            "hipRuntimeGetVersion",
        )
        return int(version.value)

    def set_device(self, device_id: int) -> None:
        self.check(self.hipSetDevice(int(device_id)), "hipSetDevice")

    def allocation_granularity(self, device_id: int) -> int:
        prop = _allocation_prop(device_id)
        granularity = ctypes.c_size_t()
        self.check(
            self.hipMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(prop),
                _HIP_MEM_GRANULARITY_MINIMUM,
            ),
            "hipMemGetAllocationGranularity",
        )
        return int(granularity.value)

    def create(self, size: int, device_id: int) -> int:
        prop = _allocation_prop(device_id)
        handle = ctypes.c_void_p()
        self.check(
            self.hipMemCreate(
                ctypes.byref(handle),
                int(size),
                ctypes.byref(prop),
                0,
            ),
            "hipMemCreate(Uncached, POSIX_FD)",
        )
        if handle.value is None:
            raise RuntimeError("hipMemCreate returned a null allocation handle")
        return int(handle.value)

    def export_fd(self, handle: int) -> int:
        fd = ctypes.c_int(-1)
        self.check(
            self.hipMemExportToShareableHandle(
                ctypes.byref(fd),
                ctypes.c_void_p(handle),
                _HIP_MEM_HANDLE_TYPE_POSIX_FD,
                0,
            ),
            "hipMemExportToShareableHandle(POSIX_FD)",
        )
        if fd.value < 0:
            raise RuntimeError(f"HIP exported an invalid POSIX fd {fd.value}")
        return int(fd.value)

    def import_fd(self, fd: int) -> int:
        handle = ctypes.c_void_p()
        # ROCm 7.1+ takes the descriptor value cast to void*, not int*.
        os_handle = ctypes.c_void_p(int(fd))
        self.check(
            self.hipMemImportFromShareableHandle(
                ctypes.byref(handle),
                os_handle,
                _HIP_MEM_HANDLE_TYPE_POSIX_FD,
            ),
            "hipMemImportFromShareableHandle(POSIX_FD)",
        )
        if handle.value is None:
            raise RuntimeError(
                "hipMemImportFromShareableHandle returned a null allocation handle"
            )
        return int(handle.value)

    def reserve(self, size: int, alignment: int) -> int:
        address = ctypes.c_void_p()
        self.check(
            self.hipMemAddressReserve(
                ctypes.byref(address),
                int(size),
                int(alignment),
                None,
                0,
            ),
            "hipMemAddressReserve",
        )
        if address.value is None:
            raise RuntimeError("hipMemAddressReserve returned a null address")
        return int(address.value)

    def map(self, address: int, size: int, handle: int) -> None:
        self.check(
            self.hipMemMap(
                ctypes.c_void_p(address),
                int(size),
                0,
                ctypes.c_void_p(handle),
                0,
            ),
            "hipMemMap",
        )

    def set_access(self, address: int, size: int, device_id: int) -> None:
        access = _HipMemAccessDesc(
            location=_HipMemLocation(
                type=_HIP_MEM_LOCATION_TYPE_DEVICE,
                id=int(device_id),
            ),
            flags=_HIP_MEM_ACCESS_PROT_READ_WRITE,
        )
        self.check(
            self.hipMemSetAccess(
                ctypes.c_void_p(address),
                int(size),
                ctypes.byref(access),
                1,
            ),
            "hipMemSetAccess(PROT_READWRITE)",
        )

    def unmap(self, address: int, size: int) -> None:
        self.check(
            self.hipMemUnmap(ctypes.c_void_p(address), int(size)),
            "hipMemUnmap",
        )

    def release(self, handle: int) -> None:
        self.check(
            self.hipMemRelease(ctypes.c_void_p(handle)),
            "hipMemRelease",
        )

    def address_free(self, address: int, size: int) -> None:
        self.check(
            self.hipMemAddressFree(ctypes.c_void_p(address), int(size)),
            "hipMemAddressFree",
        )


def _allocation_prop(device_id: int) -> _HipMemAllocationProp:
    return _HipMemAllocationProp(
        type=_HIP_MEM_ALLOCATION_TYPE_UNCACHED,
        requested_handle_type=_HIP_MEM_HANDLE_TYPE_POSIX_FD,
        location=_HipMemLocation(
            type=_HIP_MEM_LOCATION_TYPE_DEVICE,
            id=int(device_id),
        ),
        win32_handle_metadata=None,
        alloc_flags=_HipMemAllocationFlags(),
    )


@lru_cache(maxsize=1)
def _load_hip_runtime() -> _HipRuntime:
    override = os.environ.get("SGLANG_HIP_RUNTIME_LIBRARY")
    candidates = [
        override,
        ctypes.util.find_library("amdhip64"),
        "libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/lib64/libamdhip64.so",
    ]
    errors = []
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return _HipRuntime(ctypes.CDLL(candidate))
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    raise RuntimeError(
        "Unable to load the ROCm HIP runtime (libamdhip64.so)"
        + (f": {'; '.join(errors)}" if errors else "")
    )


def _format_hip_version(version: int) -> str:
    major = version // 10_000_000
    minor = (version // 100_000) % 100
    patch = version % 100_000
    return f"{major}.{minor}.{patch}"


class HipVmmBackend:
    """HIP implementation of SharedEP VMM using uncached POSIX allocations."""

    platform = "rocm"
    dlpack_device_type = 10  # kDLROCM
    supports_fabric = False

    @lru_cache(maxsize=None)
    def ensure_supported(self, device_id: int) -> None:
        if os.name != "posix":
            raise RuntimeError("HIP POSIX-fd VMM is supported only on POSIX systems")
        runtime = _load_hip_runtime()
        version = runtime.runtime_version()
        if version < _MIN_UNCACHED_VMM_RUNTIME_VERSION:
            raise RuntimeError(
                "uncached HIP VMM requires ROCm 7.2 or newer; "
                f"found HIP runtime {_format_hip_version(version)}"
            )
        runtime.set_device(device_id)

        # Probe the complete local primitive chain. Device attributes alone do
        # not guarantee that POSIX export is enabled by the driver/kernel stack.
        local_handle = None
        imported_handle = None
        exported_fd = None
        address = None
        mapped = False
        granularity = None
        probe_error = None
        try:
            granularity = self.get_allocation_granularity(
                device_id,
                allow_fabric=False,
            )
            local_handle = self.create_allocation(
                granularity,
                device_id,
                allow_fabric=False,
            )
            exported_fd = self.export_posix_fd(local_handle)
            imported_handle = self.import_posix_fd(exported_fd)
            address = self.reserve(granularity, granularity)
            self.map(address, granularity, imported_handle)
            mapped = True
            self.set_access(address, granularity, device_id)
        except BaseException as error:
            probe_error = error

        cleanup_errors = []
        cleanup_actions = []
        if mapped:
            cleanup_actions.append(
                lambda: self.unmap(address, granularity),
            )
        if address is not None:
            cleanup_actions.append(
                lambda: self.address_free(address, granularity),
            )
        if imported_handle is not None:
            cleanup_actions.append(lambda: self.release(imported_handle))
        if exported_fd is not None:
            cleanup_actions.append(lambda: os.close(exported_fd))
        if local_handle is not None:
            cleanup_actions.append(lambda: self.release(local_handle))
        for cleanup in cleanup_actions:
            try:
                cleanup()
            except BaseException as error:
                cleanup_errors.append(error)

        if probe_error is not None:
            message = f"HIP VMM capability probe failed: {probe_error}"
            if cleanup_errors:
                message += "; cleanup errors: " + "; ".join(
                    str(error) for error in cleanup_errors
                )
            raise RuntimeError(message) from probe_error
        if cleanup_errors:
            raise RuntimeError(
                "HIP VMM capability cleanup failed: "
                + "; ".join(str(error) for error in cleanup_errors)
            )

    def get_allocation_granularity(self, device_id: int, *, allow_fabric: bool) -> int:
        if allow_fabric:
            raise RuntimeError("HIP VMM does not support CUDA FABRIC handles")
        return _load_hip_runtime().allocation_granularity(device_id)

    def create_allocation(
        self, size: int, device_id: int, *, allow_fabric: bool
    ) -> int:
        if allow_fabric:
            raise RuntimeError("HIP VMM does not support CUDA FABRIC handles")
        return _load_hip_runtime().create(size, device_id)

    def release(self, handle: int) -> None:
        _load_hip_runtime().release(handle)

    def export_fabric(self, handle: int) -> bytes:
        raise RuntimeError("FABRIC VMM handles are CUDA-only")

    def export_posix_fd(self, handle: int) -> int:
        return _load_hip_runtime().export_fd(handle)

    def import_fabric(self, handle: bytes) -> int:
        raise RuntimeError("FABRIC VMM handles are CUDA-only")

    def import_posix_fd(self, fd: int) -> int:
        dup_fd = os.dup(fd)
        try:
            return _load_hip_runtime().import_fd(dup_fd)
        finally:
            os.close(dup_fd)

    def reserve(self, size: int, alignment: int = 0) -> int:
        return _load_hip_runtime().reserve(size, alignment)

    def map(self, address: int, size: int, handle: int) -> None:
        _load_hip_runtime().map(address, size, handle)

    def set_access(self, address: int, size: int, device_id: int) -> None:
        _load_hip_runtime().set_access(address, size, device_id)

    def unmap(self, address: int, size: int) -> None:
        _load_hip_runtime().unmap(address, size)

    def address_free(self, address: int, size: int) -> None:
        _load_hip_runtime().address_free(address, size)
