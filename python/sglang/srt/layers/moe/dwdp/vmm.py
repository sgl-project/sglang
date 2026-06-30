"""CUDA Virtual Memory Management (VMM) utilities for DWDP.

Provides low-level CUDA VMM operations including page alignment, handle
creation/export/import, virtual address reservation/mapping, and tensor
creation from VA pointers.  Ported from TensorRT-LLM's DWDP VMM layer.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import platform
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda  # type: ignore[no-redef]


def check_cu_result(cu_func_ret):
    """Check CUDA driver API result and raise on error."""
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {cu_result}")
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {cu_func_ret}")
        return None


def align_up(value: int, alignment: int) -> int:
    """Align *value* up to the nearest multiple of *alignment* (power of 2)."""
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    """Align *value* down to the nearest multiple of *alignment* (power of 2)."""
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    return (value // alignment) * alignment


def peer_handle_type() -> cuda.CUmemAllocationHandleType:
    """Return the peer-shareable handle type for the current architecture.

    aarch64 (GB200/GB300) -> FABRIC (IMEX channel)
    x86_64  (B200/H100)   -> POSIX_FILE_DESCRIPTOR (pidfd exchange)
    """
    arch = platform.machine().lower()
    if "aarch64" in arch:
        return cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    return cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR


def get_allocation_prop(
    device_id: int, fabric_only: bool = True
) -> cuda.CUmemAllocationProp:
    """Build a ``CUmemAllocationProp`` for *device_id*.

    *fabric_only=True* forces FABRIC handle type (safe for granularity
    queries). *fabric_only=False* picks the arch-appropriate type via
    :func:`peer_handle_type`.
    """
    location = cuda.CUmemLocation()
    location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device_id

    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = location

    if fabric_only:
        prop.requestedHandleTypes = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        )
    else:
        prop.requestedHandleTypes = peer_handle_type()

    return prop


@functools.lru_cache(maxsize=None)
def _get_allocation_granularity_cached(device_id: int) -> int:
    prop = get_allocation_prop(device_id)
    option = cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    return check_cu_result(cuda.cuMemGetAllocationGranularity(prop=prop, option=option))


def get_allocation_granularity(device_id: int, use_cache: bool = True) -> int:
    """Query VMM page granularity (typically 2 MB on GB200)."""
    if use_cache:
        return _get_allocation_granularity_cached(device_id)
    prop = get_allocation_prop(device_id)
    option = cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    return check_cu_result(cuda.cuMemGetAllocationGranularity(prop=prop, option=option))


def get_access_desc(device_id: int) -> cuda.CUmemAccessDesc:
    """Build a read/write access descriptor for *device_id*."""
    location = cuda.CUmemLocation()
    location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device_id

    desc = cuda.CUmemAccessDesc()
    desc.location = location
    desc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return desc


def create_fabric_handle(size: int, device_id: int) -> int:
    """Create a peer-shareable memory handle (FABRIC or POSIX_FD)."""
    prop = get_allocation_prop(device_id, fabric_only=False)
    handle = check_cu_result(cuda.cuMemCreate(size, prop, flags=0))
    return int(handle)


def create_local_handle(size: int, device_id: int) -> int:
    """Create a local (non-shareable) handle — no fabric routing table entry."""
    prop = get_allocation_prop(device_id, fabric_only=False)
    prop.requestedHandleTypes = cuda.CUmemAllocationHandleType(0)
    handle = check_cu_result(cuda.cuMemCreate(size, prop, flags=0))
    return int(handle)


def release_handle(handle: int) -> None:
    """Release a memory handle."""
    if handle != 0:
        check_cu_result(cuda.cuMemRelease(handle))


def _to_int(va) -> int:
    """Convert a CUdeviceptr or int to plain int for pointer arithmetic."""
    return int(va)


def _to_devptr(va):
    """Convert an int back to CUdeviceptr for CUDA driver API calls."""
    if isinstance(va, int):
        return cuda.CUdeviceptr(va)
    return va


def reserve_va(size: int, granularity: int) -> int:
    """Reserve contiguous virtual address space. Returns VA as int."""
    va = check_cu_result(cuda.cuMemAddressReserve(size, granularity, 0, 0))
    return _to_int(va)


def free_va(va: int, size: int) -> None:
    """Free reserved virtual address space."""
    if va != 0:
        check_cu_result(cuda.cuMemAddressFree(_to_devptr(va), size))


def map_handle(va: int, size: int, handle: int, offset: int = 0) -> None:
    """Map a memory handle into virtual address space."""
    check_cu_result(cuda.cuMemMap(_to_devptr(va), size, offset, handle, 0))


def unmap_va(va: int, size: int) -> None:
    """Unmap a virtual address region."""
    check_cu_result(cuda.cuMemUnmap(_to_devptr(va), size))


def set_access(va: int, size: int, device_id: int) -> None:
    """Set read/write access on a virtual address region."""
    desc = get_access_desc(device_id)
    check_cu_result(cuda.cuMemSetAccess(_to_devptr(va), size, [desc], 1))


def export_handle(handle: int):
    """Export a shareable handle.

    Returns bytes (FABRIC) or int fd (POSIX_FD).
    """
    ht = peer_handle_type()
    raw = check_cu_result(cuda.cuMemExportToShareableHandle(handle, ht, 0))
    if ht == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC:
        return bytes(raw.data) if hasattr(raw, "data") else bytes(raw)
    return int(raw)


def import_handle(payload) -> int:
    """Import a peer's exported handle. Returns local handle int."""
    ht = peer_handle_type()
    # cuda.bindings uses (shareableHandle, handleType) — no flags arg
    handle = check_cu_result(cuda.cuMemImportFromShareableHandle(payload, ht))
    return int(handle)


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_size_t),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))),
]


@ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
def _no_op_deleter(_ptr):
    pass


_FLOAT8_DTYPES = set()
for _name in ("float8_e5m2", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _FLOAT8_DTYPES.add(_dt)


def _torch_dtype_to_dl(dtype: torch.dtype) -> Tuple[int, int]:
    """Return (dlpack_type_code, bits_per_element) for a torch dtype.

    Float8 types are mapped as uint8 (code=1, 8 bits) because PyTorch's
    from_dlpack does not support kFloat with 8 bits. The caller must
    view-cast back to the original dtype after from_dlpack.
    """
    if dtype in _FLOAT8_DTYPES:
        return 1, 8  # treat as uint8 for DLPack, view-cast later
    if dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
    ):
        return 2, torch.finfo(dtype).bits
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return 0, torch.iinfo(dtype).bits
    if dtype in (torch.uint8,):
        return 1, 8
    raise NotImplementedError(f"Unsupported dtype for DLPack: {dtype}")


def tensor_from_ptr(
    ptr: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device_id: int,
) -> torch.Tensor:
    """Create a **zero-copy** PyTorch tensor over a CUDA virtual address.

    Builds a DLPack capsule via ctypes (no C extension needed) and converts
    it to a PyTorch tensor with ``torch.utils.dlpack.from_dlpack``.
    Ported from TensorRT-LLM ``_dlpack_utils.py``.
    """
    if ptr == 0:
        raise ValueError("Cannot create tensor from null pointer")

    numel = 1
    for d in shape:
        if d <= 0:
            raise ValueError(f"All dimensions must be positive, got shape={shape}")
        numel *= d

    dl_code, bits = _torch_dtype_to_dl(dtype)

    # Build shape array
    ndim = len(shape)
    ShapeArray = ctypes.c_int64 * ndim
    shape_arr = ShapeArray(*shape)

    # DLPack device: kDLCUDA = 2
    device = _DLDevice(device_type=2, device_id=device_id)
    dl_dtype = _DLDataType(code=dl_code, bits=bits, lanes=1)

    dl_tensor = _DLTensor()
    dl_tensor.data = ctypes.c_void_p(ptr)
    dl_tensor.device = device
    dl_tensor.ndim = ndim
    dl_tensor.dtype = dl_dtype
    dl_tensor.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
    dl_tensor.strides = None  # contiguous
    dl_tensor.byte_offset = 0

    managed = _DLManagedTensor()
    managed.dl_tensor = dl_tensor
    managed.manager_ctx = None
    managed.deleter = _no_op_deleter

    # Create PyCapsule named "dltensor"
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    capsule_ptr = ctypes.pythonapi.PyCapsule_New(
        ctypes.pointer(managed),
        b"dltensor",
        None,
    )
    capsule = ctypes.cast(capsule_ptr, ctypes.py_object).value

    tensor = torch.utils.dlpack.from_dlpack(capsule)
    # Prevent GC of ctypes objects while tensor is alive
    tensor._dlpack_prevent_gc = (shape_arr, managed, capsule)
    tensor = tensor.reshape(shape)
    # Float8 was transmitted as uint8 — view-cast back to original dtype
    if dtype in _FLOAT8_DTYPES:
        tensor = tensor.view(dtype)
    return tensor


class VMMHandle:
    """RAII wrapper for a CUDA VMM physical memory handle."""

    __slots__ = ("_handle", "_size", "_device_id", "_released")

    def __init__(self, size: int, device_id: int):
        granularity = get_allocation_granularity(device_id)
        aligned_size = align_up(size, granularity)
        self._handle = create_fabric_handle(aligned_size, device_id)
        self._size = aligned_size
        self._device_id = device_id
        self._released = False

    @property
    def handle(self) -> int:
        if self._released:
            raise RuntimeError("Handle has been released")
        return self._handle

    @property
    def size(self) -> int:
        return self._size

    @property
    def device_id(self) -> int:
        return self._device_id

    def release(self) -> None:
        if not self._released:
            release_handle(self._handle)
            self._released = True

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
        return False


class VARegion:
    """RAII wrapper for a reserved CUDA virtual address region."""

    __slots__ = ("_va", "_size", "_device_id", "_granularity", "_mappings", "_released")

    def __init__(self, size: int, device_id: int):
        self._device_id = device_id
        self._granularity = get_allocation_granularity(device_id)
        aligned_size = align_up(size, self._granularity)
        self._va = reserve_va(aligned_size, self._granularity)
        self._size = aligned_size
        self._mappings: List[Tuple[int, int]] = []
        self._released = False

    @property
    def va(self) -> int:
        if self._released:
            raise RuntimeError("VA region has been released")
        return self._va

    @property
    def size(self) -> int:
        return self._size

    def map(self, offset: int, size: int, handle: int, handle_offset: int = 0) -> int:
        """Map *handle* at *offset* within this region. Returns the mapped VA."""
        if offset + size > self._size:
            raise ValueError(
                f"Mapping at offset={offset} size={size} exceeds region size={self._size}"
            )
        va = self._va + offset
        map_handle(va, size, handle, handle_offset)
        self._mappings.append((offset, size))
        return va

    def unmap_all(self) -> None:
        for offset, size in self._mappings:
            try:
                unmap_va(self._va + offset, size)
            except Exception:
                pass
        self._mappings.clear()

    def release(self) -> None:
        if not self._released:
            self.unmap_all()
            free_va(self._va, self._size)
            self._released = True

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
        return False
