"""CUDA VMM primitives for DWDP: handle creation, VA reserve/map, DLPack tensor views."""

from __future__ import annotations

import ctypes
import functools
import logging
from typing import Tuple

import torch
try:
    from cuda.bindings import driver as cuda
except ImportError:
    cuda = None

from sglang.srt.distributed.device_communicators.vmm_utils import (
    check_drv,
    make_rw_access_desc,
)

logger = logging.getLogger(__name__)


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    return (value // alignment) * alignment


def _make_prop(device_id: int, handle_types: int) -> cuda.CUmemAllocationProp:
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    prop.requestedHandleTypes = handle_types
    return prop


@functools.lru_cache(maxsize=None)
def shareable_handle_types(device_id: int) -> int:
    fabric = int(cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC)
    posix = int(cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    fabric_supported = check_drv(
        cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
            device_id,
        ),
        "cuDeviceGetAttribute(FABRIC_SUPPORTED)",
    )
    if fabric_supported:
        # the attribute alone is not sufficient: drivers advertise FABRIC on
        # platforms where creation still fails (e.g. no IMEX channel), so a
        # real cuMemCreate probe decides
        combined = fabric | posix
        option = (
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )
        try:
            prop = _make_prop(device_id, combined)
            gran = check_drv(
                cuda.cuMemGetAllocationGranularity(prop=prop, option=option),
                "cuMemGetAllocationGranularity(probe)",
            )
            handle = check_drv(
                cuda.cuMemCreate(int(gran), prop, 0), "cuMemCreate(probe)"
            )
            check_drv(cuda.cuMemRelease(handle), "cuMemRelease(probe)")
            return combined
        except RuntimeError as e:
            logger.info(
                "FABRIC advertised on device %s but creation probe failed (%s); "
                "DWDP handles will be POSIX fd only",
                device_id,
                e,
            )
    return posix


@functools.lru_cache(maxsize=None)
def get_allocation_granularity(device_id: int) -> int:
    prop = _make_prop(device_id, shareable_handle_types(device_id))
    option = cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    return check_drv(
        cuda.cuMemGetAllocationGranularity(prop=prop, option=option),
        "cuMemGetAllocationGranularity",
    )


def create_fabric_handle(size: int, device_id: int) -> int:
    prop = _make_prop(device_id, shareable_handle_types(device_id))
    handle = check_drv(cuda.cuMemCreate(size, prop, flags=0), "cuMemCreate")
    return int(handle)


def create_local_handle(size: int, device_id: int) -> int:
    # non-shareable handle: does not consume a fabric routing table entry
    prop = _make_prop(device_id, 0)
    handle = check_drv(cuda.cuMemCreate(size, prop, flags=0), "cuMemCreate(local)")
    return int(handle)


def release_handle(handle: int) -> None:
    if handle != 0:
        check_drv(cuda.cuMemRelease(handle), "cuMemRelease")


def reserve_va(size: int, granularity: int) -> int:
    va = check_drv(
        cuda.cuMemAddressReserve(size, granularity, 0, 0), "cuMemAddressReserve"
    )
    return int(va)


def free_va(va: int, size: int) -> None:
    if va != 0:
        check_drv(cuda.cuMemAddressFree(va, size), "cuMemAddressFree")


def map_handle(va: int, size: int, handle: int, offset: int = 0) -> None:
    check_drv(cuda.cuMemMap(va, size, offset, handle, 0), "cuMemMap")


def unmap_va(va: int, size: int) -> None:
    check_drv(cuda.cuMemUnmap(va, size), "cuMemUnmap")


def set_access(va: int, size: int, device_id: int) -> None:
    desc = make_rw_access_desc(device_id)
    check_drv(cuda.cuMemSetAccess(va, size, [desc], 1), "cuMemSetAccess")


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


_FLOAT8_DTYPES = {
    torch.float8_e5m2,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
}


def _torch_dtype_to_dl(dtype: torch.dtype) -> Tuple[int, int]:
    # float8 goes through DLPack as uint8 (from_dlpack rejects kFloat/8-bit); caller view-casts back
    if dtype in _FLOAT8_DTYPES:
        return 1, 8
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
    if ptr == 0:
        raise ValueError("Cannot create tensor from null pointer")

    numel = 1
    for d in shape:
        if d <= 0:
            raise ValueError(f"All dimensions must be positive, got shape={shape}")
        numel *= d

    dl_code, bits = _torch_dtype_to_dl(dtype)

    ndim = len(shape)
    ShapeArray = ctypes.c_int64 * ndim
    shape_arr = ShapeArray(*shape)

    device = _DLDevice(device_type=2, device_id=device_id)  # kDLCUDA = 2
    dl_dtype = _DLDataType(code=dl_code, bits=bits, lanes=1)

    dl_tensor = _DLTensor()
    dl_tensor.data = ctypes.c_void_p(ptr)
    dl_tensor.device = device
    dl_tensor.ndim = ndim
    dl_tensor.dtype = dl_dtype
    dl_tensor.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
    dl_tensor.strides = None
    dl_tensor.byte_offset = 0

    managed = _DLManagedTensor()
    managed.dl_tensor = dl_tensor
    managed.manager_ctx = None
    managed.deleter = _no_op_deleter

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
    tensor = tensor.reshape(shape)
    if dtype in _FLOAT8_DTYPES:
        tensor = tensor.view(dtype)
    # ctypes structs must outlive the tensor or the data pointer dangles
    tensor._dlpack_prevent_gc = (shape_arr, managed, capsule)
    return tensor
