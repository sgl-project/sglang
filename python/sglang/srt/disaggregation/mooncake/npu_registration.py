"""Opt-in CANN allocation-boundary registration for the Ascend transport.
Transfer descriptors keep their original per-layer addresses and lengths.
"""

import ctypes
from functools import lru_cache


class Location(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_int)]


class Attributes(ctypes.Structure):
    _fields_ = [
        ("location", Location),
        ("page_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


@lru_cache(maxsize=1)
def api() -> ctypes.CDLL:
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtPointerGetAttributes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Attributes),
    ]
    lib.aclrtPointerGetAttributes.restype = ctypes.c_int
    lib.aclrtMemGetAddressRange.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.aclrtMemGetAddressRange.restype = ctypes.c_int
    return lib


def registration_region(ptr: int, length: int) -> tuple[int, int]:
    if ptr <= 0 or length <= 0:
        raise ValueError(f"Invalid registration region: ptr={ptr}, length={length}")
    lib = api()
    attr = Attributes()
    ret = lib.aclrtPointerGetAttributes(ctypes.c_void_p(ptr), ctypes.byref(attr))
    if ret != 0:
        raise RuntimeError(f"aclrtPointerGetAttributes failed: {ret}")
    if attr.location.type != 1:  # ACL_MEM_LOCATION_TYPE_DEVICE
        return ptr, length
    base, size = ctypes.c_void_p(), ctypes.c_size_t()
    ret = lib.aclrtMemGetAddressRange(
        ctypes.c_void_p(ptr), ctypes.byref(base), ctypes.byref(size)
    )
    if ret != 0:
        raise RuntimeError(f"aclrtMemGetAddressRange failed: {ret}")
    if (
        base.value is None
        or not base.value <= ptr < ptr + length <= base.value + size.value
    ):
        raise RuntimeError(
            f"Registration exceeds allocation: ptr={ptr} length={length} base={base.value} size={size.value}"
        )
    return base.value, size.value


def registration_regions(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # aclrtMemGetAddressRange may report a larger reserved extent than the
    # mapped tensor bytes. Register only through the last referenced byte.
    spans = {}
    for ptr, length in regions:
        base, capacity = registration_region(ptr, length)
        end = ptr + length
        if not base <= ptr < end <= base + capacity:
            raise RuntimeError("Invalid allocation span")
        spans[base] = max(spans.get(base, base), end)
    return [(base, end - base) for base, end in spans.items()]
