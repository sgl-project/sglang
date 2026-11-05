"""Exact-size ACL host allocations owned by the tensor buffer exporter."""

import ctypes
import logging
import math
from functools import lru_cache

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _acl_host_api():
    library = ctypes.CDLL("libascendcl.so")
    library.aclrtMallocHost.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    library.aclrtMallocHost.restype = ctypes.c_int
    library.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    library.aclrtFreeHost.restype = ctypes.c_int
    return library


class _PinnedAllocation:
    def __init__(self, nbytes: int):
        self.library = _acl_host_api()
        self.pointer = ctypes.c_void_p()
        result = self.library.aclrtMallocHost(ctypes.byref(self.pointer), nbytes)
        if result != 0 or not self.pointer.value:
            self.pointer = ctypes.c_void_p()
            raise RuntimeError(
                f"aclrtMallocHost failed with code {result} for {nbytes} bytes"
            )

    def __del__(self):
        pointer = getattr(self, "pointer", None)
        if pointer is not None and pointer.value:
            self.pointer = ctypes.c_void_p()
            try:
                result = self.library.aclrtFreeHost(pointer)
                if result:
                    logger.warning("aclrtFreeHost returned %s", result)
            except Exception:
                # CANN may already be unloading during interpreter shutdown.
                logger.debug("ACL host allocation cleanup failed", exc_info=True)


def allocate_acl_host(dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
    if str(device) != "cpu":
        raise ValueError("ACL HiCache host allocation requires device=cpu")
    if not hasattr(torch, "npu") or not torch.npu.get_device_name().startswith(
        "Ascend910B"
    ):
        raise RuntimeError("ACL HiCache host allocation requires an Ascend A2 device")
    if any(dimension < 0 for dimension in dims):
        raise ValueError("Host tensor dimensions must be nonnegative")
    count = math.prod(dims)
    if count == 0:
        return torch.empty(dims, dtype=dtype, device="cpu")
    nbytes = count * dtype.itemsize
    owner = _PinnedAllocation(nbytes)
    exporter = (ctypes.c_ubyte * nbytes).from_address(owner.pointer.value)
    # torch.frombuffer retains the exporter. Views retain its storage and owner.
    exporter._allocation_owner = owner
    return torch.frombuffer(exporter, dtype=dtype, count=count).reshape(dims)
