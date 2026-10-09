from __future__ import annotations

from typing import TYPE_CHECKING

import tvm_ffi

from sglang.kernels.jit.utils.common import cache_once, lazy_register_class
from sglang.kernels.jit.utils.compile import load_jit

__all__ = ["VMMHostMemory"]


@cache_once
def _jit_vmm_module() -> tvm_ffi.Module:
    return load_jit(
        "vmm_host_memory",
        extra_ldflags=["-lcuda"],
        cuda_files=["arch/vmm.cuh"],
        cuda_wrappers=[
            ("register_once", "register_vmm_host_memory"),
            ("is_available", "is_vmm_available"),
        ],
    )


def _init_vmm_host_memory() -> None:
    _jit_vmm_module().register_once()


def is_vmm_available() -> bool:
    return _jit_vmm_module().is_available()


@lazy_register_class("sgl.VMMHostMemory", _init_vmm_host_memory)
class VMMHostMemory(tvm_ffi.Object):
    """Pinned host memory at VMM granularity (2MB), shareable by POSIX fd.

    ``VMMHostMemory(size)`` creates a new allocation on the device's host NUMA
    node; ``VMMHostMemory(size, fd)`` imports a peer's exported fd and must pass
    the creator's ``size()`` exactly. Send fds over a UDS (``socket.send_fds``),
    never by number; the fd stays owned by the caller on both sides. Raises on
    hosts without host-NUMA VMM support (``CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID``
    is -1), which is the fallback signal for callers.
    """

    if TYPE_CHECKING:
        # C++ interface
        def export_fd(self) -> int: ...
        def ptr(self) -> int: ...
        def size(self) -> int: ...
        def close(self) -> None: ...

    def __init__(self, size: int, fd: int | None = None) -> None:
        self.__ffi_init__(size, fd)  # type: ignore
