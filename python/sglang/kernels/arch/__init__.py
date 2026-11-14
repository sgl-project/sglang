from __future__ import annotations

from typing import Optional

from sglang.kernels.arch.base import DeviceArch

_DEVICE_ARCH: Optional[DeviceArch] = None


def current_arch() -> DeviceArch:
    """Arch of the local GPU: `CudaArch` / `HipArch`, else a plain `DeviceArch`.

    Use `.cuda` / `.hip` to narrow; both are None off that platform. Call it at
    call time, not import time: it detects the platform, which initializes CUDA.
    """
    global _DEVICE_ARCH
    if _DEVICE_ARCH is None:
        from sglang.kernels.spec import PlatformInfo

        platform = PlatformInfo.detect()
        if platform.is_cuda:
            from sglang.kernels.arch._cuda import CudaArch

            _DEVICE_ARCH = CudaArch()
        elif platform.is_hip:
            from sglang.kernels.arch._hip import HipArch

            _DEVICE_ARCH = HipArch()
        else:
            _DEVICE_ARCH = DeviceArch()
    return _DEVICE_ARCH


__all__ = ["current_arch"]
