"""PPU device operations for the SRT platform layer.

PPU exposes a CUDA-compatible API through its torch backend, so
``PPUDeviceMixin`` inherits all device ops from ``CudaDeviceMixin`` and
only overrides identity (``_enum``, ``device_name``).

PPU presence is detected by matching the ``ZW`` tag in the CUDA device name,
keeping the check consistent with ``sglang.srt.utils.is_ppu``.
"""

from functools import lru_cache

import torch

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

_ZW_TAG = "ZW"
_ZW810E_NAME = "ZW810E"
_810_TAG = "810"


@lru_cache(maxsize=1)
def is_ppu_available() -> bool:
    """Return True when the local device is a PPU (device name holds "ZW")."""
    return bool(torch.cuda.is_available() and _ZW_TAG in torch.cuda.get_device_name())


class PPUDeviceMixin(CudaDeviceMixin):
    """PPU device ops — identical surface to CUDA via the PPU torch backend."""

    _enum: PlatformEnum = PlatformEnum.PPU
    device_name: str = "ppu"
    # device_type stays "cuda" — torch.device("cuda") is the only valid
    # device-type string for PPU devices in PyTorch.

    def is_cuda(self) -> bool:
        # PPU exposes a CUDA-compatible API, so it is also considered "cuda"
        return True


class PPUSRTPlatform(PPUDeviceMixin, SRTPlatform):
    """Default in-tree PPU SRT platform.

    Inherits CUDA capability flags since PPU exposes CUDA-compatible APIs
    (fp8, cuda graph, piecewise cuda graph are supported on PPU).
    """

    def supports_fp8(self) -> bool:
        return self.get_device_capability().to_int() >= 89

    def support_cuda_graph(self) -> bool:
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        return True

    def get_jit_cuda_arch_suffix(self) -> str:
        # PPU 810/810e devices require the "sm80a" arch variant for JIT compilation.
        return "a" if _810_TAG in self.get_device_name() else ""

    def is_zw810e(self) -> bool:
        return _ZW810E_NAME in self.get_device_name()
