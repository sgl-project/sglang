"""PPU device operations for the SRT platform layer.

T-HEAD PPU exposes a CUDA-compatible API through its torch backend, so
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
    """Return True when the local device is a T-HEAD PPU (device name holds "ZW")."""
    return bool(torch.cuda.is_available() and _ZW_TAG in torch.cuda.get_device_name())


class PPUDeviceMixin(CudaDeviceMixin):
    """PPU device ops — identical surface to CUDA via T-HEAD's torch backend."""

    _enum: PlatformEnum = PlatformEnum.PPU
    device_name: str = "ppu"
    # device_type stays "cuda" — torch.device("cuda") is the only valid
    # device-type string for PPU devices in PyTorch.

    def is_cuda(self) -> bool:
        # PPU exposes a CUDA-compatible API, so it is also considered "cuda"
        return True


class PPUSRTPlatform(PPUDeviceMixin, SRTPlatform):
    """Default in-tree T-HEAD PPU SRT platform.

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

    def get_device_num_tensorcores(self, device_id: int = 0) -> int:
        if self.is_zw810e():
            return 20
        return super().get_device_num_tensorcores(device_id)


# Load PPU hooks when running on PPU hardware.
# Each module registers @plugin_hook decorators that inject PPU-native ops
# into the respective backends transparently.
if is_ppu_available():
    from sglang.srt.hardware_backend.ppu.kernel_version_check import (
        check_ppu_kernel_version,
    )

    # Surface a wrong/mismatched sgl-kernel (e.g. a stray upstream CUDA wheel)
    # explicitly instead of running silently; the upstream engine.py assertion
    # is _is_cuda-gated and never fires on PPU.
    check_ppu_kernel_version()

    import sglang.srt.hardware_backend.ppu.attention.ppu_dsa_hooks  # noqa: F401
    import sglang.srt.hardware_backend.ppu.attention.ppu_fa3_hooks  # noqa: F401
    import sglang.srt.hardware_backend.ppu.attention.ppu_flashmla_hooks  # noqa: F401
    import sglang.srt.hardware_backend.ppu.attention.ppu_msa_hook  # noqa: F401
    import sglang.srt.hardware_backend.ppu.moe.ppu_deepep_hooks  # noqa: F401
