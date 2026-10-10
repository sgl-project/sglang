"""SAIL device operations for the SRT platform layer.

T-HEAD SAIL exposes a CUDA-compatible API through its torch backend, so
``SailDeviceMixin`` inherits all device ops from ``CudaDeviceMixin`` and
only overrides identity (``_enum``, ``device_name``).
"""

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

_ZW810E_NAME = "ZW810E"


class SailDeviceMixin(CudaDeviceMixin):
    """SAIL device ops — identical surface to CUDA via T-HEAD's torch backend."""

    _enum: PlatformEnum = PlatformEnum.SAIL
    device_name: str = "sail"
    # device_type stays "cuda" — torch.device("cuda") is the only valid
    # device-type string for SAIL devices in PyTorch.

    def is_cuda(self) -> bool:
        # SAIL exposes a CUDA-compatible API, so it is also considered "cuda"
        return True


class SailSRTPlatform(SailDeviceMixin, SRTPlatform):
    """Default in-tree T-HEAD SAIL SRT platform.

    Inherits CUDA capability flags since SAIL exposes CUDA-compatible APIs
    (fp8, cuda graph, piecewise cuda graph are supported on SAIL).
    """

    def supports_fp8(self) -> bool:
        return self.get_device_capability().to_int() >= 89

    def support_cuda_graph(self) -> bool:
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        return True

    def is_zw810e(self) -> bool:
        return _ZW810E_NAME in self.get_device_name()
