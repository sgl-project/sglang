"""MUSA identity and capabilities for the SRT platform layer.

MUSA is CUDA-alike (the torchada shim maps ``torch.cuda`` onto MUSA), so
device operations inherit from ``CudaDeviceMixin``; identity, the torch
distributed backend and the capability declaration differ.
"""

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform


class MusaDeviceMixin(CudaDeviceMixin):
    _enum: PlatformEnum = PlatformEnum.MUSA
    device_name: str = "musa"
    device_type: str = "musa"

    def get_torch_distributed_backend_str(self) -> str:
        return "mccl"


class MusaSRTPlatform(MusaDeviceMixin, SRTPlatform):
    """Default in-tree MUSA SRT platform."""

    capabilities = PlatformCapabilities(supports_triton=True, graph_capture=True)
