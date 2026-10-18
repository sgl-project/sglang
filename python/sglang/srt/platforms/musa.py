"""MUSA identity and capabilities for the SRT platform layer.

MUSA is CUDA-alike (the torchada shim maps ``torch.cuda`` onto MUSA), so
device operations inherit from ``CudaDeviceMixin``; identity, the torch
distributed backend and the capability declaration differ.
"""

import torch

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform


class MusaDeviceMixin(CudaDeviceMixin):
    _enum: PlatformEnum = PlatformEnum.MUSA
    device_name: str = "musa"
    device_type: str = "musa"

    def get_torch_distributed_backend_str(self) -> str:
        return "mccl"

    def get_device(self, device_id: int = 0) -> torch.device:
        return torch.device("musa", device_id)

    def set_device(self, device: torch.device) -> None:
        torch.musa.set_device(device)

    def current_device(self) -> int:
        return torch.musa.current_device()

    def current_stream(self, device=None):
        return torch.musa.current_stream(device)

    def synchronize(self) -> None:
        torch.musa.synchronize()

    def empty_cache(self) -> None:
        torch.musa.empty_cache()


class MusaSRTPlatform(MusaDeviceMixin, SRTPlatform):
    """Default in-tree MUSA SRT platform."""

    capabilities = PlatformCapabilities(supports_triton=True, graph_capture=True)
