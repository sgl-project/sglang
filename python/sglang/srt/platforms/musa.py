"""MUSA device operations for the SRT platform layer.

MUSA uses torchada to expose a CUDA-compatible API surface through
``torch.cuda.*``. Reuse the CUDA device operations and override only platform
identity.
"""

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform


class MusaDeviceMixin(CudaDeviceMixin):
    """MUSA device ops through torchada's CUDA-compatible API surface."""

    _enum: PlatformEnum = PlatformEnum.MUSA
    device_name: str = "musa"
    device_type: str = "musa"


class MusaSRTPlatform(MusaDeviceMixin, SRTPlatform):
    """Default in-tree MUSA SRT platform."""
