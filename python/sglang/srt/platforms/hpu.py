"""Intel Gaudi (HPU) identity for the SRT platform layer."""

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform


class HpuDeviceMixin(DeviceMixin):
    _enum: PlatformEnum = PlatformEnum.HPU
    device_name: str = "hpu"
    device_type: str = "hpu"


class HpuSRTPlatform(HpuDeviceMixin, SRTPlatform):
    """Default in-tree HPU SRT platform."""
