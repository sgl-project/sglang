"""Ascend NPU identity and capabilities for the SRT platform layer.

Device operations are still reached through the in-tree ``is_npu()`` branches
and ``torch.npu`` directly; this class exists so capability reads and identity
queries resolve for NPU instead of falling to the bare ``SRTPlatform``.
"""

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform


class NpuDeviceMixin(DeviceMixin):
    _enum: PlatformEnum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"


class NpuSRTPlatform(NpuDeviceMixin, SRTPlatform):
    """Default in-tree Ascend NPU SRT platform."""

    capabilities = PlatformCapabilities(
        supports_triton=True,
        graph_capture=True,
        piecewise_graph=True,
    )
