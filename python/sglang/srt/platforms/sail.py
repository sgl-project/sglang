"""SAIL device operations for the SRT platform layer.

T-HEAD SAIL exposes a CUDA-compatible API through its torch backend, so
``SailDeviceMixin`` inherits all device ops from ``CudaDeviceMixin`` and
only overrides identity (``_enum``, ``device_name``).
"""

from functools import lru_cache

import torch

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform

_ZW_TAG = "ZW"


@lru_cache(maxsize=1)
def is_sail_available() -> bool:
    """Return True when the local device is a T-HEAD SAIL (device name holds "ZW")."""
    return bool(torch.cuda.is_available() and _ZW_TAG in torch.cuda.get_device_name())


class SailDeviceMixin(CudaDeviceMixin):
    """SAIL device ops — identical surface to CUDA via T-HEAD's torch backend."""

    _enum: PlatformEnum = PlatformEnum.SAIL
    device_name: str = "sail"
    # device_type stays "cuda" — torch.device("cuda") is the only valid
    # device-type string for SAIL devices in PyTorch.


class SailSRTPlatform(SailDeviceMixin, SRTPlatform):
    """Default in-tree T-HEAD SAIL SRT platform."""

    capabilities = PlatformCapabilities(
        supports_triton=True,
        graph_capture=True,
        piecewise_graph=True,
        supports_pynccl=True,
    )

    def register_platform_backends(self) -> None:
        # The SAIL DeepGEMM MoE engines register their fused ops against the
        # shared MoE runner through decorators, so importing the module is the
        # registration. Done here rather than in each quantization method that
        # can select the runner, which keeps the shared paths SAIL-free.
        import sglang.srt.hardware_backend.sail.moe.deepgemm_moe  # noqa: F401
