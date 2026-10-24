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


# Load SAIL attention hooks when running on SAIL hardware.
# Each module registers @plugin_hook decorators that inject SAIL-native ops
# into the respective attention backends transparently.
if is_sail_available():
    import sglang.srt.hardware_backend.sail.attention.sail_dsa_hooks  # noqa: F401
    import sglang.srt.hardware_backend.sail.attention.sail_fa3_hooks  # noqa: F401
    import sglang.srt.hardware_backend.sail.attention.sail_flashmla_hooks  # noqa: F401
