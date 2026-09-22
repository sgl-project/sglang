"""ROCm device operations for the SRT platform layer.

PyTorch exposes ROCm through the same ``torch.cuda.*`` API surface as CUDA
(HIP is a binary shim, and ``torch.device("rocm")`` does not exist). So
``RocmDeviceMixin`` inherits all device ops from ``CudaDeviceMixin`` and
only overrides identity (``_enum``, ``device_name``).
"""

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform


class RocmDeviceMixin(CudaDeviceMixin):
    """ROCm device ops — identical surface to CUDA via torch.cuda's HIP shim."""

    _enum: PlatformEnum = PlatformEnum.ROCM
    device_name: str = "rocm"
    # device_type stays "cuda" — torch.device("cuda") is the only valid
    # device-type string for HIP devices in PyTorch.


class RocmSRTPlatform(RocmDeviceMixin, SRTPlatform):
    """Default in-tree ROCm SRT platform.

    Capability flags (supports_fp8, support_cuda_graph, support_piecewise_cuda_graph)
    keep the conservative SRTPlatform defaults rather than mirroring CudaSRTPlatform.
    They are currently only consulted in OOT branches gated on is_out_of_tree(),
    so the defaults are behaviorally inert for the in-tree ROCm path. A follow-up
    that migrates AMD-specific gating off legacy is_hip() should set these here.
    """
