"""Model-side Ascend implementations used by DeepSeek V4 target and draft."""

import torch

from sglang.srt.hardware_backend.npu.dsv4 import dsv4_mhc as mhc
from sglang.srt.hardware_backend.npu.dsv4 import dsv4_rope as rope
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

__all__ = [
    "mhc",
    "rope",
    "uses_stage_local_draft_weights",
    "needs_fp8_byte_gather",
    "engram_signed_root",
]


def uses_stage_local_draft_weights() -> bool:
    # The 910 export carries MTP-local QuaRot embedding/head weights;
    # the native 950 draft shares the target vocabulary.
    return not is_npu_arch35()


def needs_fp8_byte_gather() -> bool:
    # Ascend 950 implements neither FP8 indexing nor the E8M0 cast.
    return is_npu_arch35()


def engram_signed_root(magnitude: torch.Tensor, dot: torch.Tensor) -> torch.Tensor:
    if dot.device.type != "npu":
        return torch.copysign(magnitude, dot)
    # Preserve 950's numeric comparison; 910 graph capture reads the FP32
    # sign bit, including negative zero, without executing copysign.
    negative = dot < 0 if is_npu_arch35() else dot.view(torch.int32) < 0
    return torch.where(negative, -magnitude, magnitude)
