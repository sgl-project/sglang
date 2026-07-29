"""Semantic hardware features exposed by SGLang platforms."""

from enum import Enum


class PlatformFeature(str, Enum):
    """Optional hardware/runtime features queried through ``current_platform``."""

    NPU_NATIVE_GEMMA_RMS_NORM = "npu.native_gemma_rms_norm"


__all__ = ["PlatformFeature"]
