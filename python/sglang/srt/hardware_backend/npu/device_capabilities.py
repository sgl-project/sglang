"""Semantic feature support for Ascend NPU device families."""

import logging
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)


class NPUDeviceFamily(str, Enum):
    ASCEND_910B = "ascend_910b"
    ASCEND_910C = "ascend_910c"
    ASCEND_950 = "ascend_950"
    UNKNOWN = "unknown"


class NPUFeature(str, Enum):
    NATIVE_GEMMA_RMS_NORM = "native_gemma_rms_norm"
    TRITON_GEMMA_RMS_NORM = "triton_gemma_rms_norm"


_DEVICE_FAMILY_RANGES = (
    (220, 225, NPUDeviceFamily.ASCEND_910B),
    (250, 255, NPUDeviceFamily.ASCEND_910C),
    (260, 260, NPUDeviceFamily.ASCEND_950),
)

_FEATURES_BY_DEVICE_FAMILY = {
    NPUDeviceFamily.ASCEND_910B: frozenset({NPUFeature.NATIVE_GEMMA_RMS_NORM}),
    NPUDeviceFamily.ASCEND_910C: frozenset({NPUFeature.NATIVE_GEMMA_RMS_NORM}),
    NPUDeviceFamily.ASCEND_950: frozenset({NPUFeature.TRITON_GEMMA_RMS_NORM}),
    NPUDeviceFamily.UNKNOWN: frozenset(),
}


@lru_cache(maxsize=1)
def get_npu_device_family() -> NPUDeviceFamily:
    """Resolve and cache the current process-wide Ascend device family."""
    try:
        import torch_npu

        soc_version = int(torch_npu.npu.get_soc_version())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
        logger.warning(
            "Unable to resolve the Ascend SoC version; optional NPU features "
            "will use safe fallbacks: %s",
            e,
        )
        return NPUDeviceFamily.UNKNOWN

    for lower, upper, family in _DEVICE_FAMILY_RANGES:
        if lower <= soc_version <= upper:
            return family

    logger.warning(
        "Unknown Ascend SoC version %s; optional NPU features will use safe "
        "fallbacks.",
        soc_version,
    )
    return NPUDeviceFamily.UNKNOWN


def supports_npu_feature(feature: NPUFeature) -> bool:
    """Whether the current Ascend family supports a semantic NPU feature."""
    return feature in _FEATURES_BY_DEVICE_FAMILY[get_npu_device_family()]
