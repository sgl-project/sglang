"""Semantic feature support for Ascend NPU device families."""

import logging
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)


class NPUDeviceFamily(str, Enum):
    A2 = "a2"
    A3 = "a3"
    A5 = "a5"
    UNKNOWN = "unknown"


class NPUFeature(str, Enum):
    NATIVE_GEMMA_RMS_NORM = "native_gemma_rms_norm"


_DEVICE_FAMILY_RANGES = (
    (220, 225, NPUDeviceFamily.A2),
    (250, 255, NPUDeviceFamily.A3),
    (260, 260, NPUDeviceFamily.A5),
)

_FEATURES_BY_DEVICE_FAMILY = {
    NPUDeviceFamily.A2: frozenset({NPUFeature.NATIVE_GEMMA_RMS_NORM}),
    NPUDeviceFamily.A3: frozenset({NPUFeature.NATIVE_GEMMA_RMS_NORM}),
    NPUDeviceFamily.A5: frozenset(),
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
