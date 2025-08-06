import importlib.util
from enum import Enum
from functools import lru_cache

from packaging import version as pkg_version

from sglang.srt.layers.moe.moe_runner import get_moe_grouped_gemm_backend


@lru_cache(maxsize=1)
def should_use_flashinfer_trtllm_moe():
    result = get_moe_grouped_gemm_backend().is_flashinfer_trtllm() and (
        not importlib.util.find_spec("flashinfer")
        or pkg_version.parse(__import__("flashinfer").__version__)
        >= pkg_version.parse("0.2.9rc1")
    )
    return result


class MoeA2ABackend(Enum):

    STANDARD = ("standard", "none")
    DEEPEP = "deepep"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.STANDARD
        for member in cls:
            if value in member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value}")

    def is_deepep(self):
        return self == MoeA2ABackend.DEEPEP

    def is_standard(self):
        return self == MoeA2ABackend.STANDARD


class MoeGroupedGemmBackend(Enum):
    TRITON = "triton"
    TRITON_KERNEL = "triton_kernel"
    FLASHINFER = "flashinfer_trtllm"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"

    def is_triton(self):
        return self == MoeGroupedGemmBackend.TRITON

    def is_triton_kernel(self):
        return self == MoeGroupedGemmBackend.TRITON_KERNEL

    def is_flashinfer_trtllm(self):
        return self == MoeGroupedGemmBackend.FLASHINFER

    def is_flashinfer_cutlass(self):
        return self == MoeGroupedGemmBackend.FLASHINFER_CUTLASS


class DeepEPMode(Enum):
    NORMAL = "normal"
    LOW_LATENCY = "low_latency"
    AUTO = "auto"

    def enable_normal(self):
        return self in [DeepEPMode.NORMAL, DeepEPMode.AUTO]

    def enable_low_latency(self):
        return self in [DeepEPMode.LOW_LATENCY, DeepEPMode.AUTO]

    def resolve(self, is_extend_in_batch: bool):
        if self != DeepEPMode.AUTO:
            return self

        if is_extend_in_batch:
            return DeepEPMode.NORMAL
        else:
            return DeepEPMode.LOW_LATENCY
