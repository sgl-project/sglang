import importlib.util
from enum import Enum
from functools import lru_cache
from typing import Optional

from packaging import version as pkg_version

from sglang.srt.utils import logger


class MoeA2ABackend(Enum):

    STANDARD = "standard"
    DEEPEP = "deepep"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.STANDARD
        for member in cls:
            if value == member.value:
                return member
        raise ValueError(f"No {cls.__name__} member for value {value}")

    def is_deepep(self):
        return self == MoeA2ABackend.DEEPEP

    def is_standard(self):
        return self == MoeA2ABackend.STANDARD


class MoeRunnerBackend(Enum):
    TRITON = "triton"
    TRITON_KERNEL = "triton_kernel"
    FLASHINFER = "flashinfer_trtllm"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_MXFP4 = "flashinfer_mxfp4"

    def is_triton(self):
        return self == MoeRunnerBackend.TRITON

    def is_triton_kernel(self):
        return self == MoeRunnerBackend.TRITON_KERNEL

    def is_flashinfer_trtllm(self):
        return self == MoeRunnerBackend.FLASHINFER

    def is_flashinfer_cutlass(self):
        return self == MoeRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_mxfp4(self):
        return self == MoeRunnerBackend.FLASHINFER_MXFP4


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


MOE_A2A_BACKEND: Optional[MoeA2ABackend] = None
MOE_RUNNER_BACKEND: Optional[MoeRunnerBackend] = None
DEEPEP_MODE: Optional[DeepEPMode] = None
IS_TBO_ENABLED: Optional[bool] = None
TBO_TOKEN_DISTRIBUTION_THRESHOLD: Optional[float] = None
DEEPEP_CONFIG: Optional[str] = None


def initialize_moe_config(
    moe_a2a_backend: Optional[str],
    moe_runner_backend: Optional[str],
    deepep_mode: str,
    deepep_config: Optional[str],
    is_tbo_enabled: bool,
    tbo_token_distribution_threshold: float,
):
    global MOE_A2A_BACKEND
    global MOE_RUNNER_BACKEND
    global DEEPEP_MODE
    global DEEPEP_CONFIG
    global IS_TBO_ENABLED
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD

    MOE_A2A_BACKEND = MoeA2ABackend(moe_a2a_backend)
    MOE_RUNNER_BACKEND = MoeRunnerBackend(moe_runner_backend or "triton")
    DEEPEP_MODE = DeepEPMode(deepep_mode)
    DEEPEP_CONFIG = deepep_config or ""
    IS_TBO_ENABLED = is_tbo_enabled
    TBO_TOKEN_DISTRIBUTION_THRESHOLD = tbo_token_distribution_threshold


def get_moe_a2a_backend() -> MoeA2ABackend:
    global MOE_A2A_BACKEND
    if MOE_A2A_BACKEND is None:
        logger.warning("MOE_A2A_BACKEND is not initialized, using default backend")
        MOE_A2A_BACKEND = MoeA2ABackend(None)
    return MOE_A2A_BACKEND


def get_moe_runner_backend() -> MoeRunnerBackend:
    global MOE_RUNNER_BACKEND
    if MOE_RUNNER_BACKEND is None:
        logger.warning("MOE_RUNNER_BACKEND is not initialized, using triton backend")
        MOE_RUNNER_BACKEND = MoeRunnerBackend("triton")
    return MOE_RUNNER_BACKEND


def get_deepep_mode() -> DeepEPMode:
    global DEEPEP_MODE
    if DEEPEP_MODE is None:
        logger.warning("DEEPEP_MODE is not initialized, using auto mode")
        DEEPEP_MODE = DeepEPMode("auto")
    return DEEPEP_MODE


def get_deepep_config() -> str:
    global DEEPEP_CONFIG
    if DEEPEP_CONFIG is None:
        logger.warning("DEEPEP_CONFIG is not initialized, using default config")
        DEEPEP_CONFIG = ""
    return DEEPEP_CONFIG


def is_tbo_enabled() -> bool:
    global IS_TBO_ENABLED
    if IS_TBO_ENABLED is None:
        logger.warning("IS_TBO_ENABLED is not initialized, using False")
        IS_TBO_ENABLED = False
    return IS_TBO_ENABLED


def get_tbo_token_distribution_threshold() -> float:
    global TBO_TOKEN_DISTRIBUTION_THRESHOLD
    if TBO_TOKEN_DISTRIBUTION_THRESHOLD is None:
        logger.warning(
            "TBO_TOKEN_DISTRIBUTION_THRESHOLD is not initialized, using 0.48"
        )
        TBO_TOKEN_DISTRIBUTION_THRESHOLD = 0.48
    return TBO_TOKEN_DISTRIBUTION_THRESHOLD


@lru_cache(maxsize=1)
def should_use_flashinfer_trtllm_moe():
    result = get_moe_runner_backend().is_flashinfer_trtllm() and (
        not importlib.util.find_spec("flashinfer")
        or pkg_version.parse(__import__("flashinfer").__version__)
        >= pkg_version.parse("0.2.9rc1")
    )
    return result
