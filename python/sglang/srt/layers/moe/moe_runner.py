import importlib.util
import logging
from functools import lru_cache
from typing import Optional

from packaging import version as pkg_version

from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend, MoeGroupedGemmBackend

logger = logging.getLogger(__name__)


MOE_A2A_BACKEND: Optional[MoeA2ABackend] = None
MOE_GROUPED_GEMM_BACKEND: Optional[MoeGroupedGemmBackend] = None
DEEPEP_MODE: Optional[DeepEPMode] = None


def initialize_moe_runner(
    moe_a2a_backend: Optional[str],
    moe_grouped_gemm_backend: str,
    deepep_mode: str,
):
    global MOE_A2A_BACKEND
    global MOE_GROUPED_GEMM_BACKEND
    global DEEPEP_MODE

    MOE_A2A_BACKEND = MoeA2ABackend(moe_a2a_backend)
    MOE_GROUPED_GEMM_BACKEND = MoeGroupedGemmBackend(moe_grouped_gemm_backend)
    DEEPEP_MODE = DeepEPMode(deepep_mode)


def get_moe_a2a_backend() -> MoeA2ABackend:
    global MOE_A2A_BACKEND
    if MOE_A2A_BACKEND is None:
        logger.warning("MOE_A2A_BACKEND is not initialized, using default backend")
        MOE_A2A_BACKEND = MoeA2ABackend(None)
    return MOE_A2A_BACKEND


def get_moe_grouped_gemm_backend() -> MoeGroupedGemmBackend:
    global MOE_GROUPED_GEMM_BACKEND
    if MOE_GROUPED_GEMM_BACKEND is None:
        logger.warning(
            "MOE_GROUPED_GEMM_BACKEND is not initialized, using default backend"
        )
        MOE_GROUPED_GEMM_BACKEND = MoeGroupedGemmBackend("triton")
    return MOE_GROUPED_GEMM_BACKEND


def get_deepep_mode() -> DeepEPMode:
    global DEEPEP_MODE
    if DEEPEP_MODE is None:
        logger.warning("DEEPEP_MODE is not initialized, using default mode")
        DEEPEP_MODE = DeepEPMode("auto")
    return DEEPEP_MODE


@lru_cache(maxsize=1)
def should_use_flashinfer_trtllm_moe():
    result = get_moe_grouped_gemm_backend().is_flashinfer_trtllm() and (
        not importlib.util.find_spec("flashinfer")
        or pkg_version.parse(__import__("flashinfer").__version__)
        >= pkg_version.parse("0.2.9rc1")
    )
    return result
