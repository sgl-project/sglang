import importlib.util
from functools import lru_cache
from typing import Optional

from packaging import version as pkg_version

from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend, MoeGroupedGemmBackend

MOE_A2A_BACKEND: MoeA2ABackend = MoeA2ABackend(None)
MOE_GROUPED_GEMM_BACKEND: MoeGroupedGemmBackend = MoeGroupedGemmBackend("triton")
DEEPEP_MODE: DeepEPMode = DeepEPMode("auto")


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
    return MOE_A2A_BACKEND


def get_moe_grouped_gemm_backend() -> MoeGroupedGemmBackend:
    return MOE_GROUPED_GEMM_BACKEND


def get_deepep_mode() -> DeepEPMode:
    return DEEPEP_MODE


@lru_cache(maxsize=1)
def should_use_flashinfer_trtllm_moe():
    result = get_moe_grouped_gemm_backend().is_flashinfer_trtllm() and (
        not importlib.util.find_spec("flashinfer")
        or pkg_version.parse(__import__("flashinfer").__version__)
        >= pkg_version.parse("0.2.9rc1")
    )
    return result
