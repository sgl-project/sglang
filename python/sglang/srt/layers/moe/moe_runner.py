from typing import Optional

from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend

MOE_A2A_BACKEND = None
# MOE_GROUPED_GEMM_BACKEND = None
DEEPEP_MODE = None


def initialize_moe_runner(
    moe_a2a_backend: str,
    # moe_grouped_gemm_backend: str,
    deepep_mode: Optional[str],
):
    global MOE_A2A_BACKEND
    global DEEPEP_MODE
    MOE_A2A_BACKEND = MoeA2ABackend(moe_a2a_backend)
    DEEPEP_MODE = DeepEPMode(deepep_mode)


def get_moe_a2a_backend():
    return MOE_A2A_BACKEND


def get_deepep_mode():
    return DEEPEP_MODE
