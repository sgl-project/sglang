import logging

import torch

from sglang.srt.utils import get_bool_env_var, get_device_sm

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    try:
        import deep_gemm
    except ImportError:
        logger.warning("Failed to import deep_gemm, disable ENABLE_JIT_DEEPGEMM.")
        return False

    return get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true")


def _is_blackwell_arch() -> bool:
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major == 10


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and _is_blackwell_arch()
# Allow disabling UE8M0 scaling for accuracy-critical workloads
# This can help with DeepSeek EP accuracy issues on B200 GPUs
# Will be updated by server args in update_deepgemm_scale_ue8m0()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL and get_bool_env_var(
    "SGL_ENABLE_DEEPGEMM_UE8M0", default="true"
)


def update_deepgemm_scale_ue8m0(disable_ue8m0: bool):
    """Update DEEPGEMM_SCALE_UE8M0 based on server arguments."""
    global DEEPGEMM_SCALE_UE8M0
    if disable_ue8m0:
        DEEPGEMM_SCALE_UE8M0 = False
        logger.info("DeepGEMM UE8M0 scaling disabled via server argument")
    else:
        DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL and get_bool_env_var(
            "SGL_ENABLE_DEEPGEMM_UE8M0", default="true"
        )
