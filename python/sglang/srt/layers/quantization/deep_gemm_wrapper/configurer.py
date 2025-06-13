import logging

from sglang.srt.utils import get_bool_env_var, get_device_sm

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    try:
        import deep_gemm
    except ImportError:
        logger.warning("Failed to import deep_gemm, disable ENABLE_JIT_DEEPGEMM.")
        return False

    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    return get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true")


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

try:
    from deep_gemm import fp8_gemm_nt

    # They have not given a name to this breaking change
    DEEPGEMM_V202506 = True
except ImportError:
    DEEPGEMM_V202506 = False

DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_V202506
