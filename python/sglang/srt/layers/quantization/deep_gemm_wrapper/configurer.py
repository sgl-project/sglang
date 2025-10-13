import logging

from sglang.srt.utils import get_bool_env_var, get_device_sm, is_blackwell

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    try:
        import deep_gemm
    except ImportError:
        return False

    return get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true")


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
