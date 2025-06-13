import logging

from sglang.srt.utils import get_device_sm, get_bool_env_var

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
    DEEPGEMM_REQUIRE_UE8M0 = True
except ImportError:
    DEEPGEMM_REQUIRE_UE8M0 = False
