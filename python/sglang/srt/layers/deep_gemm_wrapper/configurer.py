import logging

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
