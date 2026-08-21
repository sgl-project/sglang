import logging

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_musa,
    is_sm100_supported,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()


def _is_deep_gemm_supported_cuda_arch(sm_version: int) -> bool:
    """Return whether DeepGEMM implements this CUDA architecture family."""
    return sm_version // 10 in (9, 10)


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if _is_cuda and not _is_deep_gemm_supported_cuda_arch(sm_version):
        return False
    if _is_musa and sm_version < 31:
        return False
    if not (_is_cuda or _is_musa):
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_sm100_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)
