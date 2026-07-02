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


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    # SM120 uses mma.sync.block_scale (no TMEM/tcgen05). Only enable when
    # the SM120-compatible DeepGEMM branch is installed (has MoE grouped GEMM).
    if sm_version == 120:
        try:
            from deep_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous  # noqa: F401
        except (ImportError, AttributeError):
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
DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (is_sm100_supported() or get_device_sm() == 120)
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)
