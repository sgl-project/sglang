import logging
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_musa,
    is_sail,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()
_is_sail = is_sail()


def _sm120_deep_gemm_apis_available() -> bool:
    try:
        import deep_gemm
    except (ImportError, OSError, RuntimeError):
        return False
    return all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "fp8_einsum",
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "transform_sf_into_required_layout",
        )
    )


def _deep_gemm_build() -> Optional[str]:
    """The DeepGEMM build this platform loads, or None when it loads none."""
    if _is_sail:
        return "sail"
    if _is_cuda:
        return "cuda"
    if _is_musa:
        return "musa"
    return None


# The minimum SM a build serves is a property of the DeepGEMM library, not of
# the device: the SAIL build reaches SM80, MUSA numbers from SM31, and the CUDA
# build needs SM90.
_DEEPGEMM_MIN_SM = {"cuda": 90, "musa": 31, "sail": 80}


def _compute_enable_deep_gemm():
    build = _deep_gemm_build()
    if build is None:
        return False
    if not envs.SGLANG_ENABLE_JIT_DEEPGEMM.get():
        return False

    sm_version = get_device_sm()
    if sm_version < _DEEPGEMM_MIN_SM[build]:
        return False
    # SM120/SM121 support (mma.sync block-scale, no TMEM) landed in DeepGEMM#324;
    # probe every API used by the SM120 DSV4 paths since installed builds may
    # expose fp8_einsum but still predate the SM120 kernels. Only the CUDA build
    # tracks upstream DeepGEMM, so the SAIL build is exempt.
    if (
        build == "cuda"
        and sm_version in (120, 121)
        and not _sm120_deep_gemm_apis_available()
    ):
        return False

    try:
        import deep_gemm  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        return False

    return True


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and not _is_sail and get_platform().is_sm100
DEEPGEMM_SCALE_UE8M0 = (
    ENABLE_JIT_DEEPGEMM
    and not _is_sail
    and (get_platform().is_sm100 or get_device_sm() in (120, 121))
)
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa or _is_sail)


def _supports_paged_sparse_mqa_logits() -> bool:
    if not DEEPGEMM_BLACKWELL:
        return False
    import deep_gemm

    return all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "get_paged_sparse_mqa_logits_metadata",
            "fp8_fp4_paged_sparse_mqa_logits",
        )
    )


DEEPGEMM_PAGED_SPARSE_MQA_LOGITS = _supports_paged_sparse_mqa_logits()
