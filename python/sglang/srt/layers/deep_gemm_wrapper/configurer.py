import logging
import os

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_musa,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()


def _sm120_deep_gemm_apis_available() -> bool:
    try:
        import deep_gemm
    except (ImportError, OSError, RuntimeError):
        return False
    if not all(
        callable(getattr(deep_gemm, name, None))
        for name in (
            "fp8_einsum",
            "m_grouped_fp8_fp4_gemm_nt_contiguous",
            "transform_sf_into_required_layout",
        )
    ):
        return False
    # Builds without SM120 kernels export the same APIs and only reject SM120
    # at call time. The JIT compiles from the packaged headers, so require the
    # SM120 FP8 GEMM kernel source those APIs dispatch to.
    package_file = getattr(deep_gemm, "__file__", None)
    if not package_file:
        return False
    return os.path.isfile(
        os.path.join(
            os.path.dirname(package_file),
            "include",
            "deep_gemm",
            "impls",
            "sm120_fp8_fp4_gemm_1d1d.cuh",
        )
    )


def _compute_enable_deep_gemm():
    if not (_is_cuda or _is_musa):
        return False
    if not envs.SGLANG_ENABLE_JIT_DEEPGEMM.get():
        return False

    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    # SM120/SM121 support (mma.sync block-scale, no TMEM) landed in DeepGEMM#324;
    # probe every API used by the SM120 DSV4 paths and the SM120 kernel source,
    # since installed builds may expose these APIs but predate the SM120 kernels.
    if sm_version in (120, 121) and not _sm120_deep_gemm_apis_available():
        return False

    try:
        import deep_gemm  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        return False

    return True


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and get_platform().is_sm100
DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (
    get_platform().is_sm100 or get_device_sm() in (120, 121)
)
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)


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
