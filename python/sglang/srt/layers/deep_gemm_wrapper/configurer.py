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
    # DeepGEMM requires TMEM/tcgen05 (SM100+datacenter), not available on SM120
    if sm_version == 120:
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
DEEPGEMM_MASKED_FP8_BACKEND = (
    envs.SGLANG_DEEPGEMM_MASKED_FP8_BACKEND.get().lower()
)
if DEEPGEMM_MASKED_FP8_BACKEND not in ("native", "flashinfer", "cake"):
    raise ValueError(
        "SGLANG_DEEPGEMM_MASKED_FP8_BACKEND must be one of: "
        "native, flashinfer, cake"
    )

# FlashInfer's batch DeepGEMM API and the Cake backend share the public
# float32 groupwise-scale ABI. Keep the existing Blackwell UE8M0 path exactly
# unchanged for the default native backend.
DEEPGEMM_MASKED_FP8_STANDARD_SCALES = DEEPGEMM_MASKED_FP8_BACKEND != "native"
DEEPGEMM_SCALE_UE8M0 = (
    DEEPGEMM_BLACKWELL and not DEEPGEMM_MASKED_FP8_STANDARD_SCALES
)
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)
DEEPGEMM_MASKED_NEED_TMA_ALIGNED_SCALES = (
    DEEPGEMM_NEED_TMA_ALIGNED_SCALES
    and not DEEPGEMM_MASKED_FP8_STANDARD_SCALES
)
