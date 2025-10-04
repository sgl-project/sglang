import logging

import torch

from sglang.srt.utils import get_bool_env_var, get_device_sm, is_blackwell
from sglang.utils import is_in_ci

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if sm_version < 90:
        return False
    if sm_version == 90 and torch.version.cuda is not None and is_in_ci():
        # Temporarily disable DeepGEMM for non-12.8/12.9 CUDA versions in CI
        # Should be removed after this issue if fixed
        if torch.version.cuda != "12.8" and torch.version.cuda != "12.9":
            return False
    try:
        import deep_gemm
    except ImportError:
        return False

    return get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true")


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
