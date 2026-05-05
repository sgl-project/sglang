import logging
import sys

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_sm100_supported

logger = logging.getLogger(__name__)


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    # SM120 (consumer Blackwell) lacks WGMMA/tcgen05 instructions
    # required by DeepGEMM SM90/SM100 kernels.
    # Block the package entirely to prevent _C.init() from triggering
    # NVCC JIT compilation of tcgen05 kernels on SM120.
    if sm_version == 120:
        import types
        import os

        dg = types.ModuleType("deep_gemm")
        dg.__path__ = []
        dg.__file__ = os.path.join(os.path.dirname(__file__), "_deep_gemm_stub.py")
        dg.__version__ = "0.0.0-blocked-sm120"
        sys.modules["deep_gemm"] = dg
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

# DeepGEMM Blackwell kernels only support SM100 (datacenter), not SM120 (consumer)
DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_sm100_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
