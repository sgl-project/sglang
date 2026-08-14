import logging
from importlib.metadata import PackageNotFoundError, version as _dist_version

from packaging.version import parse as _parse_version

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_musa,
    is_sm100_supported,
    is_sm120_supported,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()


def _is_sm120_capable_deep_gemm() -> bool:
    """Whether the installed deep_gemm ships SM120 (desktop Blackwell) kernels.

    sgl-deep-gemm >= 0.1.5 is the first release with arch-12 support (the
    scale-factor layout transforms in ``transform_sf_into_required_layout``
    and the sm120 JIT GEMM kernels). Earlier releases hard-fail on SM120
    ("Unknown SF transformation" during MXFP4 weight prep, or missing sm120
    cubins). The library's internal ``__version__`` is frozen across releases,
    so gate on the wheel's distribution metadata instead.
    """
    try:
        deep_gemm_version = _parse_version(_dist_version("sgl-deep-gemm"))
    except PackageNotFoundError:
        logger.warning(
            "sgl-deep-gemm is installed from source (no distribution metadata): "
            "cannot verify SM120 support, treating it as unsupported. Install "
            "sgl-deep-gemm>=0.1.5 from PyPI to enable DeepGEMM on SM120."
        )
        return False
    return deep_gemm_version >= _parse_version("0.1.5")


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    if not (_is_cuda or _is_musa):
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    # SM120/SM121 (RTX PRO 6000 / RTX 6000D / RTX 5090 / DGX Spark) needs
    # sgl-deep-gemm >= 0.1.5 for the arch-12 JIT kernels and SF layout
    # transforms. Note MegaMoE itself remains SM90/SM100-only; see
    # ServerArgs._handle_a2a_moe for the megamoe a2a guard.
    if is_sm120_supported() and not _is_sm120_capable_deep_gemm():
        logger.warning(
            "DeepGEMM is disabled on SM%d: sgl-deep-gemm < 0.1.5 has no SM120 "
            "kernels. Upgrade to sgl-deep-gemm>=0.1.5 to enable it.",
            sm_version,
        )
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

# SM120 uses the same packed-UE8M0 (Blackwell-style) scale layout and JIT
# kernel family as SM100, so the UE8M0-scale paths apply to both.
DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and (
    is_sm100_supported() or is_sm120_supported()
)
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)
