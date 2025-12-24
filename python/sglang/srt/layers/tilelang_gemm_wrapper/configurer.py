"""TileLang GEMM Wrapper configuration."""
import logging

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm

logger = logging.getLogger(__name__)


def _compute_enable_tilelang_gemm() -> bool:
    """Compute whether to enable TileLang GEMM.

    Enable conditions:
    1. GPU SM version >= 89 (Ada Lovelace+)
    2. tilelang package installed
    3. SGLANG_ENABLE_TILELANG_GEMM is True (default: True)
    """
    try:
        sm_version = get_device_sm()
        if sm_version < 89:
            logger.debug(
                f"TileLang GEMM disabled: SM version {sm_version} < 89 (requires Ada Lovelace+)"
            )
            return False
    except Exception as e:
        logger.debug(f"TileLang GEMM disabled: failed to get SM version: {e}")
        return False

    try:
        import tilelang  # noqa: F401
    except ImportError:
        logger.debug("TileLang GEMM disabled: tilelang package not installed")
        return False

    enabled = envs.SGLANG_ENABLE_TILELANG_GEMM.get()
    if enabled:
        logger.info("TileLang GEMM enabled")
    else:
        logger.debug("TileLang GEMM disabled via SGLANG_ENABLE_TILELANG_GEMM=0")

    return enabled


ENABLE_TILELANG_GEMM = _compute_enable_tilelang_gemm()

TILELANG_GEMM_CONFIG_DIR = envs.SGLANG_TILELANG_GEMM_CONFIG_DIR.get()
