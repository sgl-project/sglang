"""SGLang-maintained Kimi-K3 FlyDSL MoE specializations."""

import functools

# AITER owns the FlyDSL toolchain bootstrap; import it before any kernel module
# so its vendored FlyDSL path is active.
import aiter as _aiter  # noqa: F401


@functools.cache
def is_gfx950() -> bool:
    """Return whether the current GPU is gfx950; False when it cannot be queried."""
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime() == "gfx950"
    except (AssertionError, KeyError, RuntimeError):
        return False
