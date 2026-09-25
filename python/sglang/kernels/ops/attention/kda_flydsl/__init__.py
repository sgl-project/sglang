"""SGLang-maintained Kimi-K3 FlyDSL specializations."""

# AITER owns the FlyDSL toolchain bootstrap and shared tensor/buffer shims.
# Import it before local kernel modules so its vendored FlyDSL path is active.
import aiter as _aiter  # noqa: F401

from .kimi_k3_kda_decode import (
    flydsl_kimi_k3_kda_decode,
    flydsl_kimi_k3_kda_decode_with_f_b,
    is_flydsl_kimi_k3_kda_decode_supported,
)

__all__ = [
    "flydsl_kimi_k3_kda_decode",
    "flydsl_kimi_k3_kda_decode_with_f_b",
    "is_flydsl_kimi_k3_kda_decode_supported",
]
