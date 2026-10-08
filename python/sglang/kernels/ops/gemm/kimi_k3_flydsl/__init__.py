"""SGLang-maintained Kimi-K3 FlyDSL specializations."""

# AITER owns the FlyDSL toolchain bootstrap and shared tensor/buffer shims.
# Import it before local kernel modules so its vendored FlyDSL path is active.
import aiter as _aiter  # noqa: F401

from .kimi_k3_kda_input_group64 import (
    kimi_k3_kda_input_group64,
    quantize_kimi_k3_kda_input_group64,
    supports_kimi_k3_kda_input_group64,
)

__all__ = [
    "kimi_k3_kda_input_group64",
    "quantize_kimi_k3_kda_input_group64",
    "supports_kimi_k3_kda_input_group64",
]
