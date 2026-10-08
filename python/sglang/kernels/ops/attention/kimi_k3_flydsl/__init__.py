"""SGLang-maintained Kimi-K3 FlyDSL specializations."""

# AITER owns the FlyDSL toolchain bootstrap and shared tensor/buffer shims.
# Import it before local kernel modules so its vendored FlyDSL path is active.
import aiter as _aiter  # noqa: F401

from .kimi_k3_mla_gate import kimi_k3_mla_gate, supports_kimi_k3_mla_gate

__all__ = [
    "kimi_k3_mla_gate",
    "supports_kimi_k3_mla_gate",
]
