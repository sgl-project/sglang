"""SGLang-maintained Kimi-K3 FlyDSL specializations."""

# AITER owns the FlyDSL toolchain bootstrap and shared tensor/buffer shims.
# Import it before local kernel modules so its vendored FlyDSL path is active.
import aiter as _aiter  # noqa: F401

from .kimi_k3_kda_input_group64 import (
    kimi_k3_kda_input_group64,
    quantize_kimi_k3_kda_input_group64,
    supports_kimi_k3_kda_input_group64,
)
from .kimi_k3_mla_gate import kimi_k3_mla_gate, supports_kimi_k3_mla_gate
from .kimi_k3_moe_preroute_fp8 import (
    kimi_k3_moe_tri_projection_fp8,
    kimi_k3_shared_down_fp8,
    supports_kimi_k3_moe_tri_projection_fp8,
    supports_kimi_k3_shared_down_fp8,
    supports_kimi_k3_shared_down_fp8_weight,
)
from .latent_moe_tail_fp8 import (
    latent_moe_tail_fp8,
    quantize_latent_moe_tail_weight,
    supports_latent_moe_tail_fp8,
)

__all__ = [
    "kimi_k3_kda_input_group64",
    "kimi_k3_mla_gate",
    "kimi_k3_moe_tri_projection_fp8",
    "kimi_k3_shared_down_fp8",
    "latent_moe_tail_fp8",
    "quantize_kimi_k3_kda_input_group64",
    "quantize_latent_moe_tail_weight",
    "supports_kimi_k3_kda_input_group64",
    "supports_kimi_k3_mla_gate",
    "supports_kimi_k3_moe_tri_projection_fp8",
    "supports_kimi_k3_shared_down_fp8",
    "supports_kimi_k3_shared_down_fp8_weight",
]
