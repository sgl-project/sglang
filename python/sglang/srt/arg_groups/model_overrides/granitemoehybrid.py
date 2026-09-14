"""Config-time override declarations for granitemoehybrid.

Architectures: GraniteMoeHybridForCausalLM.
"""

from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform


@_register_for("GraniteMoeHybridForCausalLM")
def _granite_moe_hybrid_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    has_mamba = any(
        layer_type == "mamba" for layer_type in getattr(hf_config, "layer_types", [])
    )
    if has_mamba and get_platform().is_sm100 and cfg.attention_backend is None:
        return {"attention_backend": "flashinfer"}
    return {}
