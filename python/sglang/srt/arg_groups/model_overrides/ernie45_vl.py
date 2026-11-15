"""Config-time override declarations for ernie45_vl."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.runtime_context import attn_dp_enabled_of


@_register_for("Ernie4_5_VLMoeForConditionalGeneration")
def _ernie45_vl_overrides(server_args: Any, hf_config: Any) -> dict:
    if attn_dp_enabled_of(resolving_view(server_args)):
        raise ValueError(
            "ERNIE 4.5 VL MoE does not support DP attention: its attention "
            "heads are split over the full TP group."
        )
    return {}
