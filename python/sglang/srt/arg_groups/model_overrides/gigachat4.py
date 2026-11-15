"""Config-time override declarations for gigachat4.

Unlike GigaChat 3.5 (multi-head MTP, multi-layer EAGLE), GigaChat 4 ships one
MTP head served recurrently by the standard EAGLE worker.
"""

from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import _register_for


@_register_for("GigaChat4ForCausalLM", "GigaChat4ForCausalLMNextN")
def _gigachat4_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {"disable_shared_experts_fusion": True}
    return overrides
