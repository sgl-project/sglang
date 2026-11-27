"""Config-time override declarations for gigachat4."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import _register_for


@_register_for("GigaChat4ForCausalLM", "GigaChat4ForCausalLMNextN")
def _gigachat4_overrides(server_args: Any, hf_config: Any) -> dict:
    return {"disable_shared_experts_fusion": True}
