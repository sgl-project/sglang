"""Config-time override declarations for lfm2."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform


@_register_for(
    "Lfm2ForCausalLM",
    "Lfm2MoeForCausalLM",
    "Lfm2VlForConditionalGeneration",
)
def _lfm2_overrides(server_args: Any, hf_config: Any) -> dict:
    # LFM2-VL reuses the LFM2 language model; its vision tower keeps the mm default.
    cfg = resolving_view(server_args)
    if get_platform().is_sm100 and cfg.attention_backend is None:
        return {"attention_backend": "flashinfer"}
    return {}
