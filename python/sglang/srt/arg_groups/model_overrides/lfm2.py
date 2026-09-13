"""Config-time override declarations for lfm2.

Architectures: Lfm2ForCausalLM, Lfm2MoeForCausalLM, Lfm2VlForConditionalGeneration.
"""

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
    # LFM2-VL shares the hybrid ShortConv language model, so it takes the same
    # SM100 language attention default. The vision tower keeps the multimodal
    # attention default.
    cfg = resolving_view(server_args)
    if get_platform().is_sm100 and cfg.attention_backend is None:
        return {"attention_backend": "flashinfer"}
    return {}
