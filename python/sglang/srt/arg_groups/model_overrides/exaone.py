"""Config-time override declarations for exaone."""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for("Exaone4ForCausalLM", "ExaoneMoEForCausalLM", "ExaoneMoeForCausalLM")
def _exaone_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    if hf_config.sliding_window_pattern is not None:
        logger.warning(
            f"Disabling hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
        )
        overrides["disable_hybrid_swa_memory"] = True
        if resolving_view(server_args).attention_backend is None:
            if get_platform().is_cuda and get_platform().is_sm100:
                overrides["attention_backend"] = "trtllm_mha"
            elif get_platform().is_cuda and get_platform().device_sm >= 80:
                overrides["attention_backend"] = "fa3"
            else:
                overrides["attention_backend"] = "triton"
    return overrides
