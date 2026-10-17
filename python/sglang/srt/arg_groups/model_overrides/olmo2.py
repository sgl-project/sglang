"""Config-time override declarations for olmo2."""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for("Olmo2ForCausalLM")
def _olmo2_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    # Olmo3 is served as Olmo2ForCausalLM; Olmo3Config sets is_hybrid_swa.
    text_config = getattr(hf_config, "text_config", None)
    is_hybrid_swa = getattr(hf_config, "is_hybrid_swa", False) or getattr(
        text_config, "is_hybrid_swa", False
    )
    if is_hybrid_swa and cfg.enable_hierarchical_cache:
        logger.warning(
            "Hierarchical cache is enabled; keeping the hybrid SWA memory pool "
            "disabled for Olmo3 despite its sliding/full layer mix."
        )
        is_hybrid_swa = False
    if not is_hybrid_swa:
        logger.warning(
            f"Disabling hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
        )
        overrides["disable_hybrid_swa_memory"] = True
    if cfg.attention_backend is None:
        if get_platform().is_cuda and get_platform().is_sm100:
            overrides["attention_backend"] = "trtllm_mha"
        elif get_platform().is_cuda and get_platform().device_sm >= 80:
            overrides["attention_backend"] = "fa3"
        else:
            overrides["attention_backend"] = "triton"
    return overrides
