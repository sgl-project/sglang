"""Config-time override declarations for exaone."""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)

logger = logging.getLogger(__name__)


@_register_for("Exaone4ForCausalLM", "ExaoneMoEForCausalLM")
def _exaone_overrides(server_args: Any, hf_config: Any) -> dict:
    if hf_config.sliding_window_pattern is None:
        return {}
    architecture = hf_config.architectures[0]
    if architecture == "Exaone4ForCausalLM":
        if not resolving_view(server_args).enable_hierarchical_cache:
            return {}
        reason = "with hierarchical cache"
    else:
        reason = "as it is not yet supported"
    logger.warning(f"Disabling hybrid SWA memory for {architecture} {reason}.")
    return {"disable_hybrid_swa_memory": True}
