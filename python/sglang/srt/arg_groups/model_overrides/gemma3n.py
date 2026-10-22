"""Config-time override declarations for gemma3n."""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
)

logger = logging.getLogger(__name__)


@_register_for("Gemma3nForCausalLM", "Gemma3nForConditionalGeneration")
def _gemma3n_overrides(server_args: Any, hf_config: Any) -> dict:
    logger.warning(
        f"Disable hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
    )
    return {"disable_hybrid_swa_memory": True}
