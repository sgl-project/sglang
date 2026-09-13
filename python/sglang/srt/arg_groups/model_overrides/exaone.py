"""Config-time override declarations for exaone.

Architectures: Exaone4ForCausalLM, ExaoneMoEForCausalLM.
"""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
)

logger = logging.getLogger(__name__)


@_register_for("Exaone4ForCausalLM", "ExaoneMoEForCausalLM")
def _exaone_overrides(server_args: Any, hf_config: Any) -> dict:
    if hf_config.sliding_window_pattern is not None:
        logger.warning(
            f"Disabling hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
        )
        return {"disable_hybrid_swa_memory": True}
    return {}
