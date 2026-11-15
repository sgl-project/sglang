"""Config-time overrides for GigaChat 3.5 and GigaChat 4."""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)

logger = logging.getLogger(__name__)


@_register_for(
    "GigaChat35ForCausalLM",
    "GigaChat35ForCausalLMNextN",
    "GigaChat4ForCausalLM",
    "GigaChat4ForCausalLMNextN",
)
def _gigachat_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {"disable_shared_experts_fusion": True}
    if (
        hf_config.architectures[0]
        in ("GigaChat35ForCausalLM", "GigaChat35ForCausalLMNextN")
        and cfg.speculative_algorithm == "EAGLE"
    ):
        logger.info(
            "Enable multi-layer EAGLE speculative decoding for GigaChat 3.5 model."
        )
        overrides["enable_multi_layer_eagle"] = True
    return overrides
