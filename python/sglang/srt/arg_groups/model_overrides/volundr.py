"""Config-time override declarations for volundr."""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)

logger = logging.getLogger(__name__)


@_register_for("VolundrForCausalLM", "VolundrForConditionalGeneration")
def _volundr_overrides(server_args: Any, hf_config: Any) -> dict:
    """Volundr serving defaults (models/volundr.py checks them at the first forward):

    * page size 1: the BCSA far field pools K/V straight out of the token-indexed KV cache.
    * radix cache off: Engram's per-request previous-token table is not carried across a prefix
      hit.
    * the FlashAttention-3 backend: the BCSA sliding window and its log-sum-exp come from one
      windowed FA3 call. An explicit attention-backend flag still wins (and fails at startup if
      unsupported).
    """
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {"page_size": 1}
    if not cfg.disable_radix_cache:
        logger.info(
            "Volundr: disabling the radix cache (not yet supported for Engram)."
        )
        overrides["disable_radix_cache"] = True
    if is_attention_backend_not_set(cfg):
        overrides["attention_backend"] = "fa3"
    return overrides
