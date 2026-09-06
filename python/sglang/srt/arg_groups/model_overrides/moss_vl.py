"""Config-time override declarations for moss_vl.

Architectures: MossVLForConditionalGeneration.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    attention_backends_of,
    is_attention_backend_not_set,
    resolved_view,
    resolving_view,
)

logger = logging.getLogger(__name__)


@_register_for("MossVLForConditionalGeneration")
def _moss_vl_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    if is_attention_backend_not_set(resolving_view(server_args)):
        overrides["prefill_attention_backend"] = "flashinfer"
        logger.info("Use flashinfer as default prefill attention backend for Moss-VL")
    prefill_backend = (
        overrides.get("prefill_attention_backend")
        or attention_backends_of(resolved_view(server_args))[0]
    )
    assert prefill_backend == "flashinfer", (
        "MossVLForConditionalGeneration requires flashinfer prefill "
        "attention backend for cross-attention custom mask support."
    )
    return overrides
