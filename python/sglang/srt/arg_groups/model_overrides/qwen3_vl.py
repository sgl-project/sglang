"""Config-time override declarations for qwen3_vl.

Architectures: Qwen3VLForConditionalGeneration.
"""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for("Qwen3VLForConditionalGeneration")
def _qwen3vl_overrides(server_args: Any, hf_config: Any) -> dict:

    cfg = resolving_view(server_args)
    if (
        get_platform().is_hip
        and envs.SGLANG_USE_AITER_UNIFIED_ATTN.get()
        and cfg.page_size is None
    ):
        logger.info(
            "Setting page_size=16 for aiter unified attention on Qwen3VLForConditionalGeneration."
        )
        return {"page_size": 16}
    return {}
