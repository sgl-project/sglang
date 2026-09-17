"""Config-time override declarations for gemma4.

Architectures: Gemma4ForCausalLM, Gemma4ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    model_config_of,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for(
    "Gemma4ForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4UnifiedForConditionalGeneration",
)
def _gemma4_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    default_attention_backend = "trtllm_mha" if get_platform().is_sm100 else "triton"
    if is_attention_backend_not_set(cfg):
        logger.info(
            f"Use {default_attention_backend} as default attention backend for Gemma4"
        )
        overrides["attention_backend"] = default_attention_backend
    # If only one split backend is set, keep the other side on a
    # Gemma4-compatible fallback instead of letting generic backend selection
    # choose an unsupported backend later.
    elif cfg.attention_backend is None:
        overrides["attention_backend"] = default_attention_backend
    if get_platform().is_sm100 and cfg.moe_runner_backend == "auto":
        if model_config_of(server_args).quantization == "modelopt_fp4":
            overrides["quantization"] = "modelopt_fp4"
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on "
                "SM100 for Gemma-4 (modelopt_fp4)"
            )
    return overrides
