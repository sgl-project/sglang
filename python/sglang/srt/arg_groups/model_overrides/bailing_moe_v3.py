"""Config-time override declarations for bailing_moe_v3.

Architectures: BailingMoeV3ForCausalLM,
BailingMoeV3VLForConditionalGeneration.
"""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    model_config_of,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for(
    "BailingMoeV3ForCausalLM",
    "BailingMoeV3VLForConditionalGeneration",
)
def _bailing_moe_v3_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if (
        cfg.moe_runner_backend != "auto"
        or cfg.device != "cuda"
        or cfg.moe_a2a_backend != "none"
        or get_platform().is_hip
    ):
        return {}
    if not (
        get_platform().is_sm90 or get_platform().is_sm100 or get_platform().is_sm120
    ):
        return {}

    model_config = model_config_of(server_args)
    if model_config.quantization != "fp8" or not model_config.is_fp4_experts:
        return {}

    model_arch = hf_config.architectures[0]
    logger.info(
        "Bailing V3 mixed FP8/MXFP4 checkpoint: "
        "moe_runner_backend=flashinfer_mxfp4 for %s.",
        model_arch,
    )
    return {"moe_runner_backend": "flashinfer_mxfp4"}
