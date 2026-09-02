"""Config-time override declarations for qwen3_moe.

Architectures: InternS2PreviewForConditionalGeneration, Qwen3MoeForCausalLM, Qwen3NextForCausalLM, Qwen3VLMoeForConditionalGeneration, Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import get_quantization_config

logger = logging.getLogger(__name__)


@_register_for(
    "Qwen3MoeForCausalLM",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "InternS2PreviewForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
)
def _qwen3_moe_family_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if get_platform().is_sm100:
        quant_method = get_quantization_config(hf_config)
        quantization = cfg.quantization
        if (
            quantization is None
            and not server_args._quantization_explicitly_unset
            and quant_method is not None
        ):
            overrides["quantization"] = quant_method
            quantization = quant_method
        if (
            (quantization in ("fp8", "modelopt_fp4") or quantization is None)
            and cfg.moe_a2a_backend == "none"
            and cfg.moe_runner_backend == "auto"
        ):
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on sm100 for "
                f"{hf_config.architectures[0]}"
            )
    return overrides
