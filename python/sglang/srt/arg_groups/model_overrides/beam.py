"""Config-time override declarations for beam.

Architectures: BeamForCausalLM.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import get_quantization_config

logger = logging.getLogger(__name__)


@_register_for("BeamForCausalLM")
def _beam_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    # Beam serves on TRT-LLM MHA (FMHA-v2/XQA on Hopper and SM120, trtllm-gen
    # on Blackwell) rather than the generic Hopper default (fa3). Other
    # platforms keep the generic selection.
    if is_attention_backend_not_set(cfg) and (
        get_platform().is_sm90 or get_platform().is_sm100 or get_platform().is_sm120
    ):
        logger.info("Use trtllm_mha as default attention backend for Beam")
        overrides["attention_backend"] = "trtllm_mha"
    # Select defaults by checkpoint format; explicit runner choices are preserved.
    quantization = cfg.quantization or get_quantization_config(hf_config)
    quant_config = getattr(hf_config, "quantization_config", None) or {}
    if quantization == "modelopt":
        quantization = quant_config.get("quant_algo")
    if (
        quantization in ("modelopt_fp4", "NVFP4")
        and get_platform().is_sm100
        and cfg.moe_runner_backend == "auto"
        and cfg.moe_a2a_backend == "none"
    ):
        logger.info("Use flashinfer_trtllm as MoE runner backend for quantized Beam")
        overrides["moe_runner_backend"] = "flashinfer_trtllm"
    return overrides
