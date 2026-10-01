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
from sglang.srt.layers.modelopt_utils import canonicalize_modelopt_quant_algo
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import get_quantization_config

logger = logging.getLogger(__name__)


@_register_for("BeamForCausalLM")
def _beam_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    # Beam is validated on TRT-LLM MHA (Hopper and SM120 FMHA-v2/XQA,
    # Blackwell trtllm-gen); the generic Hopper default (fa3) is not. Other
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
        quantization = canonicalize_modelopt_quant_algo(quant_config.get("quant_algo"))
    if quantization in ("compressed-tensors", "compressed_tensors"):
        quantization = quant_config.get("format")
    default_moe_runner = {
        "modelopt_fp4": "flashinfer_trtllm",
        "nvfp4-pack-quantized": "flashinfer_trtllm",
    }.get(quantization)
    if (
        default_moe_runner is not None
        and get_platform().is_sm100
        and cfg.moe_runner_backend == "auto"
        and cfg.moe_a2a_backend == "none"
    ):
        logger.info("Use %s as default MoE runner backend for Beam", default_moe_runner)
        overrides["moe_runner_backend"] = default_moe_runner
    return overrides
