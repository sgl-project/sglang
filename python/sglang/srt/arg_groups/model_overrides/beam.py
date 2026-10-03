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
    # Beam is validated on TRT-LLM MHA (Hopper and SM120 FMHA-v2/XQA,
    # Blackwell trtllm-gen); the generic Hopper default (fa3) is not. Other
    # platforms keep the generic selection.
    if is_attention_backend_not_set(cfg) and (
        get_platform().is_sm90 or get_platform().is_sm100 or get_platform().is_sm120
    ):
        logger.info("Use trtllm_mha as default attention backend for Beam")
        overrides["attention_backend"] = "trtllm_mha"
    # BeamMoE runs NVFP4 experts only on flashinfer_trtllm; BF16 keeps auto.
    is_quantized = (
        cfg.quantization is not None or get_quantization_config(hf_config) is not None
    )
    if (
        is_quantized
        and get_platform().is_sm100
        and cfg.moe_runner_backend == "auto"
        and cfg.moe_a2a_backend == "none"
    ):
        logger.info("Use flashinfer_trtllm as MoE runner backend for quantized Beam")
        overrides["moe_runner_backend"] = "flashinfer_trtllm"
    return overrides
