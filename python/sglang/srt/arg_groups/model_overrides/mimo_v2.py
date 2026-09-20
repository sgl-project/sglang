"""Config-time override declarations for mimo_v2.

Architectures: MiMoV2FlashForCausalLM, MiMoV2ForCausalLM.
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


# Keep in sync with MIMO_V2_MODEL_ARCHS (server_args.py / configs/hf_config.py).
@_register_for("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM")
def _mimo_v2_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    if cfg.speculative_algorithm == "EAGLE":
        logger.info("Enable multi-layer EAGLE speculative decoding for MiMoV2 model.")
        overrides["enable_multi_layer_eagle"] = True

    # On Blackwell "auto" falls through to the triton fused-MoE runner, ~12%
    # slower at bs=1 decode. FP4 checkpoints use flashinfer_mxfp4 instead.
    if (
        get_platform().is_sm100
        and cfg.moe_runner_backend == "auto"
        and get_quantization_config(hf_config) == "fp8"
    ):
        overrides["moe_runner_backend"] = "flashinfer_trtllm"
        logger.info("MiMoV2 FP8 on SM100: moe_runner_backend=flashinfer_trtllm.")
    return overrides
