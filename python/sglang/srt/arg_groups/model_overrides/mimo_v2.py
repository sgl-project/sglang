"""Config-time override declarations for mimo_v2."""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    model_config_of,
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

    if get_platform().is_sm100 and is_attention_backend_not_set(cfg):
        overrides["attention_backend"] = "fa4"
        logger.info("MiMoV2 on SM100: attention_backend=fa4.")

    # Mixed checkpoints also advertise quant_method=fp8; select the runner
    # using the routed-expert layout already resolved by ModelConfig.
    if (
        get_platform().is_sm100
        and cfg.moe_runner_backend == "auto"
        and get_quantization_config(hf_config) == "fp8"
    ):
        if model_config_of(server_args).is_fp4_experts:
            # Let the all-to-all backend choose its compatible runner.
            if cfg.moe_a2a_backend == "none":
                overrides["moe_runner_backend"] = "flashinfer_mxfp4"
        else:
            # Avoid the slower Triton default for ordinary FP8 checkpoints.
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
        if "moe_runner_backend" in overrides:
            logger.info(
                "MiMoV2 on SM100: moe_runner_backend=%s.",
                overrides["moe_runner_backend"],
            )
    elif (
        get_platform().is_sm90
        and cfg.moe_runner_backend == "auto"
        and get_quantization_config(hf_config) == "fp8"
        and model_config_of(server_args).is_fp4_experts
    ):
        # The auto runner resolves to Triton FP8, which cannot read packed MXFP4.
        overrides["moe_runner_backend"] = "marlin"
        logger.info("MiMoV2 on SM90: moe_runner_backend=marlin.")
    return overrides
