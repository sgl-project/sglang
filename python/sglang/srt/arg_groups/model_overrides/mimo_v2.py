"""Config-time override declarations for mimo_v2."""

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


def _has_mxfp4_routed_experts(hf_config: Any) -> bool:
    # Same keys ModelConfig reads to set is_fp4_experts for MiMo-V2: mxfp4
    # checkpoints declare quant_method fp8 and mark the routed experts.
    quantization_config = getattr(hf_config, "quantization_config", None) or {}
    return (
        quantization_config.get("routed_experts_quant_method") == "mxfp4"
        or str(quantization_config.get("store_dtype") or "").lower() == "mxfp4"
    )


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

    # On Blackwell "auto" falls through to the triton fused-MoE runner, ~12%
    # slower at bs=1 decode. FP4 checkpoints use flashinfer_mxfp4 instead,
    # including fp8 ones whose routed experts are mxfp4: the FP8 block-scale
    # runner rejects their weights.
    if (
        get_platform().is_sm100
        and cfg.moe_runner_backend == "auto"
        and get_quantization_config(hf_config) == "fp8"
    ):
        if _has_mxfp4_routed_experts(hf_config):
            overrides["moe_runner_backend"] = "flashinfer_mxfp4"
            logger.info(
                "MiMoV2 FP8 with mxfp4 routed experts on SM100: "
                "moe_runner_backend=flashinfer_mxfp4."
            )
        else:
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info("MiMoV2 FP8 on SM100: moe_runner_backend=flashinfer_trtllm.")
    return overrides
