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


def _lora_wraps_experts(server_args: Any) -> bool:
    """Whether LoRA wraps the routed experts: LoRAManager wraps FusedMoE when its
    targets hold both gate_up_proj and down_proj ("all" holds every module)."""
    from sglang.srt.arg_groups.lora_hook import parse_lora_paths
    from sglang.srt.lora.lora_config import LoRAConfig
    from sglang.srt.lora.utils import get_normalized_target_modules

    cfg = resolving_view(server_args)
    if not (cfg.enable_lora or (cfg.enable_lora is None and cfg.lora_paths)):
        return False
    try:
        if cfg.lora_target_modules:
            targets = get_normalized_target_modules(cfg.lora_target_modules)
        else:
            targets = set()
            for lora_ref in parse_lora_paths(cfg.lora_paths):
                config = LoRAConfig(lora_ref.lora_path)
                targets |= get_normalized_target_modules(config.target_modules)
    except Exception as e:
        # Keep the expert checks for targets that cannot be read here.
        logger.warning(
            "MiMoV2: cannot resolve the LoRA target modules (%s); treating them "
            "as including the experts.",
            e,
        )
        return True
    return "all" in targets or {"gate_up_proj", "down_proj"} <= targets


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

    if (
        get_quantization_config(hf_config) == "fp8"
        and model_config_of(server_args).is_fp4_experts
        and _lora_wraps_experts(server_args)
    ):
        # Only the Marlin runner applies expert LoRA to MXFP4 experts, and only
        # with the standard token dispatcher.
        if (
            cfg.moe_runner_backend not in ("auto", "marlin")
            or cfg.moe_a2a_backend != "none"
        ):
            raise ValueError(
                "MiMoV2 expert LoRA on MXFP4 experts requires --moe-runner-backend "
                f"marlin and --moe-a2a-backend none, got {cfg.moe_runner_backend} "
                f"and {cfg.moe_a2a_backend}. LoRA targets on attention only (q/k/v/o_proj, "
                "from --lora-target-modules or the adapter configs) keep other runners."
            )
        if cfg.moe_runner_backend == "auto":
            overrides["moe_runner_backend"] = "marlin"
            logger.info(
                "MiMoV2 expert LoRA on MXFP4 experts: moe_runner_backend=marlin."
            )
        return overrides

    # Mixed checkpoints also advertise quant_method=fp8; select the runner
    # using the routed-expert layout already resolved by ModelConfig.
    if (
        get_platform().is_sm100
        and cfg.moe_runner_backend == "auto"
        and get_quantization_config(hf_config) == "fp8"
    ):
        # Let the all-to-all backend choose its compatible runner.
        if cfg.moe_a2a_backend == "none":
            if model_config_of(server_args).is_fp4_experts:
                overrides["moe_runner_backend"] = "flashinfer_mxfp4"
            else:
                # Avoid the slower Triton default for ordinary FP8 checkpoints.
                overrides["moe_runner_backend"] = "flashinfer_trtllm"
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
