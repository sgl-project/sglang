"""Config-time override declarations for nemotron_h.

Architectures: NemotronHForCausalLM, NemotronHPuzzleForCausalLM.
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


@_register_for("NemotronHForCausalLM", "NemotronHPuzzleForCausalLM")
def _nemotron_h_overrides(server_args: Any, hf_config: Any) -> dict:
    """NemotronH quantization / MoE runner / attention backend defaults
    (absorbed from the retired arg_groups/nemotron_h_hook.py; the mamba radix
    cache handling and the triton-backend assert stay in the arch branch)."""
    cfg = resolving_view(server_args)
    model_arch = hf_config.architectures[0]
    model_config = model_config_of(server_args)
    overrides: Dict[str, Any] = {}

    is_modelopt = model_config.quantization in [
        "modelopt",
        "modelopt_fp8",
        "modelopt_fp4",
        "modelopt_mixed",
    ]
    quantization = cfg.quantization
    if is_modelopt:
        assert model_config.hf_config.mlp_hidden_act == "relu2"
        if model_config.quantization == "modelopt":
            quant_algo = model_config.hf_config.quantization_config["quant_algo"]
            if quant_algo == "MIXED_PRECISION":
                quantization = "modelopt_mixed"
            else:
                quantization = (
                    "modelopt_fp4" if quant_algo == "NVFP4" else "modelopt_fp8"
                )
        else:
            quantization = model_config.quantization
        overrides["quantization"] = quantization

    has_w4a16_moe_layers = False
    if is_modelopt and quantization == "modelopt_mixed":
        has_w4a16_moe_layers = any(
            info.get("quant_algo") == "W4A16_NVFP4" and ".experts." in name
            for name, info in hf_config.quantization_config.get(
                "quantized_layers", {}
            ).items()
        )

    if has_w4a16_moe_layers:
        if cfg.moe_a2a_backend != "none":
            raise ValueError("W4A16_NVFP4 MoE layers require --moe-a2a-backend=none.")
        if cfg.moe_runner_backend not in ("auto", "marlin"):
            raise ValueError(
                "W4A16_NVFP4 MoE layers require --moe-runner-backend=marlin."
            )
        if cfg.moe_runner_backend == "auto":
            overrides["moe_runner_backend"] = "marlin"
            logger.info(
                "Use marlin as MoE runner backend for "
                f"{model_arch} with W4A16_NVFP4 MoE layers"
            )
    elif (is_modelopt or model_config.quantization is None) and (
        cfg.moe_runner_backend == "auto"
    ):
        if get_platform().is_sm100 and cfg.moe_a2a_backend == "none":
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                f"Use flashinfer_trtllm as MoE runner backend on sm100 for {model_arch}"
            )
        elif (
            (
                model_config.quantization in ("modelopt_fp4", "modelopt_mixed")
                or quantization == "modelopt_fp4"
            )
            and get_platform().is_cuda
            and (8, 0) <= get_platform().device_capability < (10, 0)
        ):
            overrides["moe_runner_backend"] = "marlin"
            logger.info(
                "Use marlin as MoE runner backend on SM80-SM90 for "
                f"{model_arch} {model_config.quantization}"
            )
        else:
            overrides["moe_runner_backend"] = "flashinfer_cutlass"

    if get_platform().is_blackwell and is_attention_backend_not_set(cfg):
        if cfg.speculative_algorithm is not None:
            speculative_algorithm = cfg.speculative_algorithm.upper()
            if get_platform().is_sm100 and cfg.speculative_eagle_topk in (
                None,
                1,
            ):
                overrides["attention_backend"] = "trtllm_mha"
                if cfg.page_size is None:
                    overrides["page_size"] = 64
                if cfg.mamba_radix_cache_strategy == "auto":
                    overrides["mamba_radix_cache_strategy"] = "extra_buffer"
                if (
                    cfg.speculative_draft_attention_backend is None
                    and speculative_algorithm in ("EAGLE", "NEXTN", "DSPARK")
                ):
                    overrides["speculative_draft_attention_backend"] = "trtllm_mha"
            else:
                overrides["attention_backend"] = "triton"
                if (
                    cfg.speculative_draft_attention_backend is None
                    and speculative_algorithm in ("EAGLE", "NEXTN", "DFLASH", "DSPARK")
                ):
                    overrides["speculative_draft_attention_backend"] = "flashinfer"
        elif get_platform().is_sm100:
            overrides["attention_backend"] = "trtllm_mha"
    return overrides
