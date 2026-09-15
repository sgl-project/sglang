"""Config-time override declarations for deepseek_v4.

Architectures: DeepseekV4ForCausalLM.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    model_config_of,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import is_flashinfer_available

logger = logging.getLogger(__name__)


@_register_for("DeepseekV4ForCausalLM")
def _deepseek_v4_overrides(server_args: Any, hf_config: Any) -> dict:
    """Attention, page and MoE defaults; KV-cache dtype, NPU split-backend setup,
    request limits and validation remain in deepseek_v4_hook.
    """
    cfg = resolving_view(server_args)

    model_arch = hf_config.architectures[0]
    overrides: Dict[str, Any] = {"attention_backend": "dsv4"}

    # MXFP8 serves this checkpoint's 32-wide ue8m0 blocks on SM100/SM103;
    # explicit backend choices, including Triton, take precedence.
    quant = getattr(hf_config, "quantization_config", None) or {}
    if (
        getattr(hf_config, "model_type", None) == "deepseek_v41"
        and cfg.device == "cuda"
        and not get_platform().is_hip
        and get_platform().is_sm100
        and cfg.fp8_gemm_runner_backend == "auto"
        and quant.get("quant_method") == "fp8"
        and quant.get("weight_block_size") == [32, 32]
        and quant.get("scale_fmt") == "ue8m0"
        and is_flashinfer_available()
    ):
        overrides["fp8_gemm_runner_backend"] = "flashinfer_cutedsl"
        logger.info("Use flashinfer_cutedsl for DeepSeek-V4.1 MXFP8 dense GEMMs.")

    page_size = 256
    if cfg.device == "npu":
        # NPU keeps the device-aware "dsv4" backend (the registry routes it to
        # the Ascend V4 subclass); only the pool geometry / dtype differ.
        # set_default_server_args() pins all three backends to "ascend" for
        # generic NPU models; override that here so V4 stays consistently on
        # dsv4.
        page_size = 128
        overrides["prefill_attention_backend"] = "dsv4"
        overrides["decode_attention_backend"] = "dsv4"
    overrides["page_size"] = page_size
    logger.info(
        f"Use dsv4 attention backend for {model_arch}, setting page_size to {page_size}."
    )

    if cfg.moe_runner_backend == "auto":
        model_config = model_config_of(server_args)
        # nvidia/DeepSeek-V4-Pro-NVFP4 uses the routed TRT-LLM runner.
        if model_config.nvfp4_moe_meta is not None:
            overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
            logger.info(
                "Use flashinfer_trtllm_routed as MoE runner backend for "
                f"{model_arch} hybrid FP8+NVFP4 checkpoint."
            )
        elif (
            cfg.device == "cuda"
            and not get_platform().is_hip
            and cfg.moe_a2a_backend == "none"
            and not envs.SGLANG_DSV4_FP4_DEQUANT.get()
            and model_config.is_fp4_experts
            and (
                get_platform().is_sm90
                or get_platform().is_sm100
                or get_platform().is_sm120
            )
        ):
            overrides["moe_runner_backend"] = "flashinfer_mxfp4"
            logger.info(f"Use flashinfer_mxfp4 as MoE runner backend for {model_arch}.")
    return overrides
