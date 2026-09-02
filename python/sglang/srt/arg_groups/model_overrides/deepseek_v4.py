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

logger = logging.getLogger(__name__)


@_register_for("DeepseekV4ForCausalLM")
def _deepseek_v4_overrides(server_args: Any, hf_config: Any) -> dict:
    """DeepSeek V4 attention/page/window/MoE-runner defaults (from
    arg_groups/deepseek_v4_hook.py). The kv-cache dtype and NPU split-backend
    writes, the max_running_requests fill and the validations stay in the
    hook at its legacy slot."""
    cfg = resolving_view(server_args)
    from sglang.srt.server_args import ServerArgs

    model_arch = hf_config.architectures[0]
    overrides: Dict[str, Any] = {"attention_backend": "dsv4"}

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

    if cfg.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio:
        overrides["swa_full_tokens_ratio"] = 0.1
        logger.info(f"Setting swa_full_tokens_ratio to 0.1 for {model_arch}.")

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
            logger.info(
                "Use flashinfer_mxfp4 as MoE runner backend for " f"{model_arch}."
            )
    return overrides
