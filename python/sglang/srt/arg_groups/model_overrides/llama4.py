"""Config-time override declarations for llama4.

Architectures: Llama4ForCausalLM, Llama4ForConditionalGeneration.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


# Keep in sync with LLAMA4_MODEL_ARCHS (server_args.py).
@_register_for("Llama4ForConditionalGeneration", "Llama4ForCausalLM")
def _llama4_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if cfg.device == "cpu":
        return {}
    overrides: Dict[str, Any] = {}
    # Auto-select attention backend for Llama4 if not specified
    if cfg.attention_backend is None:
        if get_platform().is_sm100:
            backend, platform = "trtllm_mha", "sm100"
        elif get_platform().is_sm90:
            backend, platform = "fa3", "sm90"
        elif get_platform().is_hip:
            backend, platform = "aiter", "hip"
        elif cfg.device == "xpu":
            backend, platform = "intel_xpu", "xpu"
        else:
            backend, platform = "triton", "other platforms"
        logger.warning(
            f"Use {backend} as attention backend on {platform} for Llama4 model"
        )
        overrides["attention_backend"] = backend
    if get_platform().is_sm100 and cfg.moe_runner_backend == "auto":
        if cfg.quantization in {"fp8", "modelopt_fp8"}:
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on SM100 for Llama4"
            )
    return overrides
