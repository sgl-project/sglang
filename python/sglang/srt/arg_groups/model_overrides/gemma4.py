"""Config-time override declarations for gemma4."""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    model_config_of,
    resolving_view,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)

_BUILTIN_ATTENTION_BACKENDS = (
    "trtllm_mha",
    "triton",
    "ascend",
    "intel_xpu",
    "intel_amx",
    "aiter",
)


def gemma4_attention_backends() -> tuple[str, ...]:
    """Return the attention backends accepted by Gemma4 on this platform.

    An OOT platform owns the compatibility of its declared default backend.
    Built-in platforms keep Gemma4's model-specific allowlist unchanged.
    """
    if not current_platform.is_out_of_tree():
        return _BUILTIN_ATTENTION_BACKENDS

    platform_default = current_platform.get_default_attention_backend()
    return tuple(dict.fromkeys((*_BUILTIN_ATTENTION_BACKENDS, platform_default)))


def _gemma4_default_attention_backend(is_diffusion: bool) -> str:
    """Preserve built-in defaults while honoring the active OOT platform."""
    if is_diffusion:
        return "triton"
    if current_platform.is_out_of_tree():
        return current_platform.get_default_attention_backend()
    return "trtllm_mha" if get_platform().is_sm100 else "triton"


@_register_for(
    "Gemma4ForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4UnifiedForConditionalGeneration",
    "DiffusionGemmaForBlockDiffusion",
)
def _gemma4_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    is_diffusion = (
        hf_config is not None
        and hf_config.architectures[0] == "DiffusionGemmaForBlockDiffusion"
    )
    default_attention_backend = _gemma4_default_attention_backend(is_diffusion)
    if is_attention_backend_not_set(cfg):
        logger.info(
            f"Use {default_attention_backend} as default attention backend for Gemma4"
        )
        overrides["attention_backend"] = default_attention_backend
    # If only one split backend is set, keep the other side on a
    # Gemma4-compatible fallback instead of letting generic backend selection
    # choose an unsupported backend later.
    elif cfg.attention_backend is None:
        overrides["attention_backend"] = default_attention_backend
    if get_platform().is_sm100 and cfg.moe_runner_backend == "auto":
        if model_config_of(server_args).quantization == "modelopt_fp4":
            overrides["quantization"] = "modelopt_fp4"
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on "
                "SM100 for Gemma-4 (modelopt_fp4)"
            )
    return overrides
