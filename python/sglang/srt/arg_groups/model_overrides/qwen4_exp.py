"""Config-time override declarations for qwen4_exp.

Architectures: Qwen4ExpForConditionalGeneration.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    get_default_attn_backend,
    mamba_extra_buffer_of,
    model_config_of,
    resolved_view,
    resolving_view,
    use_mla_backend,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for("Qwen4ExpForConditionalGeneration")
def _qwen4_exp_overrides(server_args: Any, hf_config: Any) -> dict:
    """Compressed QSA must own ``page_size`` here,
    so the qwen3_5 hybrid attention-shape policy is restated rather than shared.
    page_size=64 needs page-aligned full-KV allocation (slots are full_slot // ratio),
    which MambaRadixCache allows only with mamba extra-buffer or --disable-radix-cache.
    """
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}

    if cfg.ple_offload_embedding is None:
        import torch

        overrides["ple_offload_embedding"] = (
            get_platform().is_cuda
            and model_config_of(server_args).dtype == torch.bfloat16
        )

    text_config = getattr(hf_config, "text_config", hf_config)
    if (
        getattr(text_config, "num_experts", None) is not None
        and cfg.moe_dense_tp_size == 1
    ):
        overrides["moe_dense_tp_size"] = None

    if get_platform().is_sm100 and cfg.attention_backend is None:
        sm100_default_attn_backend = "triton"
        default_attn_backend = get_default_attn_backend(
            server_args,
            use_mla_backend=use_mla_backend(server_args),
            model_config=model_config_of(server_args),
        )
        if default_attn_backend == "trtllm_mha" and not (
            not mamba_extra_buffer_of(resolved_view(server_args))
            and not cfg.disable_radix_cache
            and cfg.speculative_algorithm is None
        ):
            sm100_default_attn_backend = "trtllm_mha"
        overrides["attention_backend"] = sm100_default_attn_backend
        overrides["page_size"] = 64 if sm100_default_attn_backend == "trtllm_mha" else 1

    from sglang.srt.layers.attention.qsa.config import (
        QSA_VARIANT_COMPRESSED,
        parse_qsa_profile,
    )

    profile = parse_qsa_profile(hf_config)
    if profile is not None and profile.variant == QSA_VARIANT_COMPRESSED:
        # Compressed slot = full_slot // ratio; all backends need page-aligned pages.
        # mamba_radix_cache_strategy resolves later, so do not gate on it.
        overrides["page_size"] = 64
        logger.info(
            "Setting page size to 64 for compressed QSA "
            "(full//ratio compressed addressing)."
        )
    return overrides
