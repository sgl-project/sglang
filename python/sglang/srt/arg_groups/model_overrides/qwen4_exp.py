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
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


@_register_for("Qwen4ExpForConditionalGeneration")
def _qwen4_exp_overrides(server_args: Any, hf_config: Any) -> dict:
    """Qwen4-Exp keeps the MoE config under ``text_config``; every layer is
    sparse, so a dense-MLP TP size of 1 only stalls the DP MoE path.

    The arch shares the qwen3_5 hybrid stack but is not registered with
    ``_qwen3_5_hybrid_overrides``: compressed QSA must own ``page_size``
    (two family modules may never declare the same field for one arch), so
    the hybrid family's attention-shape policy is restated here and the
    compressed variant overrides its page choice. Compressed QSA pins
    page_size=64: its compressed cache is addressed as
    ``full_slot // compress_ratio`` (the DSV4 scheme), which requires
    page-aligned full-KV allocation with the page a multiple of the compress
    ratio, and page-granular prefix sharing so shared pages share their
    compressed slots. MambaRadixCache supports page_size > 1 only with the
    mamba extra-buffer strategy, so the QSA pool fails fast at boot when
    neither that nor --disable-radix-cache holds.
    """
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}

    if cfg.ple_offload_embedding is None:
        import torch

        overrides["ple_offload_embedding"] = (
            is_cuda() and model_config_of(server_args).dtype == torch.bfloat16
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
        overrides["page_size"] = (
            64 if sm100_default_attn_backend == "trtllm_mha" else 1
        )

    from sglang.srt.layers.attention.qsa.config import (
        QSA_VARIANT_COMPRESSED,
        parse_qsa_profile,
    )

    profile = parse_qsa_profile(hf_config)
    if profile is not None and profile.variant == QSA_VARIANT_COMPRESSED:
        # Unconditional, like DeepSeek-V4's page-256 declaration: compressed
        # addressing is full_slot // ratio and requires page-aligned
        # allocation on every backend. Do not gate this on
        # mamba_radix_cache_strategy — that field resolves in a later pass
        # (mid-resolution it still holds the unresolved default, which made
        # this declaration silently skip on non-SM100 boxes); if the finally
        # resolved strategy cannot support page > 1, MambaRadixCache's own
        # boot assertion reports it.
        overrides["page_size"] = 64
        logger.info(
            "Setting page size to 64 for compressed QSA "
            "(full//ratio compressed addressing)."
        )
    return overrides
