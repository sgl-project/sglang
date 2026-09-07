"""Config-time override declarations for inkling.

Architectures: InklingForConditionalGeneration, InklingForConditionalGenerationMTP.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for(
    "InklingForConditionalGeneration",
    "InklingForConditionalGenerationMTP",
)
def _inkling_overrides(server_args: Any, hf_config: Any) -> dict:
    """Inkling architecture defaults: SWA / mamba KV-pool ratios tuned for the
    hybrid-SWA layout, the extra-buffer mamba strategy, and the unified radix
    tree (which Inkling requires — models/inkling.py asserts it). The full-graph
    prefill default is set separately (inline, before cuda-graph resolution) —
    see ServerArgs.__post_init__ / _apply_inkling_prefill_cuda_graph_default. The
    server-arg defaults each yield to an explicit user value (the ratios read
    `None`, the mamba strategy its "auto" token); the prefill declaration is
    materialized
    before _parse_cuda_graph_config folds cuda_graph_backend_prefill into
    prefill.backend, and an explicit --cuda-graph-backend-prefill /
    --disable-prefill-cuda-graph still wins. The unified-radix env write follows
    the MiniMax-M3 handler precedent (env is not a resolvable server-arg)."""
    cfg = resolving_view(server_args)

    overrides: Dict[str, Any] = {}
    # NOTE: the full-graph prefill default is NOT set here. cuda-graph config is
    # resolved in __post_init__ before declarations are materialized, so a
    # cuda_graph_backend_prefill declared here lands too late (the breakable
    # default would already have been auto-disabled for this multimodal arch).
    # It is set inline before _handle_cuda_graph_config instead.
    if cfg.swa_full_tokens_ratio is None:
        overrides["swa_full_tokens_ratio"] = 0.1
    if cfg.mamba_full_memory_ratio is None:
        overrides["mamba_full_memory_ratio"] = 0.1
    # Inkling requires the extra-buffer mamba strategy (inkling.py asserts
    # enable_mamba_extra_buffer()); the generic "auto" resolution does not cover
    # Inkling, so pin it here. Yields to an explicit --mamba-scheduler-strategy.
    #
    # Compared against the unresolved token rather than the class default: the
    # default only answers "unset" while nothing has declared the field first,
    # and `_mamba_radix_cache_resolution` would -- from the slot just above
    # `collect_model_override_declarations` -- for an architecture whose
    # linear-attention spec sets `uses_mamba_radix_cache`. Inkling has no such
    # spec today, but giving it one would silently stop this pin from firing.
    if cfg.mamba_radix_cache_strategy == "auto":
        overrides["mamba_radix_cache_strategy"] = "extra_buffer"
    # Inkling attention runs only on the fa4 (Blackwell) or triton backends --
    # models/inkling_common/attn.py asserts attention_backend in {fa4, triton}.
    # The generic resolver would otherwise pick trtllm_mha (SM100) / fa3
    # (Hopper), so a bare launch fails on the first attention forward. Pin a
    # supported default when the user left every attention-backend flag unset
    # (mirrors the MiniMax-M3 SM100 fa4-default above); an explicit
    # --attention-backend / --prefill/decode-attention-backend still wins.
    if is_attention_backend_not_set(cfg):
        inkling_attn_backend = "fa4" if get_platform().is_sm100 else "triton"
        overrides["attention_backend"] = inkling_attn_backend
        logger.info(
            f"Use {inkling_attn_backend} as the attention backend for Inkling "
            "(requires fa4 or triton)."
        )
    envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.set(True)
    return overrides
