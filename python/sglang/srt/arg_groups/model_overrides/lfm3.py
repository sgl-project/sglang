"""Config-time override declarations for LFM3."""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import _register_for, resolving_view
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for(
    "Lfm3ForCausalLM",
    "Lfm3VlForConditionalGeneration",
    "Lfm3AudioForConditionalGeneration",
    "Lfm3OmniForConditionalGeneration",
)
def _lfm3_overrides(server_args: Any, hf_config: Any) -> dict:
    """LFM3 attention defaults: FlashInfer on CUDA, aiter on HIP."""
    cfg = resolving_view(server_args)
    if cfg.speculative_algorithm is not None:
        raise ValueError("LFM3 does not support speculative decoding yet.")
    if cfg.attn_dp_size > 1:
        raise ValueError("LFM3 does not support DP attention yet.")
    updates = {}
    backend = cfg.attention_backend
    if backend is None:
        backend = "aiter" if get_platform().is_hip else "flashinfer"
        updates["attention_backend"] = backend
        logger.info(f"Use {backend} as attention backend for LFM3")

    prefill_backend = cfg.prefill_attention_backend or backend
    if prefill_backend == "flashinfer":
        # FlashInfer's ragged planner ignores SWA within prefill chunks.
        # Use paged prefill until separate SWA/full ragged plans are supported.
        if not envs.SGLANG_FLASHINFER_USE_PAGED.is_set():
            envs.SGLANG_FLASHINFER_USE_PAGED.set(True)
            logger.info(
                "Use FlashInfer paged prefill for LFM3 sliding-window attention"
            )
        elif not envs.SGLANG_FLASHINFER_USE_PAGED.get():
            raise ValueError(
                "LFM3 sliding-window layers require FlashInfer paged prefill; "
                "unset SGLANG_FLASHINFER_USE_PAGED or set it to true."
            )

    return updates
