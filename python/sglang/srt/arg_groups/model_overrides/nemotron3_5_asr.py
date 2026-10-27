"""Config-time override declarations for Nemotron 3.5 ASR.

Architectures: Nemotron3_5AsrForRNNT.
"""

from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)


@_register_for("Nemotron3_5AsrForRNNT")
def _nemotron3_5_asr_overrides(server_args: Any, hf_config: Any) -> dict[str, Any]:
    cfg = resolving_view(server_args)
    # Decoder state lives in the model per request, keyed by pool slot, so a
    # cached token prefix cannot stand in for running the model on it.
    # CUDA graphs are disabled in handle_model_capability_adjustments.
    overrides: dict[str, Any] = {"disable_radix_cache": True}
    # Nothing is written to the KV pool, so it only needs to hold the tokens of
    # running requests (a 30 s clip transcribes to well under 1k tokens).
    # Sized by memory instead, the placeholder's tiny per-token footprint
    # yields ~10^8 slots, and buffers sized by token count no longer fit.
    if cfg.max_total_tokens is None:
        overrides["max_total_tokens"] = 1 << 20
    # The model never calls attention; Triton needs no kernel JIT for the
    # placeholder head shape.
    if is_attention_backend_not_set(cfg):
        overrides["attention_backend"] = "triton"
    return overrides
