"""Config-time override declarations for gemma2_gemma3."""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    attention_backends_of,
    get_default_attn_backend,
    model_config_of,
    resolving_view,
    use_mla_backend,
)

logger = logging.getLogger(__name__)


def _resolved_attention_backends(server_args: Any) -> tuple:
    """(prefill, decode) attention backends with unset sides defaulted."""
    prefill, decode = attention_backends_of(resolving_view(server_args))
    if prefill is None or decode is None:
        default = get_default_attn_backend(
            server_args,
            use_mla_backend=use_mla_backend(server_args),
            model_config=model_config_of(server_args),
        )
        prefill, decode = prefill or default, decode or default
    return prefill, decode


@_register_for(
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
)
def _gemma2_gemma3_overrides(server_args: Any, hf_config: Any) -> dict:
    architecture = hf_config.architectures[0]
    if architecture in ("Gemma3nForCausalLM", "Gemma3nForConditionalGeneration"):
        reason = "as it is not yet supported"
    elif resolving_view(server_args).enable_hierarchical_cache:
        reason = "with hierarchical cache"
    else:
        backends = _resolved_attention_backends(server_args)
        if "aiter" in backends:
            reason = "on the aiter attention backend"
        elif architecture != "Gemma2ForCausalLM" and "fa3" in backends:
            # https://github.com/sgl-project/sglang/issues/9123
            reason = "on the fa3 attention backend"
        else:
            return {}
    logger.warning(f"Disable hybrid SWA memory for {architecture} {reason}.")
    return {"disable_hybrid_swa_memory": True}
