"""Config-time override declarations for qwen3_5.

Architectures: InternS2MobiusForConditionalGeneration, InternS2PreviewForConditionalGeneration, Qwen3NextForCausalLM, Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration.
"""

from typing import Any

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


@_register_for(
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "InternS2PreviewForConditionalGeneration",
    "InternS2MobiusForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
)
def _qwen3_5_hybrid_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if not get_platform().is_sm100 or cfg.attention_backend is not None:
        return {}
    sm100_default_attn_backend = "triton"
    # trtllm_mha requires speculative_eagle_topk == 1 and page_size > 1.
    # get_default_attn_backend handles the eagle_topk check.
    # There is only one case where page_size=1 is required,
    # which is when radix cache is enabled and both extra_buffer
    # and spec decoding are disabled.
    default_attn_backend = get_default_attn_backend(
        server_args,
        use_mla_backend=use_mla_backend(server_args),
        model_config=model_config_of(server_args),
    )
    # The mamba radix-cache pass runs before this dispatch: read the
    # declared strategy through the view (the legacy branch observed the
    # already-written field here).
    if default_attn_backend == "trtllm_mha" and not (
        not mamba_extra_buffer_of(resolved_view(server_args))
        and not cfg.disable_radix_cache
        and cfg.speculative_algorithm is None
    ):
        sm100_default_attn_backend = "trtllm_mha"
    return {
        "attention_backend": sm100_default_attn_backend,
        "page_size": 64 if sm100_default_attn_backend == "trtllm_mha" else 1,
    }
