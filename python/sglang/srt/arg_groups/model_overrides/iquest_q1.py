from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    resolving_view,
)


@_register_for("IQuestQ1ForCausalLM", "IQuestQ1MTP")
def _iquest_q1_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    requested = (
        cfg.attention_backend,
        cfg.prefill_attention_backend,
        cfg.decode_attention_backend,
    )
    if any(backend not in (None, "fa3") for backend in requested):
        raise ValueError(
            "IQuestQ1 applies its attention sink from the softmax LSE and "
            "requires FA3 attention for both prefill and decode."
        )
    if cfg.speculative_algorithm is not None:
        if cfg.speculative_algorithm.upper() not in ("EAGLE", "NEXTN"):
            raise ValueError("IQuest Q1 MTP requires serial EAGLE or NEXTN drafting.")
        if cfg.speculative_draft_attention_backend not in (None, "fa3"):
            raise ValueError("IQuestQ1 requires FA3 attention for the MTP draft.")
    if cfg.attn_cp_size > 1 or cfg.dcp_size > 1 or cfg.enable_prefill_cp:
        raise ValueError("IQuestQ1 does not support context parallelism.")
    if cfg.speculative_eagle_topk is not None and cfg.speculative_eagle_topk > 1:
        raise ValueError(
            "IQuestQ1 speculative decoding requires --speculative-eagle-topk 1."
        )
    if cfg.enable_multi_layer_eagle:
        raise ValueError("IQuest Q1 MTP drafting does not support multi-layer EAGLE.")
    if cfg.enable_dp_attention:
        raise ValueError("IQuest Q1 does not support DP attention.")
    if cfg.pp_size > 1:
        raise ValueError("IQuest Q1 does not support pipeline parallelism.")
    overrides = {}
    if is_attention_backend_not_set(cfg):
        overrides["attention_backend"] = "fa3"
        return overrides
    if cfg.attention_backend is None and cfg.prefill_attention_backend is None:
        overrides["prefill_attention_backend"] = "fa3"
    if cfg.attention_backend is None and cfg.decode_attention_backend is None:
        overrides["decode_attention_backend"] = "fa3"
    return overrides
