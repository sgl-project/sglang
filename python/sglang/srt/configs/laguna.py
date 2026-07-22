# coding=utf-8
# Copyright 2023-2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Laguna (poolside/Laguna-XS.2) model configuration."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


def _first_not_none(*candidates: Any) -> Any:
    """First non-None candidate. Unlike `a or b`, preserves falsy values."""
    return next((c for c in candidates if c is not None), None)


def normalize_gating(value: Any) -> Literal["per-head", "per-element", "disabled"]:
    if value in (True, "per-head"):
        return "per-head"
    if value == "per-element":
        return "per-element"
    if value in (False, None, "disabled"):
        return "disabled"
    raise ValueError(
        "gating must be one of True, False, None, "
        '"per-head", "per-element", or "disabled"; '
        f"got {value!r}."
    )


def _to_sglang_rope_scaling(rope_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """HF per-layer rope dict → SGLang `get_rope` `rope_scaling`. None means plain RoPE."""
    if not rope_params:
        return None
    rope_type = rope_params.get("rope_type") or rope_params.get("type")
    if rope_type in (None, "default"):
        return None

    out: Dict[str, Any] = {"rope_type": rope_type}
    pass_through = (
        "factor",
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "extrapolation_factor",
        "truncate",
        "low_freq_factor",
        "high_freq_factor",
        "mscale",
        "mscale_all_dim",
        "short_factor",
        "long_factor",
        "short_mscale",
        "long_mscale",
    )
    for key in pass_through:
        if key in rope_params:
            out[key] = rope_params[key]
    if "attention_factor" in rope_params:
        # HF spells it attention_factor; SGLang's factory reads attn_factor.
        out["attn_factor"] = rope_params["attention_factor"]
    return out


class LagunaConfig(PretrainedConfig):
    model_type = "laguna"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        gating: bool | str = True,
        sliding_window: int = 512,
        layer_types: Optional[List[str]] = None,
        mlp_layer_types: Optional[List[str]] = None,
        num_attention_heads_per_layer: Optional[List[int]] = None,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 512,
        moe_routed_scaling_factor: float = 1.0,
        moe_router_logit_softcapping: float = 0.0,
        moe_apply_router_weight_on_input: bool = False,
        # Per-layer-type rope dict; nested under "full_attention" / "sliding_attention".
        rope_parameters: Optional[Dict[str, Any]] = None,
        partial_rotary_factor: Optional[float] = None,
        rope_theta: Optional[float] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        bos_token_id: Optional[int] = 2,
        eos_token_id: Optional[Any] = None,
        pad_token_id: Optional[int] = 9,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.gating = normalize_gating(gating)
        self.sliding_window = sliding_window

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.moe_routed_scaling_factor = moe_routed_scaling_factor
        self.moe_router_logit_softcapping = moe_router_logit_softcapping
        self.moe_apply_router_weight_on_input = moe_apply_router_weight_on_input

        # Synthesise per-layer schedules when the caller omits them so the model
        # file can index by layer_id without per-call guards.
        self.layer_types = (
            list(layer_types)
            if layer_types
            else ["full_attention" for _ in range(num_hidden_layers)]
        )
        self.mlp_layer_types = (
            list(mlp_layer_types)
            if mlp_layer_types
            else (["dense"] + ["sparse"] * (num_hidden_layers - 1))
        )
        self.num_attention_heads_per_layer = (
            list(num_attention_heads_per_layer)
            if (num_attention_heads_per_layer)
            else [num_attention_heads] * num_hidden_layers
        )
        if len(self.num_attention_heads_per_layer) != num_hidden_layers:
            raise ValueError(
                "num_attention_heads_per_layer must have one entry per layer: "
                f"expected num_hidden_layers={num_hidden_layers}, "
                f"got {len(self.num_attention_heads_per_layer)}."
            )

        # SGLang's hybrid-SWA core reads `swa_*` KV/head_dim from hf_text_config.
        # Per-layer Q-head count is read directly from num_attention_heads_per_layer.
        # DFlash draft configs can be all-SWA. In that case there is no full
        # layer geometry to expose, so use layer 0 for the default attention
        # fields and keep per-layer Q-head geometry explicit.
        full_idx = (
            self.layer_types.index("full_attention")
            if "full_attention" in self.layer_types
            else 0
        )
        self.num_attention_heads = self.num_attention_heads_per_layer[full_idx]
        self.swa_num_key_value_heads = num_key_value_heads
        self.swa_head_dim = head_dim
        self.swa_v_head_dim = head_dim

        # Released checkpoint nests rope_parameters under layer-type keys.
        rp = rope_parameters if isinstance(rope_parameters, dict) else {}
        has_full_attention = "full_attention" in self.layer_types
        swa_rp = rp.get("sliding_attention") or {}
        full_rp = rp.get("full_attention") or (swa_rp if not has_full_attention else {})

        # transformers v5 aliases `rope_scaling` ↔ `rope_parameters` on
        # PretrainedConfig — writing one clobbers the other. Keep the nested
        # form on those two slots (so HF's reference modeling code can index
        # rope_parameters[layer_type] when invoked via trust_remote_code) and
        # publish our SGLang-shaped flat rope dicts under different names.
        self.rope_parameters = rope_parameters

        self.rope_theta = _first_not_none(
            full_rp.get("rope_theta"), rope_theta, 10000.0
        )
        self.partial_rotary_factor = _first_not_none(
            full_rp.get("partial_rotary_factor"), partial_rotary_factor, 1.0
        )
        self.full_rope_scaling = _first_not_none(
            _to_sglang_rope_scaling(full_rp), rope_scaling
        )

        self.swa_rope_theta = _first_not_none(swa_rp.get("rope_theta"), self.rope_theta)
        self.swa_partial_rotary_factor = _first_not_none(
            swa_rp.get("partial_rotary_factor"), self.partial_rotary_factor
        )
        self.swa_rope_scaling = _to_sglang_rope_scaling(swa_rp)

        # DeepSeek-style aliases consumed by cross-cutting infra outside this
        # model file: `lora/mem_pool.py` and `lora/utils.py` read
        # `n_routed_experts` / `n_shared_experts` / `first_k_dense_replace`,
        # `elastic_ep/expert_backup_*` reads `n_routed_experts`. The
        # hardcoded `n_shared_experts=1` and `norm_topk_prob=True` reflect
        # Laguna's fixed architecture (one shared expert, sigmoid-renormalized
        # top-k routing).
        self.n_routed_experts = num_experts
        self.n_shared_experts = 1
        self.routed_scaling_factor = moe_routed_scaling_factor
        self.norm_topk_prob = True
        self.first_k_dense_replace = (
            self.mlp_layer_types.index("sparse")
            if "sparse" in self.mlp_layer_types
            else num_hidden_layers
        )
