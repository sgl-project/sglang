"""FFN spec of a draft: dense or MoE per layer, read from the draft's own config."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import msgspec

from sglang.srt.layers.draft.config_utils import (
    cfg_get,
    is_int,
    nested_dict,
    optional_positive_int,
    required_bool,
    required_positive_int,
    text_config,
)

FFN_KIND_DENSE = "dense"
FFN_KIND_MOE = "moe"
FFN_KINDS = (FFN_KIND_DENSE, FFN_KIND_MOE)
_EXPERT_COUNT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts")
# What sglang's MoE stack implements: TopK scoring, FusedMoE activations, top-k methods.
_SCORING_FUNCS = ("softmax", "sigmoid", "sqrtsoftplus")
_HIDDEN_ACTS = ("silu", "swigluoai")
_TOPK_METHODS = (None, "greedy", "noaux_tc")


def _expert_count(text: Any) -> Optional[int]:
    """The routed expert count under any of its HF names; None for a dense draft."""
    counts = {
        key: optional_positive_int(cfg_get(text, key, None), key)
        for key in _EXPERT_COUNT_KEYS
    }
    present = {key: value for key, value in counts.items() if value is not None}
    if len(set(present.values())) > 1:
        raise ValueError(
            f"Draft FFN: the expert count is declared twice with different "
            f"values: {present}."
        )
    return next(iter(present.values()), None)


def _optional_positive_float(value: Any, field_name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(
            f"Draft FFN: {field_name} must be a positive number, got {value!r}."
        )
    return float(value)


def _kind_list(raw: Any, key: str, names: dict) -> Tuple[str, ...]:
    """A per-layer list of kind names, mapped to FFN_KINDS."""
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise ValueError(
            f"Draft FFN: {key} must be a non-empty list of {tuple(names)}, got {raw!r}."
        )
    for index, name in enumerate(raw):
        if name not in names:
            raise ValueError(
                f"Draft FFN: {key}[{index}]={name!r} is not one of {tuple(names)}."
            )
    return tuple(names[name] for name in raw)


class DraftDenseFFNSpec(msgspec.Struct, frozen=True, kw_only=True):
    intermediate_size: int
    hidden_act: str

    @classmethod
    def from_config(cls, draft_hf_config: Any) -> DraftDenseFFNSpec:
        text = text_config(draft_hf_config)
        return cls(
            intermediate_size=required_positive_int(
                cfg_get(text, "intermediate_size", None),
                "intermediate_size",
                "for dense layers",
            ),
            hidden_act=str(cfg_get(text, "hidden_act", "silu")),
        )


class DraftMoeFFNSpec(msgspec.Struct, frozen=True, kw_only=True):
    """The HF MoE semantics sglang's TopK and FusedMoE can run, family-neutral."""

    num_experts: int
    top_k: int
    intermediate_size: int
    scoring_func: str  # softmax, sigmoid or sqrtsoftplus
    norm_topk_prob: bool  # renormalize the chosen experts' scores to sum to one
    expert_bias: bool  # the checkpoint carries a trained expert-selection bias
    num_expert_group: Optional[int]  # None: plain top-k, no group limit
    topk_group: Optional[int]
    routed_scaling_factor: float
    hidden_act: str
    swiglu_limit: Optional[float]  # None: no clamp on the expert activation
    swiglu_alpha: Optional[float]
    shared_expert_intermediate_size: Optional[int]  # None: no shared expert
    shared_expert_gated: bool  # sigmoid gate on the shared expert (Qwen style)

    @classmethod
    def from_config(cls, draft_hf_config: Any) -> DraftMoeFFNSpec:
        text = text_config(draft_hf_config)
        dflash_cfg = nested_dict(draft_hf_config, "dflash_config")
        num_experts = _expert_count(text)
        if num_experts is None:
            raise ValueError(
                f"Draft FFN: MoE layers need one of {_EXPERT_COUNT_KEYS} to be set."
            )
        top_k = required_positive_int(
            cfg_get(text, "num_experts_per_tok", None),
            "num_experts_per_tok",
            "for MoE layers",
        )
        if top_k > num_experts:
            raise ValueError(
                f"Draft FFN: num_experts_per_tok={top_k} exceeds num_experts="
                f"{num_experts}."
            )
        # Mixtral-style configs size the experts with intermediate_size alone.
        intermediate_size = required_positive_int(
            cfg_get(
                text, "moe_intermediate_size", cfg_get(text, "intermediate_size", None)
            ),
            "moe_intermediate_size",
            "for MoE layers",
        )
        scoring_func = str(cfg_get(text, "scoring_func", "softmax"))
        if scoring_func not in _SCORING_FUNCS:
            raise ValueError(
                f"Draft FFN: scoring_func={scoring_func!r} is not implemented by "
                f"sglang's TopK router; available are {_SCORING_FUNCS}."
            )
        hidden_act = str(cfg_get(text, "hidden_act", "silu"))
        if hidden_act not in _HIDDEN_ACTS:
            raise ValueError(
                f"Draft FFN: hidden_act={hidden_act!r} is not implemented by "
                f"sglang's MoE experts; available are {_HIDDEN_ACTS}."
            )
        topk_method = cfg_get(text, "topk_method", None)
        if topk_method not in _TOPK_METHODS:
            raise ValueError(
                f"Draft FFN: topk_method={topk_method!r} is not implemented; "
                f"available are {_TOPK_METHODS}."
            )
        num_expert_group, topk_group = cls._groups(text, num_experts, top_k)
        bias_flag = dflash_cfg.get(
            "moe_router_enable_expert_bias",
            cfg_get(text, "moe_router_enable_expert_bias", False),
        )
        shared_size, shared_gated = cls._shared_expert(text, intermediate_size)
        return cls(
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
            scoring_func=scoring_func,
            norm_topk_prob=required_bool(
                cfg_get(text, "norm_topk_prob", True), "norm_topk_prob"
            ),
            expert_bias=required_bool(bias_flag, "moe_router_enable_expert_bias")
            or topk_method == "noaux_tc",
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            routed_scaling_factor=_optional_positive_float(
                cfg_get(text, "routed_scaling_factor", None), "routed_scaling_factor"
            )
            or 1.0,
            hidden_act=hidden_act,
            swiglu_limit=_optional_positive_float(
                cfg_get(text, "swiglu_limit", None), "swiglu_limit"
            ),
            swiglu_alpha=_optional_positive_float(
                cfg_get(text, "swiglu_alpha", None), "swiglu_alpha"
            ),
            shared_expert_intermediate_size=shared_size,
            shared_expert_gated=shared_gated,
        )

    @staticmethod
    def _groups(
        text: Any, num_experts: int, top_k: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """DeepSeek's group-limited routing: n_group / num_expert_group and topk_group."""
        groups = cfg_get(text, "n_group", cfg_get(text, "num_expert_group", None))
        groups = optional_positive_int(groups, "n_group")
        if groups is None or groups == 1:
            return None, None
        if num_experts % groups:
            raise ValueError(
                f"Draft FFN: n_group={groups} does not divide num_experts={num_experts}."
            )
        topk_group = required_positive_int(
            cfg_get(text, "topk_group", None), "topk_group", "when n_group > 1"
        )
        if topk_group > groups:
            raise ValueError(
                f"Draft FFN: topk_group={topk_group} exceeds n_group={groups}."
            )
        return groups, topk_group

    @staticmethod
    def _shared_expert(text: Any, intermediate_size: int) -> Tuple[Optional[int], bool]:
        """Qwen's gated shared expert, or DeepSeek's summed ones, never both."""
        gated_size = cfg_get(text, "shared_expert_intermediate_size", None) or 0
        summed_count = cfg_get(text, "n_shared_experts", None) or 0
        for name, value in (
            ("shared_expert_intermediate_size", gated_size),
            ("n_shared_experts", summed_count),
        ):
            if not is_int(value) or value < 0:
                raise ValueError(
                    f"Draft FFN: {name} must be a non-negative integer, got {value!r}."
                )
        if gated_size and summed_count:
            raise ValueError(
                "Draft FFN: shared_expert_intermediate_size and n_shared_experts "
                "declare two shared-expert layouts; set only one."
            )
        if gated_size:
            return int(gated_size), True
        if summed_count:
            return int(summed_count) * intermediate_size, False
        return None, False


class DraftMixFFNSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One FFN kind per layer; dense / moe only when a layer uses them and the config sizes them."""

    layer_kinds: Optional[Tuple[str, ...]]  # None: every layer is the same kind
    dense: Optional[DraftDenseFFNSpec]
    moe: Optional[DraftMoeFFNSpec]

    @classmethod
    def from_config(cls, draft_hf_config: Any) -> DraftMixFFNSpec:
        """Dense unless an expert count is set; ffn_layer_types may mark layers either way.

        The dense sub-spec is optional: hosts such as DFlash build dense layers themselves.
        """
        text = text_config(draft_hf_config)
        num_layers = optional_positive_int(
            cfg_get(text, "num_hidden_layers", None), "num_hidden_layers"
        )
        has_experts = _expert_count(text) is not None
        default_kind = FFN_KIND_MOE if has_experts else FFN_KIND_DENSE
        layer_kinds = cls._layer_kinds(text, num_layers, default_kind)
        kinds_in_use = set(layer_kinds) if layer_kinds is not None else {default_kind}
        dense = moe = None
        if FFN_KIND_DENSE in kinds_in_use and cfg_get(text, "intermediate_size", None):
            dense = DraftDenseFFNSpec.from_config(draft_hf_config)
        if FFN_KIND_MOE in kinds_in_use:
            moe = DraftMoeFFNSpec.from_config(draft_hf_config)
        return cls(layer_kinds=layer_kinds, dense=dense, moe=moe)

    @staticmethod
    def _layer_kinds(
        text: Any, num_layers: Optional[int], default_kind: str
    ) -> Optional[Tuple[str, ...]]:
        """Per-layer kinds from whichever HF schedule key the config uses."""
        kinds = DraftMixFFNSpec._schedule(text, num_layers, default_kind)
        if kinds is None:
            return None if num_layers is None else (default_kind,) * num_layers
        if num_layers is not None and len(kinds) != num_layers:
            raise ValueError(
                f"Draft FFN: the FFN schedule has {len(kinds)} entries but "
                f"num_hidden_layers={num_layers}."
            )
        if FFN_KIND_MOE in kinds and default_kind == FFN_KIND_DENSE:
            raise ValueError(
                "Draft FFN: the FFN schedule names MoE layers but none of "
                f"{_EXPERT_COUNT_KEYS} is set."
            )
        return kinds

    @staticmethod
    def _schedule(
        text: Any, num_layers: Optional[int], default_kind: str
    ) -> Optional[Tuple[str, ...]]:
        """ffn_layer_types, mlp_layer_types, moe_layer_freq, first_k_dense_replace, mlp_only_layers."""
        if (raw := cfg_get(text, "ffn_layer_types", None)) is not None:
            return _kind_list(
                raw, "ffn_layer_types", {"dense": FFN_KIND_DENSE, "moe": FFN_KIND_MOE}
            )
        if (raw := cfg_get(text, "mlp_layer_types", None)) is not None:
            return _kind_list(
                raw,
                "mlp_layer_types",
                {"dense": FFN_KIND_DENSE, "sparse": FFN_KIND_MOE},
            )
        freq = cfg_get(text, "moe_layer_freq", None)
        if isinstance(freq, (list, tuple)):  # MiniMax: 0 marks a dense layer
            return tuple(FFN_KIND_MOE if f else FFN_KIND_DENSE for f in freq)
        first_dense = cfg_get(text, "first_k_dense_replace", None)
        only_dense = cfg_get(text, "mlp_only_layers", None)
        sparse_step = cfg_get(text, "decoder_sparse_step", None)
        if first_dense is None and only_dense is None and sparse_step is None:
            return None
        if num_layers is None:
            raise ValueError("Draft FFN: the FFN schedule needs num_hidden_layers.")
        first_dense = int(first_dense or 0)  # DeepSeek: the first k layers are dense
        freq = int(freq or 1)
        only_dense = set(int(i) for i in (only_dense or ()))  # Qwen3-MoE
        sparse_step = int(sparse_step or 1)
        return tuple(
            FFN_KIND_MOE
            if i >= first_dense
            and i % freq == 0
            and i not in only_dense
            and (i + 1) % sparse_step == 0
            else FFN_KIND_DENSE
            for i in range(num_layers)
        )

    def kind(self, layer_id: int) -> str:
        if self.layer_kinds is None:
            return FFN_KIND_MOE if self.moe is not None else FFN_KIND_DENSE
        if not 0 <= layer_id < len(self.layer_kinds):
            raise ValueError(
                f"Draft FFN: layer {layer_id} is outside the "
                f"{len(self.layer_kinds)} layers ffn_layer_types describes."
            )
        return self.layer_kinds[layer_id]

    @property
    def has_moe(self) -> bool:
        return self.moe is not None
