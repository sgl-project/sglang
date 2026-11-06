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
    num_experts: int
    top_k: int
    intermediate_size: int
    norm_topk_prob: bool  # renormalize the chosen experts' scores to sum to one
    scoring_func: str
    hidden_act: str
    shared_expert_intermediate_size: Optional[int]  # None: no shared expert
    expert_bias: bool  # the checkpoint carries a trained expert-selection bias

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
        intermediate_size = required_positive_int(
            cfg_get(text, "moe_intermediate_size", None),
            "moe_intermediate_size",
            "for MoE layers (intermediate_size is not a substitute)",
        )
        shared_size = cfg_get(text, "shared_expert_intermediate_size", 0)
        shared_size = 0 if shared_size is None else shared_size
        if not is_int(shared_size) or shared_size < 0:
            raise ValueError(
                "Draft FFN: shared_expert_intermediate_size must be a non-negative "
                f"integer, got {shared_size!r}."
            )
        expert_bias = dflash_cfg.get(
            "moe_router_enable_expert_bias",
            cfg_get(text, "moe_router_enable_expert_bias", False),
        )
        return cls(
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
            norm_topk_prob=required_bool(
                cfg_get(text, "norm_topk_prob", True), "norm_topk_prob"
            ),
            scoring_func=str(cfg_get(text, "scoring_func", "softmax")),
            hidden_act=str(cfg_get(text, "hidden_act", "silu")),
            shared_expert_intermediate_size=int(shared_size) or None,
            expert_bias=required_bool(expert_bias, "moe_router_enable_expert_bias"),
        )


class DraftMixFFNSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One FFN kind per layer; dense / moe exist only when some layer uses them."""

    layer_kinds: Optional[Tuple[str, ...]]  # None: every layer is the same kind
    dense: Optional[DraftDenseFFNSpec]
    moe: Optional[DraftMoeFFNSpec]

    @classmethod
    def from_config(cls, draft_hf_config: Any) -> DraftMixFFNSpec:
        """Dense unless an expert count is set; ffn_layer_types may mark layers either way."""
        text = text_config(draft_hf_config)
        num_layers = optional_positive_int(
            cfg_get(text, "num_hidden_layers", None), "num_hidden_layers"
        )
        has_experts = _expert_count(text) is not None
        default_kind = FFN_KIND_MOE if has_experts else FFN_KIND_DENSE
        layer_kinds = cls._layer_kinds(text, num_layers, default_kind)
        kinds_in_use = set(layer_kinds) if layer_kinds is not None else {default_kind}
        dense = moe = None
        if FFN_KIND_DENSE in kinds_in_use:
            dense = DraftDenseFFNSpec.from_config(draft_hf_config)
        if FFN_KIND_MOE in kinds_in_use:
            moe = DraftMoeFFNSpec.from_config(draft_hf_config)
        return cls(layer_kinds=layer_kinds, dense=dense, moe=moe)

    @staticmethod
    def _layer_kinds(
        text: Any, num_layers: Optional[int], default_kind: str
    ) -> Optional[Tuple[str, ...]]:
        raw = cfg_get(text, "ffn_layer_types", None)
        if raw is None:
            return None if num_layers is None else (default_kind,) * num_layers
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            raise ValueError(
                f"Draft FFN: ffn_layer_types must be a non-empty list of "
                f"{FFN_KINDS}, got {raw!r}."
            )
        kinds = tuple(raw)
        for index, kind in enumerate(kinds):
            if kind not in FFN_KINDS:
                raise ValueError(
                    f"Draft FFN: ffn_layer_types[{index}]={kind!r} is not one of "
                    f"{FFN_KINDS}."
                )
        if num_layers is not None and len(kinds) != num_layers:
            raise ValueError(
                f"Draft FFN: ffn_layer_types has {len(kinds)} entries but "
                f"num_hidden_layers={num_layers}."
            )
        if FFN_KIND_MOE in kinds and default_kind == FFN_KIND_DENSE:
            raise ValueError(
                "Draft FFN: ffn_layer_types names MoE layers but none of "
                f"{_EXPERT_COUNT_KEYS} is set."
            )
        return kinds

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
