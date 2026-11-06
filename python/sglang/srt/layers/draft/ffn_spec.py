"""FFN spec of a draft: dense or MoE per layer, read from the draft's own config."""

from __future__ import annotations

from typing import Optional, Tuple

import msgspec

FFN_KIND_DENSE = "dense"
FFN_KIND_MOE = "moe"
FFN_KINDS = (FFN_KIND_DENSE, FFN_KIND_MOE)


class DraftDenseFFNSpec(msgspec.Struct, frozen=True, kw_only=True):
    intermediate_size: int
    hidden_act: str


class DraftMoeFFNSpec(msgspec.Struct, frozen=True, kw_only=True):
    num_experts: int
    top_k: int
    intermediate_size: int
    norm_topk_prob: bool  # renormalize the chosen experts' scores to sum to one
    scoring_func: str
    hidden_act: str
    shared_expert_intermediate_size: Optional[int]  # None: no shared expert
    expert_bias: bool  # the checkpoint carries a trained expert-selection bias


class DraftMixFFNSpec(msgspec.Struct, frozen=True, kw_only=True):
    """One FFN kind per layer; dense / moe exist only when some layer uses them."""

    layer_kinds: Optional[Tuple[str, ...]]  # None: every layer is default_kind
    default_kind: str
    dense: Optional[DraftDenseFFNSpec]
    moe: Optional[DraftMoeFFNSpec]

    def kind(self, layer_id: int) -> str:
        if self.layer_kinds is None:
            return self.default_kind
        if not 0 <= layer_id < len(self.layer_kinds):
            raise ValueError(
                f"Draft FFN: layer {layer_id} is outside the "
                f"{len(self.layer_kinds)} layers ffn_layer_types describes."
            )
        return self.layer_kinds[layer_id]

    @property
    def has_dense(self) -> bool:
        return self.dense is not None

    @property
    def has_moe(self) -> bool:
        return self.moe is not None
