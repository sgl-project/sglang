from __future__ import annotations

from enum import Enum

import msgspec
import torch


class RouteViewKind(str, Enum):
    RAW = "raw"
    ALIGNED = "aligned"


class RouteView(msgspec.Struct, frozen=True, kw_only=True):
    view: RouteViewKind
    block_size: int
    topk_ids: torch.Tensor
    token_lora_mapping: torch.Tensor
    num_local_experts: int
    is_shared_outer: bool
    max_loras: int
    maybe_sorted_pair_ids: torch.Tensor | None = None
    maybe_block_virtual_expert_ids: torch.Tensor | None = None
    maybe_num_pairs_post_padded: torch.Tensor | None = None

    @property
    def lora_experts_per_adapter(self) -> int:
        return 1 if self.is_shared_outer else self.num_local_experts

    @property
    def num_virtual_experts(self) -> int:
        return self.lora_experts_per_adapter * self.max_loras

    def _require(self, value, field: str, needed: RouteViewKind):
        if value is None:
            raise ValueError(
                f"route view {self.view.value!r} did not build {field}; the "
                f"consumer must request view {needed.value!r} or derive it inline"
            )
        return value

    @property
    def sorted_pair_ids(self) -> torch.Tensor:
        return self._require(
            self.maybe_sorted_pair_ids, "sorted_pair_ids", RouteViewKind.ALIGNED
        )

    @property
    def block_virtual_expert_ids(self) -> torch.Tensor:
        return self._require(
            self.maybe_block_virtual_expert_ids,
            "block_virtual_expert_ids",
            RouteViewKind.ALIGNED,
        )

    @property
    def num_pairs_post_padded(self) -> torch.Tensor:
        return self._require(
            self.maybe_num_pairs_post_padded,
            "num_pairs_post_padded",
            RouteViewKind.ALIGNED,
        )
