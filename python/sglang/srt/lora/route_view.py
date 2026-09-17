from __future__ import annotations

from enum import Enum

import msgspec
import torch


class RouteViewKind(str, Enum):
    RAW = "raw"
    ALIGNED = "aligned"


class RouteView(msgspec.Struct, frozen=True, kw_only=True):
    """Rows grouped into blocks of one (adapter slot, group) bucket.

    ``token_slots`` [tokens] selects adapters. Optional ``group_ids``
    [tokens, width] selects heads or experts; otherwise each token is one row.
    Valid slots and groups map to ``slot * groups_per_slot + group``;
    groups_per_slot == 1 folds all live groups into the slot's bucket.
    Negative groups mark padding or nonlocal experts. ALIGNED views sort rows
    by bucket and pad to whole blocks.
    """

    view: RouteViewKind
    block_size: int
    token_slots: torch.Tensor
    group_ids: torch.Tensor | None
    groups_per_slot: int
    max_loras: int
    maybe_sorted_pair_ids: torch.Tensor | None = None
    maybe_block_bucket_ids: torch.Tensor | None = None
    maybe_num_pairs_post_padded: torch.Tensor | None = None

    @property
    def num_buckets(self) -> int:
        """Buckets before the sentinel."""
        return self.groups_per_slot * self.max_loras

    @property
    def num_tokens(self) -> int:
        return self.token_slots.numel()

    @property
    def num_rows(self) -> int:
        return (
            self.token_slots.numel()
            if self.group_ids is None
            else self.group_ids.numel()
        )

    @property
    def width(self) -> int:
        return 1 if self.group_ids is None else self.group_ids.shape[1]

    @property
    def kernel_groups(self) -> torch.Tensor:
        """Group IDs, or an unread placeholder when HAS_GROUPS=False."""
        return self.token_slots if self.group_ids is None else self.group_ids

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
    def block_bucket_ids(self) -> torch.Tensor:
        return self._require(
            self.maybe_block_bucket_ids,
            "block_bucket_ids",
            RouteViewKind.ALIGNED,
        )

    @property
    def num_pairs_post_padded(self) -> torch.Tensor:
        return self._require(
            self.maybe_num_pairs_post_padded,
            "num_pairs_post_padded",
            RouteViewKind.ALIGNED,
        )
