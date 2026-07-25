"""Transfer metadata for owner-sharded cache families."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OwnerShardedTransferBuffer:
    """A CP owner-sharded source buffer and its decode-side item geometry."""

    tensor: torch.Tensor
    item_bytes: int
    owner_page_bytes: int
    owner_pages_per_item: int = 1
    rank_stride_owner_pages: int | None = None

    def __post_init__(self) -> None:
        if self.item_bytes <= 0 or self.owner_page_bytes <= 0:
            raise ValueError("shared PD transfer sizes must be positive")
        if self.owner_pages_per_item <= 0:
            raise ValueError("shared PD owner-pages-per-item must be positive")
        if (
            self.rank_stride_owner_pages is not None
            and self.rank_stride_owner_pages <= 0
        ):
            raise ValueError("shared PD rank stride must be positive")
        if self.item_bytes != self.owner_page_bytes * self.owner_pages_per_item:
            raise ValueError(
                "shared PD item geometry does not match its owner-page geometry"
            )
        if not self.tensor.is_contiguous():
            raise ValueError("shared PD source tensor must be contiguous")
        if self.tensor.numel() * self.tensor.element_size() % self.owner_page_bytes:
            raise ValueError(
                "shared PD source tensor cannot be represented as owner-page rows"
            )

    def owner_page_rows(self) -> torch.Tensor:
        return self.tensor.view(torch.uint8).reshape(-1, self.owner_page_bytes)
