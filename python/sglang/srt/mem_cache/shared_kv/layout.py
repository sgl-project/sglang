from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class OwnerShardedLayout:
    """Map logical rows into rank-major owner-sharded storage."""

    cp_size: int
    ownership_granule: int
    logical_rows: int
    physical_blocks_per_rank: Optional[int] = None

    def __post_init__(self) -> None:
        if self.cp_size <= 0:
            raise ValueError(f"cp_size must be positive, got {self.cp_size}")
        if self.ownership_granule <= 0:
            raise ValueError(
                "ownership_granule must be positive, " f"got {self.ownership_granule}"
            )
        if self.logical_rows < 0:
            raise ValueError(
                f"logical_rows must be non-negative, got {self.logical_rows}"
            )
        if (
            self.physical_blocks_per_rank is not None
            and self.physical_blocks_per_rank < self.minimum_blocks_per_rank
        ):
            raise ValueError(
                "physical_blocks_per_rank must be at least "
                f"{self.minimum_blocks_per_rank}, got "
                f"{self.physical_blocks_per_rank}"
            )

    @property
    def logical_blocks(self) -> int:
        return (
            self.logical_rows + self.ownership_granule - 1
        ) // self.ownership_granule

    @property
    def minimum_blocks_per_rank(self) -> int:
        return (self.logical_blocks + self.cp_size - 1) // self.cp_size

    @property
    def blocks_per_rank(self) -> int:
        if self.physical_blocks_per_rank is not None:
            return self.physical_blocks_per_rank
        return self.minimum_blocks_per_rank

    @property
    def physical_rows_per_rank(self) -> int:
        return self.blocks_per_rank * self.ownership_granule

    def _validate_rank(self, rank: int) -> None:
        if rank < 0 or rank >= self.cp_size:
            raise ValueError(f"rank must be in [0, {self.cp_size}), got {rank}")

    @staticmethod
    def _preserve_sentinels(
        logical_rows: torch.Tensor, translated_rows: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(logical_rows >= 0, translated_rows, logical_rows)

    def _logical_parts(
        self, logical_rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_rows = torch.where(logical_rows >= 0, logical_rows, 0)
        logical_blocks = torch.div(
            safe_rows, self.ownership_granule, rounding_mode="floor"
        )
        block_offsets = safe_rows % self.ownership_granule
        owners = logical_blocks % self.cp_size
        owner_blocks = torch.div(logical_blocks, self.cp_size, rounding_mode="floor")
        return logical_blocks, block_offsets, owners, owner_blocks

    def owner_rank(self, logical_rows: torch.Tensor) -> torch.Tensor:
        _, _, owners, _ = self._logical_parts(logical_rows)
        return self._preserve_sentinels(logical_rows, owners)

    def owner_local_rows(self, logical_rows: torch.Tensor) -> torch.Tensor:
        _, block_offsets, _, owner_blocks = self._logical_parts(logical_rows)
        translated = owner_blocks * self.ownership_granule + block_offsets
        return self._preserve_sentinels(logical_rows, translated)

    def physical_rows(self, logical_rows: torch.Tensor) -> torch.Tensor:
        _, block_offsets, owners, owner_blocks = self._logical_parts(logical_rows)
        physical_blocks = owners * self.blocks_per_rank + owner_blocks
        translated = physical_blocks * self.ownership_granule + block_offsets
        return self._preserve_sentinels(logical_rows, translated)

    def rank_relative_rows(
        self, logical_rows: torch.Tensor, *, rank: int
    ) -> torch.Tensor:
        self._validate_rank(rank)
        _, block_offsets, owners, owner_blocks = self._logical_parts(logical_rows)
        relative_owners = (owners - rank) % self.cp_size
        relative_blocks = relative_owners * self.blocks_per_rank + owner_blocks
        translated = relative_blocks * self.ownership_granule + block_offsets
        return self._preserve_sentinels(logical_rows, translated)

    def owned_row_mask(self, logical_rows: torch.Tensor, *, rank: int) -> torch.Tensor:
        self._validate_rank(rank)
        _, _, owners, _ = self._logical_parts(logical_rows)
        return (logical_rows >= 0) & (owners == rank)
