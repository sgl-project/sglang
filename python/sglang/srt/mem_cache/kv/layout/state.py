"""The state sub-pool layout: conv and temporal state rows of every layer in one slot entry."""

from dataclasses import dataclass
from typing import Tuple

import torch

from sglang.srt.mem_cache.kv.layout.base import SubPoolSpec
from sglang.srt.mem_cache.layout.fused_draft import DraftStateRegion
from sglang.srt.mem_cache.layout.token_major import (
    align_entry_bytes,
    align_part_offset,
)


def _prod(iterable) -> int:
    out = 1
    for x in iterable:
        out *= int(x)
    return out


@dataclass(frozen=True, kw_only=True)
class MambaSubPoolSpec(SubPoolSpec):
    """Per-slot layout of one Mamba-shaped sub-pool.

    With `draft_region` (a `DraftStateRegion`) set, the draft's state block
    follows the host's inside every slot's entry:

        [ conv[0] x L | ... | temporal x L | draft conv[0] x Ld | ... | pad ]

    Unfused, the entry keeps its raw (unaligned) size, byte-identical to before.
    """

    conv_state_shapes: Tuple[Tuple[int, ...], ...]  # one shape per conv tensor
    conv_dtype: torch.dtype
    temporal_state_shape: Tuple[int, ...]
    temporal_dtype: torch.dtype
    conv_slice_axis: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert len(self.conv_state_shapes) > 0, "conv_state_shapes must be non-empty"
        if self.draft_region is not None:
            assert isinstance(self.draft_region, DraftStateRegion), (
                "mamba state entries carry a draft STATE region, never dense K/V rows"
            )
            self.draft_region.validate()

    def conv_row_bytes(self, idx: int) -> int:
        return _prod(self.conv_state_shapes[idx]) * self.conv_dtype.itemsize

    def temporal_row_bytes(self) -> int:
        return _prod(self.temporal_state_shape) * self.temporal_dtype.itemsize

    def host_entry_bytes(self) -> int:
        """Host (target-only) bytes for one slot, before any draft block."""
        total = 0
        for i in range(len(self.conv_state_shapes)):
            total += self.layer_num * self.conv_row_bytes(i)
        total += self.layer_num * self.temporal_row_bytes()
        return total

    def draft_offset_in_entry(self) -> int:
        """Byte offset of the fused draft state block inside one slot's entry."""
        assert self.draft_region is not None
        return align_part_offset(self.host_entry_bytes())

    def entry_bytes(self) -> int:
        if self.draft_region is None:
            return self.host_entry_bytes()
        return align_entry_bytes(
            self.draft_offset_in_entry() + self.draft_region.entry_bytes()
        )

    def get_dtype(self) -> torch.dtype:
        return self.conv_dtype  # representative state dtype; matches MambaPool.dtype
