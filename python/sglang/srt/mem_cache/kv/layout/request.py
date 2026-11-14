"""The request layout: payload a running request owns beside its state slot and
the prefix tree never caches, one entry per request row."""

from dataclasses import dataclass
from typing import ClassVar, Dict, Tuple

import torch

from sglang.srt.mem_cache.kv.decl import TensorPartDecl
from sglang.srt.mem_cache.kv.layout.base import SubPoolSpec
from sglang.srt.mem_cache.kv.layout.parts import TensorEntryLayout
from sglang.srt.mem_cache.layout.token_major import align_entry_bytes


@dataclass(frozen=True, kw_only=True)
class RequestSubPoolSpec(SubPoolSpec):
    """Per-row layout of the request sub-pool: every part over its layers."""

    _allowed_grow_directions: ClassVar[Tuple[str, ...]] = ("float",)
    # A row's new owner starts from the parts' reset values, all zero.
    zero_units_on_handout: ClassVar[bool] = True

    parts: Tuple[TensorPartDecl, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        nonzero = [p.name for p in self.parts if p.reset_value != 0]
        if nonzero:
            raise ValueError(
                f"request parts reset by zeroing their unit; {nonzero} reset to "
                "another value"
            )

    def entry_layout(self) -> TensorEntryLayout:
        return TensorEntryLayout(self.parts)

    def entry_bytes(self) -> int:
        # Whole-unit copies and zeroing move int64 words.
        return align_entry_bytes(self.entry_layout().entry_bytes)

    def get_dtype(self) -> torch.dtype:
        return torch.uint8

    def part_views(self, arena) -> Dict[str, torch.Tensor]:
        """Each part as ``[layer, unit, *shape]`` (``[unit, *shape]`` without
        layers) over the band's whole extent; a unit's stride is the entry."""
        return self.entry_layout().views(
            arena.raw, arena.extent_entries(self.name), entry_bytes=self.entry_bytes()
        )
