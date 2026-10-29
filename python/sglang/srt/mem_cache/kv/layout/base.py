"""The spec every unified sub-pool layout implements, and the row checks layouts share."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.layout.fused_draft import DraftRegion
from sglang.srt.mem_cache.layout.token_major import ROW_ALIGN_BYTES


def _check_row_alignment(sub_pool: str, **row_bytes: int) -> None:
    """Entry parts are laid out in ROW_ALIGN_BYTES units; a model whose
    per-rank rows are not cannot use the unified pool."""
    for part, nbytes in row_bytes.items():
        if nbytes % ROW_ALIGN_BYTES:
            raise ValueError(
                f"--enable-unified-memory needs every per-rank KV row to be a "
                f"multiple of {ROW_ALIGN_BYTES} bytes; sub-pool {sub_pool!r} has a "
                f"{nbytes}-byte {part} row. Run this model without "
                "--enable-unified-memory."
            )


def check_physical_id_bound(*, sub_pool: str, n_rows: int) -> None:
    """Physical ids must fit the int32 read-index buffers."""
    assert n_rows < 2**31, (
        f"sub-pool {sub_pool!r}: {n_rows} physical ids exceed the int32 "
        f"bound (2^31) that read-index buffers assume. "
        "Reduce max_total_num_tokens."
    )


def dense_rows(arena, name: str) -> int:
    """Token rows a dense view of band ``name`` spans: whole units only."""
    unit = arena.band(name).unit_entries
    return arena.extent_entries(name) // unit * unit


def _store_dtype_for(kv_cache_dtype: torch.dtype) -> torch.dtype:
    if kv_cache_dtype in (
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        return torch.uint8
    return kv_cache_dtype


@dataclass(frozen=True, kw_only=True)
class SubPoolSpec(ABC):
    """Abstract per-slot layout of one sub-pool's band in a `KVArena`.

    ``grow_direction`` places the sub-pool in the buffer's chain: the two
    ``"up"``/``"down"`` END pools own the buffer's two ends and grow inward;
    ``"float"`` middles live between the ends' frontiers (a float's position is
    carried entirely by its allocator's two watermarks — every view spans the
    whole buffer, so relocation never rebuilds views).
    """

    # Grow directions this subclass accepts. Scratch-class specs (e.g. the
    # spec-decode band) narrow this to ("float",).
    _allowed_grow_directions: ClassVar[Tuple[str, ...]] = ("up", "down", "float")
    # Whether units of this layout are zeroed when handed out: readers that load
    # whole pages give rows past the sequence a zero weight, so a NaN left there
    # by another band's bytes would still reach the output.
    zero_units_on_handout: ClassVar[bool] = False

    name: str
    layer_num: int
    grow_direction: str  # "up" | "down" | "float"
    # Fused draft region riding in this sub-pool's entries; None = unfused.
    # Each kind accepts the region kind its entry can lay out, after the host
    # block at `draft_offset_in_entry()`.
    draft_region: Optional[DraftRegion] = None

    def __post_init__(self):
        assert self.grow_direction in self._allowed_grow_directions, (
            f"{type(self).__name__}.grow_direction must be one of "
            f"{self._allowed_grow_directions}; got {self.grow_direction!r}"
        )
        assert self.layer_num > 0 or (
            self.layer_num == 0 and self.draft_region is not None
        ), "an empty target band must carry declared draft parts"

    @abstractmethod
    def entry_bytes(self) -> int:
        """Bytes for one slot across all `layer_num` layers."""
        raise NotImplementedError

    @abstractmethod
    def get_dtype(self) -> torch.dtype:
        """Storage dtype (informational). Multi-dtype subclasses return the dominant buffer's."""
        raise NotImplementedError


class HostPoolOptions(msgspec.Struct, frozen=True, kw_only=True):
    """Host allocation policy, separate from a layout's device buffer facts."""

    ratio: float
    size: float
    page_size: int
    layout: str
    allocator_type: str
    pool_label: str
    dcp_size: int = 1
    dcp_rank: int = 0
    pin_memory: bool = True
