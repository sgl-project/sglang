"""The MHA sub-pool layout: per-layer K and V rows in one token-major entry."""

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.kv.layout.base import (
    SubPoolSpec,
    _check_row_alignment,
    _store_dtype_for,
    check_physical_id_bound,
    dense_rows,
)
from sglang.srt.mem_cache.layout.fused_draft import DenseDraftRegion
from sglang.srt.mem_cache.layout.token_major import (
    DenseEntryLayout,
    DensePart,
    align_entry_bytes,
    align_part_offset,
    build_dense_views,
)


@dataclass(frozen=True, kw_only=True)
class MHASubPoolSpec(SubPoolSpec):
    """Per-slot layout of one MHA-shaped sub-pool. `v_head_dim` defaults to `head_dim`.

    With `draft_region` set, the draft model's K and V rows are two more
    parts of every slot's entry, after the host parts:

        [ K_0 | V_0 | ... | K_{Lh-1} | V_{Lh-1} | dK_0 | dV_0 | ... | pad ]

    The slot stride stays the one entry, so host and draft views are indexed
    by the same physical token id; only their offsets differ.
    """

    zero_units_on_handout: ClassVar[bool] = True

    head_num: int
    head_dim: int
    store_dtype: torch.dtype
    kv_cache_dtype: Optional[torch.dtype] = None
    v_head_dim: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.head_num > 0, f"head_num must be positive; got {self.head_num}"
        assert self.head_dim > 0, f"head_dim must be positive; got {self.head_dim}"
        if self.kv_cache_dtype is None:
            object.__setattr__(self, "kv_cache_dtype", self.store_dtype)
        expected_store_dtype = _store_dtype_for(self.kv_cache_dtype)
        assert self.store_dtype == expected_store_dtype, (
            "MHASubPoolSpec.store_dtype must match the storage dtype required by "
            f"kv_cache_dtype; got kv_cache_dtype={self.kv_cache_dtype}, "
            f"store_dtype={self.store_dtype}, expected={expected_store_dtype}"
        )
        if self.v_head_dim is None:
            object.__setattr__(self, "v_head_dim", self.head_dim)
        assert self.v_head_dim > 0, (
            f"v_head_dim must be positive; got {self.v_head_dim}"
        )
        if self.draft_region is not None:
            assert isinstance(self.draft_region, DenseDraftRegion), (
                "dense entries carry dense draft K/V rows, never a state region"
            )
            self.draft_region.validate()

    def k_row_bytes(self) -> int:
        return self.head_num * self.head_dim * self.store_dtype.itemsize

    def v_row_bytes(self) -> int:
        return self.head_num * self.v_head_dim * self.store_dtype.itemsize

    def host_entry_bytes(self) -> int:
        """Host (target-only) bytes for one slot, before any draft parts."""
        return self.layer_num * (self.k_row_bytes() + self.v_row_bytes())

    def draft_offset_in_entry(self) -> int:
        """Byte offset of the fused draft parts inside one slot's entry."""
        assert self.draft_region is not None
        return align_part_offset(self.host_entry_bytes())

    def entry_bytes(self) -> int:
        if self.draft_region is None:
            return align_entry_bytes(self.host_entry_bytes())
        return align_entry_bytes(
            self.draft_offset_in_entry() + self.draft_region.entry_bytes()
        )

    def page_bytes(self, page_size: int) -> int:
        return page_size * self.entry_bytes()

    def layout(self) -> DenseEntryLayout:
        """Token-major entry ``[K_0 | V_0 | K_1 | V_1 | ...]`` per slot; a page
        is ``page_size`` such entries back to back."""
        _check_row_alignment(self.name, K=self.k_row_bytes(), V=self.v_row_bytes())
        layer_stride = self.k_row_bytes() + self.v_row_bytes()
        parts = (
            DensePart(
                name="k",
                offset_bytes=0,
                layer_stride_bytes=layer_stride,
                layer_num=self.layer_num,
                row_shape=(self.head_num, self.head_dim),
                dtype=self.store_dtype,
            ),
            DensePart(
                name="v",
                offset_bytes=self.k_row_bytes(),
                layer_stride_bytes=layer_stride,
                layer_num=self.layer_num,
                row_shape=(self.head_num, self.v_head_dim),
                dtype=self.store_dtype,
            ),
        )
        if self.draft_region is not None:
            parts += self.draft_region.parts(self.draft_offset_in_entry())
        return DenseEntryLayout(entry_bytes=self.entry_bytes(), parts=parts)

    def views(self, arena) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-layer K and V views over this sub-pool's band of ``arena``,
        indexed by the physical token id."""
        rows = dense_rows(arena, self.name)
        check_physical_id_bound(sub_pool=self.name, n_rows=rows)
        return self._part_views(arena, rows, "k"), self._part_views(arena, rows, "v")

    def draft_views(self, arena) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-layer K and V views of the draft parts fused into this
        sub-pool's entries: the same rows, the same ids."""
        assert self.draft_region is not None, (
            f"sub-pool {self.name!r} carries no fused draft region"
        )
        rows = dense_rows(arena, self.name)
        return (
            self._part_views(arena, rows, "draft_k"),
            self._part_views(arena, rows, "draft_v"),
        )

    def _part_views(self, arena, rows: int, part: str) -> List[torch.Tensor]:
        return build_dense_views(
            arena.raw, layout=self.layout(), part_name=part, num_slots=rows
        )

    def get_dtype(self) -> torch.dtype:
        return self.store_dtype
