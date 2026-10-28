"""The MLA sub-pool layout: one latent row per layer in a token-major entry."""

from dataclasses import dataclass
from typing import ClassVar, List, Tuple

import torch

from sglang.srt.mem_cache.kv.layout.base import (
    SubPoolSpec,
    _check_row_alignment,
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
class MLASubPoolSpec(SubPoolSpec):
    """Per-slot layout of one MLA-shaped sub-pool.

    One latent row (``kv_lora_rank + qk_rope_head_dim``) per token per layer; V
    is a prefix slice of the same row, so there is no separate V region. Not a
    subclass of ``MHASubPoolSpec`` — the K+V byte math and the ``v_head_dim > 0``
    invariant there do not apply. With `draft_region` set, the draft's K and V
    parts follow the latent rows inside every slot's entry, as on an MHA host.
    """

    zero_units_on_handout: ClassVar[bool] = True

    kv_lora_rank: int
    qk_rope_head_dim: int
    store_dtype: torch.dtype

    def __post_init__(self):
        super().__post_init__()
        assert self.kv_lora_rank > 0, (
            f"kv_lora_rank must be positive; got {self.kv_lora_rank}"
        )
        assert self.qk_rope_head_dim > 0, (
            f"qk_rope_head_dim must be positive; got {self.qk_rope_head_dim}"
        )
        if self.draft_region is not None:
            assert isinstance(self.draft_region, DenseDraftRegion), (
                "dense entries carry dense draft K/V rows, never a state region"
            )
            self.draft_region.validate()

    @property
    def kv_cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def row_bytes(self) -> int:
        return self.kv_cache_dim * self.store_dtype.itemsize

    def host_entry_bytes(self) -> int:
        """Host (target-only) bytes for one slot, before any draft parts."""
        return self.layer_num * self.row_bytes()

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

    def layout(self) -> DenseEntryLayout:
        _check_row_alignment(self.name, latent=self.row_bytes())
        parts = (
            DensePart(
                name="kv",
                offset_bytes=0,
                layer_stride_bytes=self.row_bytes(),
                layer_num=self.layer_num,
                row_shape=(1, self.kv_cache_dim),
                dtype=self.store_dtype,
            ),
        )
        if self.draft_region is not None:
            parts += self.draft_region.parts(self.draft_offset_in_entry())
        return DenseEntryLayout(entry_bytes=self.entry_bytes(), parts=parts)

    def views(self, arena) -> List[torch.Tensor]:
        """Per-layer latent views over this sub-pool's band of ``arena``,
        indexed by the physical token id."""
        rows = dense_rows(arena, self.name)
        check_physical_id_bound(sub_pool=self.name, n_rows=rows)
        return build_dense_views(
            arena.raw, layout=self.layout(), part_name="kv", num_slots=rows
        )

    def draft_views(self, arena) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-layer K and V views of the draft parts fused into this
        sub-pool's entries: the same rows, the same ids."""
        assert self.draft_region is not None, (
            f"sub-pool {self.name!r} carries no fused draft region"
        )
        rows = dense_rows(arena, self.name)
        return tuple(
            build_dense_views(
                arena.raw, layout=self.layout(), part_name=part, num_slots=rows
            )
            for part in ("draft_k", "draft_v")
        )

    def get_dtype(self) -> torch.dtype:
        return self.store_dtype
