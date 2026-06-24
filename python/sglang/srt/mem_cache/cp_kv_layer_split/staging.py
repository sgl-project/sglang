"""Staging buffers and index remap kernels for CP KV LayerSplit."""

from __future__ import annotations

from typing import Callable, Optional

import torch


@torch.compile(dynamic=True)
def build_active_pages_mask(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
) -> torch.Tensor:
    local_mask = torch.zeros(max_pages, dtype=torch.int32, device=indices.device)
    valid = indices >= 0
    safe_indices = torch.clamp(indices, min=0)
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")
    local_mask.index_put_(
        (page_ids.flatten().to(torch.long),),
        valid.flatten().to(torch.int32),
        accumulate=True,
    )
    return local_mask


def all_reduce_active_pages_mask(local_mask: torch.Tensor, pynccl_comm) -> torch.Tensor:
    """Sum the per-rank active-page mask across the attention-CP group."""
    with pynccl_comm.change_state(enable=True):
        pynccl_comm.all_reduce(local_mask)
    return local_mask


@torch.compile(dynamic=True)
def remap_indices_to_staging(
    indices: torch.Tensor,
    selected_pages: torch.Tensor,
    page_size: int,
    max_pages: int,
) -> torch.Tensor:
    page_map = torch.full((max_pages,), -1, dtype=torch.int32, device=indices.device)
    page_map[selected_pages.to(torch.long)] = torch.arange(
        selected_pages.numel(), dtype=torch.int32, device=indices.device
    )

    valid = indices >= 0
    safe_indices = torch.clamp(indices, min=0)
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")
    offsets = safe_indices - page_ids * page_size
    new_pages = page_map[page_ids.to(torch.long)].to(indices.dtype)
    remapped = new_pages * page_size + offsets
    return torch.where(valid, remapped, indices)


@torch.compile(dynamic=True)
def remap_page_table_to_staging(
    page_table: torch.Tensor,
    selected_pages: torch.Tensor,
    max_pages: int,
) -> torch.Tensor:
    page_map = torch.full((max_pages,), -1, dtype=torch.int32, device=page_table.device)
    page_map[selected_pages.to(torch.long)] = torch.arange(
        selected_pages.numel(), dtype=torch.int32, device=page_table.device
    )

    valid = page_table >= 0
    safe_pages = torch.clamp(page_table, min=0)
    remapped = page_map[safe_pages.to(torch.long)].to(page_table.dtype)
    return torch.where(valid, remapped, page_table)


def active_pages_for_indices(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
    pynccl_comm,
) -> torch.Tensor:
    """Select pages touched by any CP rank; all ranks must call in the same order."""
    local_mask = build_active_pages_mask(indices, page_size, max_pages)
    local_mask = all_reduce_active_pages_mask(local_mask, pynccl_comm)
    # TODO: avoid dynamic-shape overhead
    return torch.nonzero(local_mask, as_tuple=False).flatten()


class StagingBufferManager:
    """Family-keyed grow-only staging buffers for compact reads."""

    def __init__(self) -> None:
        self._buffers: dict[str, Optional[torch.Tensor]] = {}

    def get_or_grow(
        self,
        family: str,
        num_pages: int,
        allocate_fn: Callable[[int], torch.Tensor],
    ) -> torch.Tensor:
        buffer = self._buffers.get(family)
        if buffer is None or buffer.shape[0] < num_pages:
            buffer = allocate_fn(num_pages)
            self._buffers[family] = buffer
        return buffer

    def get_existing(self, family: str) -> Optional[torch.Tensor]:
        return self._buffers.get(family)
