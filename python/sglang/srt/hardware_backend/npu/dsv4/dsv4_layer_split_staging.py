"""Active-page staging and remap for the DSV4 NPU layer-split pool.

Remote-layer reads compact only the pages the current batch references (the
union across the attention-CP group) into the scratch buffer instead of whole
layers, and remap indices/page tables onto the compacted copy. Pure torch;
the collective is an int32 all-reduce over the CP group's device group.
"""

from __future__ import annotations

from typing import Optional

import torch


def build_active_pages_mask(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
) -> torch.Tensor:
    """Per-page hit counts for token ``indices`` (-1 entries are padding)."""
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


def all_reduce_active_pages_mask(local_mask: torch.Tensor, group) -> torch.Tensor:
    """Sum the per-rank active-page mask across the attention-CP group."""
    torch.distributed.all_reduce(local_mask, group=group.device_group)
    return local_mask


def selected_active_pages(local_mask: torch.Tensor) -> torch.Tensor:
    """Sorted page ids with a nonzero hit count."""
    return torch.nonzero(local_mask, as_tuple=False).flatten()


def remap_indices_to_staging(
    indices: torch.Tensor,
    selected_pages: torch.Tensor,
    page_size: int,
    max_pages: int,
) -> torch.Tensor:
    """Token indices rebased onto the compacted staging copy (padding kept)."""
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


def remap_page_table_to_staging(
    page_table: torch.Tensor,
    selected_pages: torch.Tensor,
    max_pages: int,
) -> torch.Tensor:
    """Page table entries rebased onto the compacted staging copy."""
    page_map = torch.full(
        (max_pages,), -1, dtype=torch.int32, device=page_table.device
    )
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
    group,
) -> torch.Tensor:
    """Pages touched by any CP rank; all ranks must call in the same order."""
    local_mask = build_active_pages_mask(indices, page_size, max_pages)
    local_mask = all_reduce_active_pages_mask(local_mask, group)
    return selected_active_pages(local_mask)
