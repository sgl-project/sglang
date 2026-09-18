"""Request-local attention state carried between DeepSeek V4 PP stages."""

import torch


def remap_sparse_slots(
    slots: torch.Tensor,
    source_pages: torch.Tensor,
    target_pages: torch.Tensor,
    slots_per_page: int,
    num_pages: torch.Tensor,
) -> torch.Tensor:
    """Translate physical slots through each query's logical page table.

    Radix eviction and HiCache reload may allocate different physical pages on
    each stage. Even shared prefixes can have different sharing on each stage,
    so a single global physical-to-physical map is insufficient.
    """
    if slots.numel() == 0:
        return slots.clone()
    logical_ids = torch.arange(source_pages.shape[-1], device=source_pages.device)
    active_pages = logical_ids < num_pages.unsqueeze(-1)
    searchable_pages = source_pages.masked_fill(
        ~active_pages, torch.iinfo(source_pages.dtype).max
    )
    sorted_pages, logical_pages = searchable_pages.sort(dim=-1)
    physical_pages = slots.clamp_min(0) // slots_per_page
    matches = torch.searchsorted(
        sorted_pages.contiguous(), physical_pages.contiguous()
    ).clamp_max(source_pages.shape[-1] - 1)
    logical = logical_pages.gather(-1, matches)
    remapped = (
        target_pages.gather(-1, logical) * slots_per_page + slots % slots_per_page
    )
    valid = (slots >= 0) & (sorted_pages.gather(-1, matches) == physical_pages)
    return torch.where(valid, remapped, -1).to(slots.dtype)
