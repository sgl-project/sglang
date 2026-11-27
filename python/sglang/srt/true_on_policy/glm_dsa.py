"""Canonical logical DSA slots, before cache-coordinate mapping."""

import torch


def canonical_slot_order(indices):
    """Sort valid logical indices ascending and retain -1 padding at the end."""
    pad = torch.iinfo(torch.int32).max
    keys = torch.where(
        indices >= 0,
        indices.to(torch.int32),
        torch.full_like(indices, pad, dtype=torch.int32),
    )
    keys = torch.sort(keys, dim=-1).values
    return torch.where(keys == pad, torch.full_like(keys, -1), keys)
