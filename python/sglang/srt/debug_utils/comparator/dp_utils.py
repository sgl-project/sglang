"""DP filtering: keep only the non-empty dp_rank items."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch

from sglang.srt.debug_utils.dump_loader import ValueWithMeta

_PARALLEL_INFO_KEYS = ("sglang_parallel_info", "megatron_parallel_info")

_DP_RANK_FIELD = "dp_rank"
_DP_SIZE_FIELD = "dp_size"


def filter_to_non_empty_dp_rank(items: list[ValueWithMeta]) -> list[ValueWithMeta]:
    """Filter items to the single non-empty dp_rank.

    - dp_size <= 1: return items unchanged.
    - dp_size > 1: group by dp_rank, assert exactly one group has non-empty
      tensors, return that group.
    """
    if not items:
        return items

    dp_info: Optional[tuple[int, int]] = _extract_dp_info(items[0].meta)
    if dp_info is None:
        return items

    _dp_rank, dp_size = dp_info
    if dp_size <= 1:
        return items

    has_any_tensor: bool = any(isinstance(item.value, torch.Tensor) for item in items)
    if not has_any_tensor:
        return items

    groups: dict[int, list[ValueWithMeta]] = defaultdict(list)
    for item in items:
        item_dp: Optional[tuple[int, int]] = _extract_dp_info(item.meta)
        rank: int = item_dp[0] if item_dp is not None else 0
        groups[rank].append(item)

    non_empty_ranks: list[int] = [
        rank for rank, group in groups.items() if _group_has_data(group)
    ]

    assert len(non_empty_ranks) == 1, (
        f"Expected exactly 1 non-empty dp_rank, got {len(non_empty_ranks)}: "
        f"ranks={non_empty_ranks}"
    )

    return groups[non_empty_ranks[0]]


def _extract_dp_info(meta: dict) -> Optional[tuple[int, int]]:
    """Extract (dp_rank, dp_size) from meta's parallel_info block."""
    for key in _PARALLEL_INFO_KEYS:
        info = meta.get(key)
        if not isinstance(info, dict) or not info:
            continue

        dp_rank = info.get(_DP_RANK_FIELD)
        dp_size = info.get(_DP_SIZE_FIELD)
        if dp_rank is not None and dp_size is not None:
            return (int(dp_rank), int(dp_size))

    return None


def _group_has_data(group: list[ValueWithMeta]) -> bool:
    """Check if any tensor in the group is non-empty (numel > 0)."""
    return any(
        isinstance(item.value, torch.Tensor) and item.value.numel() > 0
        for item in group
    )
