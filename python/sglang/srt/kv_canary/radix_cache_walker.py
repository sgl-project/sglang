from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    RadixCacheWalkResult,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.radix_cache import TreeNode


def walk_radix_cache_for_canary(
    *,
    radix_cache: BasePrefixCache,
    unlocked_only: bool = False,
    swa_resident_only: bool = False,
) -> RadixCacheWalkResult:
    """Walk the radix tree and emit flat (slot_indices, positions, prev_slot_indices) tensors.

    With both flags False (default), emits every slot held by the radix cache (including slots
    also referenced by a currently-running req — that overlap is harmless redundancy with the
    per-forward HEAD/TAIL path). ``unlocked_only=True`` skips nodes still locked by a running
    req. ``swa_resident_only=True`` skips SWA-tombstoned nodes (slots evicted from the SWA
    window)."""
    cache_type = type(radix_cache)
    if cache_type is UnifiedRadixCache:
        return radix_cache.tree_core.walk_for_kv_canary(
            unlocked_only=unlocked_only, swa_resident_only=swa_resident_only
        )
    if cache_type is not RadixCache:
        raise NotImplementedError(
            f"walk_radix_cache_for_canary does not support {cache_type.__name__}"
        )

    # RadixCache has no SWA tier, so every node it holds is SWA-resident and
    # swa_resident_only is a no-op here.
    slot_buf: list[int] = []
    position_buf: list[int] = []
    prev_slot_buf: list[int] = []

    _walk_radix_subtree(
        node=radix_cache.root_node,
        depth=0,
        parent_last_slot=-1,
        slot_buf=slot_buf,
        position_buf=position_buf,
        prev_slot_buf=prev_slot_buf,
        is_root=True,
        unlocked_only=unlocked_only,
    )

    slot_tensor = torch.tensor(slot_buf, dtype=torch.int64)
    position_tensor = torch.tensor(position_buf, dtype=torch.int64)
    prev_slot_tensor = torch.tensor(prev_slot_buf, dtype=torch.int64)
    return RadixCacheWalkResult(
        slot_indices=slot_tensor,
        positions=position_tensor,
        prev_slot_indices=prev_slot_tensor,
    )


def _walk_radix_subtree(
    *,
    node: TreeNode,
    depth: int,
    parent_last_slot: int,
    slot_buf: list[int],
    position_buf: list[int],
    prev_slot_buf: list[int],
    is_root: bool,
    unlocked_only: bool,
) -> None:
    node_slots = _node_slots_for_canary(node=node)

    emit_slots = not is_root and (not unlocked_only or node.lock_ref == 0)

    chain_last_slot = parent_last_slot
    for j, slot in enumerate(node_slots):
        prev = parent_last_slot if j == 0 else node_slots[j - 1]
        if emit_slots:
            slot_buf.append(slot)
            position_buf.append(depth + j)
            prev_slot_buf.append(prev)
        chain_last_slot = slot

    child_depth = depth + len(node_slots)
    for child in node.children.values():
        _walk_radix_subtree(
            node=child,
            depth=child_depth,
            parent_last_slot=chain_last_slot,
            slot_buf=slot_buf,
            position_buf=position_buf,
            prev_slot_buf=prev_slot_buf,
            is_root=False,
            unlocked_only=unlocked_only,
        )


def _node_slots_for_canary(*, node: TreeNode) -> list[int]:
    value: Any = node.value
    if isinstance(value, torch.Tensor):
        return [int(s) for s in value.tolist()]
    return []
