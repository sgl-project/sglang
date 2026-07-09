from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.mem_cache.unified_cache_components import (
    BASE_COMPONENT_TYPE,
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode


@dataclass(frozen=True, slots=True, kw_only=True)
class RadixCacheWalkResult:
    slot_indices: torch.Tensor
    positions: torch.Tensor
    prev_slot_indices: torch.Tensor


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
    if (
        cache_type is not RadixCache
        and cache_type is not SWARadixCache
        and cache_type is not UnifiedRadixCache
    ):
        raise NotImplementedError(
            f"walk_radix_cache_for_canary does not support {cache_type.__name__}"
        )

    slot_buf: list[int] = []
    position_buf: list[int] = []
    prev_slot_buf: list[int] = []

    _walk_radix_subtree(
        node=radix_cache.root_node,
        radix_cache=radix_cache,
        depth=0,
        parent_last_slot=-1,
        slot_buf=slot_buf,
        position_buf=position_buf,
        prev_slot_buf=prev_slot_buf,
        is_root=True,
        unlocked_only=unlocked_only,
        swa_resident_only=swa_resident_only,
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
    node: TreeNode | UnifiedTreeNode,
    radix_cache: BasePrefixCache,
    depth: int,
    parent_last_slot: int,
    slot_buf: list[int],
    position_buf: list[int],
    prev_slot_buf: list[int],
    is_root: bool,
    unlocked_only: bool,
    swa_resident_only: bool,
) -> None:
    node_slots = _node_slots_for_canary(node=node, radix_cache=radix_cache)

    if unlocked_only:
        emit_slots = not is_root and _node_is_unlocked_for_canary(
            node=node, radix_cache=radix_cache
        )
    else:
        emit_slots = not is_root
    if swa_resident_only:
        emit_slots = emit_slots and _node_is_swa_resident_for_canary(
            node=node,
            radix_cache=radix_cache,
        )

    chain_last_slot = parent_last_slot
    for j, slot in enumerate(node_slots):
        prev = parent_last_slot if j == 0 else node_slots[j - 1]
        if emit_slots:
            slot_buf.append(slot)
            position_buf.append(depth + j)
            prev_slot_buf.append(prev)
        chain_last_slot = slot

    child_depth = depth + _node_len_for_canary(
        node=node,
        radix_cache=radix_cache,
        node_slots=node_slots,
        is_root=is_root,
    )
    for child in node.children.values():
        _walk_radix_subtree(
            node=child,
            radix_cache=radix_cache,
            depth=child_depth,
            parent_last_slot=chain_last_slot,
            slot_buf=slot_buf,
            position_buf=position_buf,
            prev_slot_buf=prev_slot_buf,
            is_root=False,
            unlocked_only=unlocked_only,
            swa_resident_only=swa_resident_only,
        )


def _node_slots_for_canary(
    *,
    node: TreeNode | UnifiedTreeNode,
    radix_cache: BasePrefixCache,
) -> list[int]:
    value: Any
    if type(radix_cache) is UnifiedRadixCache:
        value = node.component_data[BASE_COMPONENT_TYPE].value
    else:
        value = node.value

    if isinstance(value, torch.Tensor):
        return [int(s) for s in value.tolist()]
    return []


def _node_len_for_canary(
    *,
    node: TreeNode | UnifiedTreeNode,
    radix_cache: BasePrefixCache,
    node_slots: list[int],
    is_root: bool,
) -> int:
    if type(radix_cache) is not UnifiedRadixCache:
        return len(node_slots)

    if is_root or node.key is None:
        return len(node_slots)
    return len(node.key)


def _node_is_unlocked_for_canary(
    *,
    node: TreeNode | UnifiedTreeNode,
    radix_cache: BasePrefixCache,
) -> bool:
    if type(radix_cache) is RadixCache:
        return node.lock_ref == 0

    if type(radix_cache) is SWARadixCache:
        return node.full_lock_ref == 0

    if type(radix_cache) is UnifiedRadixCache:
        return node.component_data[BASE_COMPONENT_TYPE].lock_ref == 0

    raise NotImplementedError(
        f"walk_radix_cache_for_canary does not support {type(radix_cache).__name__}"
    )


def _node_is_swa_resident_for_canary(
    *,
    node: TreeNode | UnifiedTreeNode,
    radix_cache: BasePrefixCache,
) -> bool:
    if type(radix_cache) is SWARadixCache:
        return not node.swa_tombstone

    if type(radix_cache) is UnifiedRadixCache:
        if ComponentType.SWA not in radix_cache.tree_components:
            return True
        return node.component_data[ComponentType.SWA].value is not None

    return True
