"""SWA KV kept only for the window before each cached Mamba checkpoint."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
    EvictLayer,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
        UnifiedTreeCore,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode


class _Tier(NamedTuple):
    """Attribute names of a node's SWA value, lock count, and window count on device or host."""

    layer: EvictLayer
    value: str
    lock_ref: str
    cover_ref: str


_TIERS = (
    _Tier(EvictLayer.DEVICE, "value", "lock_ref", "cover_ref"),
    _Tier(EvictLayer.HOST, "host_value", "host_lock_ref", "host_cover_ref"),
)


def _is_reclaimable(node: UnifiedTreeNode, host: bool) -> bool:
    tier = _TIERS[host]
    cd = node.component_data[ComponentType.SWA]
    return (
        getattr(cd, tier.value) is not None
        and getattr(cd, tier.lock_ref) == 0
        and getattr(cd, tier.cover_ref) == 0
    )


def _window(
    tree_core: UnifiedTreeCore, checkpoint: UnifiedTreeNode
) -> list[UnifiedTreeNode]:
    """Nodes of the window ending at checkpoint, deepest first."""
    window_tokens = tree_core.checkpoint_window_tokens
    nodes = []
    covered = 0
    node = checkpoint
    while node is not tree_core.root_node and covered < window_tokens:
        nodes.append(node)
        covered += len(node.key)
        node = node.parent
    assert covered <= window_tokens, (
        f"window of node {checkpoint.id} does not start on a node "
        f"boundary ({covered} > {window_tokens} tokens)"
    )
    return nodes


def store_checkpoint(
    tree_core: UnifiedTreeCore,
    checkpoint: UnifiedTreeNode,
    cache_actions: list[CacheAction | ComponentAction],
) -> None:
    """Split the node that straddles a new checkpoint's window start, then add the window to its nodes' counts."""
    window_tokens = tree_core.checkpoint_window_tokens
    covered = 0
    node = checkpoint
    while node is not tree_core.root_node and covered < window_tokens:
        if covered + len(node.key) > window_tokens:
            _, action = tree_core._split_node(
                node.key, node, covered + len(node.key) - window_tokens
            )
            if action is not None:
                cache_actions.append(action)
            break
        covered += len(node.key)
        node = node.parent
    acquire_window(tree_core, checkpoint, False)


def acquire_window(
    tree_core: UnifiedTreeCore, checkpoint: UnifiedTreeNode, host: bool
) -> None:
    """Add a new checkpoint's window to the count of every node it includes."""
    tier = _TIERS[host]
    uncovered = tree_core.swa_uncovered[host]
    for node in _window(tree_core, checkpoint):
        cd = node.component_data[ComponentType.SWA]
        setattr(cd, tier.cover_ref, getattr(cd, tier.cover_ref) + 1)
        uncovered.pop(node.id, None)


def release_window(
    tree_core: UnifiedTreeCore,
    checkpoint: UnifiedTreeNode,
    host: bool,
    device_frees: dict[ComponentType, list[torch.Tensor]],
    host_frees: dict[ComponentType, list[torch.Tensor]],
    cache_actions: list[CacheAction | ComponentAction],
    tracker: Optional[dict[ComponentType, int]],
) -> None:
    """Drop a checkpoint's window, freeing KV no other window covers."""
    tier = _TIERS[host]
    for node in _window(tree_core, checkpoint):
        cd = node.component_data[ComponentType.SWA]
        ref = getattr(cd, tier.cover_ref)
        assert ref > 0, f"window release hit {tier.cover_ref}=0 on node {node.id}"
        setattr(cd, tier.cover_ref, ref - 1)
        if not _is_reclaimable(node, host):
            continue
        if host:
            _free(
                tree_core, node, True, device_frees, host_frees, cache_actions, tracker
            )
        else:
            release_device_kv(
                tree_core, node, device_frees, host_frees, cache_actions, tracker
            )


def release_device_kv(
    tree_core: UnifiedTreeCore,
    node: UnifiedTreeNode,
    device_frees: dict[ComponentType, list[torch.Tensor]],
    host_frees: dict[ComponentType, list[torch.Tensor]],
    cache_actions: list[CacheAction | ComponentAction],
    tracker: Optional[dict[ComponentType, int]],
) -> None:
    """Free a node's unlocked device SWA KV outside every window, or queue it while a host window still needs it."""
    cd = node.component_data[ComponentType.SWA]
    # Hack: a write-through backup stops at an in-flight ancestor, so a host window can lack this node's SWA.
    # TODO: back up that SWA when the ancestor's ack lands, then hold the node only until then.
    if cd.host_cover_ref > 0 and cd.host_value is None:
        queue_if_uncovered(tree_core, node, False)
    else:
        _free(tree_core, node, False, device_frees, host_frees, cache_actions, tracker)


def _free(
    tree_core: UnifiedTreeCore,
    node: UnifiedTreeNode,
    host: bool,
    device_frees: dict[ComponentType, list[torch.Tensor]],
    host_frees: dict[ComponentType, list[torch.Tensor]],
    cache_actions: list[CacheAction | ComponentAction],
    tracker: Optional[dict[ComponentType, int]],
) -> None:
    tree_core._evict_component_and_detach_lru(
        node,
        tree_core.components_by_type[ComponentType.SWA],
        device_frees,
        host_frees,
        cache_actions,
        target=_TIERS[host].layer,
        tracker=tracker,
    )
    tree_core._update_evictable_leaf_sets(node)


def queue_if_uncovered(
    tree_core: UnifiedTreeCore, node: UnifiedTreeNode, host: bool
) -> None:
    """Queue the node to be freed if its SWA KV on this tier is unlocked and outside every window."""
    if _is_reclaimable(node, host):
        tree_core.swa_uncovered[host][node.id] = node


def dequeue(
    tree_core: UnifiedTreeCore, node: UnifiedTreeNode, target: EvictLayer
) -> None:
    for host, tier in enumerate(_TIERS):
        if tier.layer in target:
            tree_core.swa_uncovered[host].pop(node.id, None)


def _pop_reclaimable(
    tree_core: UnifiedTreeCore, host: bool
) -> Optional[UnifiedTreeNode]:
    uncovered = tree_core.swa_uncovered[host]
    while uncovered:
        node_id = next(iter(uncovered))
        node = uncovered.pop(node_id)
        if tree_core._node_arena.get(node_id) is node and _is_reclaimable(node, host):
            return node
    return None


def redistribute_on_node_split(
    tree_core: UnifiedTreeCore,
    new_parent: UnifiedTreeNode,
    child: UnifiedTreeNode,
) -> None:
    """Copy the child's window counts and free-queue entry to the new parent of a split."""
    parent_cd = new_parent.component_data[ComponentType.SWA]
    child_cd = child.component_data[ComponentType.SWA]
    for host, tier in enumerate(_TIERS):
        setattr(parent_cd, tier.cover_ref, getattr(child_cd, tier.cover_ref))
        queue_if_uncovered(tree_core, new_parent, bool(host))


def evict_device_next_node(
    tree_core: UnifiedTreeCore,
    request_cnt: int,
    tracker: dict[ComponentType, int],
    device_frees: dict[ComponentType, list[torch.Tensor]],
    host_frees: dict[ComponentType, list[torch.Tensor]],
    cache_actions: list[CacheAction | ComponentAction],
) -> None:
    """Free one node's SWA KV that is outside every window. Windowed KV is freed with its checkpoint."""
    if tracker[ComponentType.SWA] < request_cnt:
        node = _pop_reclaimable(tree_core, False)
        if node is not None:
            _free(
                tree_core, node, False, device_frees, host_frees, cache_actions, tracker
            )


def drive_host_eviction(
    tree_core: UnifiedTreeCore,
    num_tokens: int,
    tracker: dict[ComponentType, int],
    device_frees: dict[ComponentType, list[torch.Tensor]],
    host_frees: dict[ComponentType, list[torch.Tensor]],
    cache_actions: list[CacheAction | ComponentAction],
) -> None:
    """Free host SWA KV outside every window, then evict host checkpoints with their windows in LRU order."""
    swa = ComponentType.SWA
    while tracker[swa] < num_tokens:
        node = _pop_reclaimable(tree_core, True)
        if node is None:
            break
        _free(tree_core, node, True, device_frees, host_frees, cache_actions, tracker)
    mamba = tree_core.components_by_type[ComponentType.MAMBA]
    while tracker[swa] < num_tokens:
        checkpoint_tracker: dict[ComponentType, int] = defaultdict(int)
        mamba.drive_host_eviction(
            1, checkpoint_tracker, device_frees, host_frees, cache_actions
        )
        if not any(checkpoint_tracker.values()):
            break
        for component, count in checkpoint_tracker.items():
            tracker[component] += count


def validate(
    tree_core: UnifiedTreeCore,
    reachable_nodes: set[UnifiedTreeNode],
    report_error: Callable[[str], None],
) -> None:
    """Report window counts and reclaim queues that disagree with the tree."""
    swa = ComponentType.SWA
    nodes = [node for node in reachable_nodes if node is not tree_core.root_node]
    for host, tier in enumerate(_TIERS):
        expected: dict[int, int] = defaultdict(int)
        for node in nodes:
            if getattr(node.component_data[ComponentType.MAMBA], tier.value) is None:
                continue
            for covered in _window(tree_core, node):
                expected[covered.id] += 1
        uncovered = tree_core.swa_uncovered[host]
        for node in nodes:
            ref = getattr(node.component_data[swa], tier.cover_ref)
            if ref != expected.get(node.id, 0):
                report_error(
                    f"{swa} node {node.id} {tier.cover_ref}={ref}, "
                    f"expected {expected.get(node.id, 0)}"
                )
            if _is_reclaimable(node, bool(host)) and uncovered.get(node.id) is not node:
                report_error(
                    f"{swa} node {node.id} holds uncovered {tier.layer.name} "
                    "KV unqueued"
                )
