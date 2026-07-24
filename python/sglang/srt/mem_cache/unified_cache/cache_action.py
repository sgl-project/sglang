"""A TreeCore emits CacheActions through the TreeCoreInterface to guide
Controller behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId


class ReplaceWriteThroughOnNodeSplit(msgspec.Struct, frozen=True):
    """Replace the pending write-through node on a node split:

        parent -> node    =>    parent -> new_node -> new_child

    old_node_id (the pre-split node) is replaced by new_node_id + new_child_node_id.
    """

    ack_id: int
    old_node_id: NodeId
    new_node_id: NodeId
    new_child_node_id: NodeId


class FreeDeviceKV(msgspec.Struct, frozen=True):
    """Free unreferenced device KV slots (a SWA-aware combined free)."""

    indices: list[torch.Tensor]


class ComponentAction(msgspec.Struct, frozen=True):
    """Base for component-routed actions; the cache dispatches each one to
    ``component_type``'s class-level ``apply_component_action``; every subclass
    declares its ``component_type`` field."""


class FreeComponentDeviceSlot(ComponentAction, frozen=True):
    """Free only the given ``component_type``'s device KV slots."""

    indices: list[torch.Tensor]
    component_type: ComponentType


class FreeComponentHostSlot(ComponentAction, frozen=True):
    """Free the given ``component_type``'s host KV pages."""

    host_indices: list[torch.Tensor]
    component_type: ComponentType


class BackupKV(msgspec.Struct, frozen=True):
    """Back up node_ids device->host in order, stopping at the first failure; write-through
    ids form a contiguous root-first parent chain (each id's parent precedes it), write-back
    actions carry a single eviction victim."""

    node_ids: list[NodeId]


class MambaEvictExcessPathStates(ComponentAction, frozen=True):
    """Deferred per-path Mamba state-cap eviction from the tail's root path;
    ordered after the insert's BackupKVs so in-flight write-through locks
    shield the pending backup chain."""

    tail_node_id: NodeId
    component_type: ComponentType = ComponentType.MAMBA


class RebuildFullToSWAMapping(ComponentAction, frozen=True):
    """Rebuild the SWA allocator's full->swa index mapping for loaded chunks."""

    full_indices: list[torch.Tensor]
    swa_indices: list[torch.Tensor]
    component_type: ComponentType = ComponentType.SWA


class RecoverSWAWithLockedFull(ComponentAction, frozen=True):
    """Recover an SWA tombstone whose full is locked: keep the locked full, remap it
    onto the incoming full's SWA translation, and free only the incoming full."""

    node_id: NodeId
    kept_full: torch.Tensor
    incoming_full: torch.Tensor
    component_type: ComponentType = ComponentType.SWA


class SWARebuild(ComponentAction, frozen=True):
    """Rebuild a node's SWA value by translating its source full value, then store it."""

    node_id: NodeId
    source_value: torch.Tensor
    component_type: ComponentType = ComponentType.SWA


# Cache-owned actions, applied by UnifiedRadixCache itself.
CacheAction = ReplaceWriteThroughOnNodeSplit | FreeDeviceKV | BackupKV
