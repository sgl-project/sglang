"""A TreeCore emits CacheActions through the TreeCoreInterface to guide
Controller behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    import torch

    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId


class ReplaceWriteThroughOnNodeSplit(NamedTuple):
    """Replace the pending write-through node on a node split:

        parent -> node    =>    parent -> new_node -> new_child

    old_node_id (the pre-split node) is replaced by new_node_id + new_child_node_id.
    """

    ack_id: int
    old_node_id: NodeId
    new_node_id: NodeId
    new_child_node_id: NodeId


class FreeDeviceKV(NamedTuple):
    """Free unreferenced device KV slots (a SWA-aware combined free)."""

    indices: list[torch.Tensor]


@dataclass(frozen=True)
class ComponentAction:
    """Base for component-routed actions; the cache dispatches each one to
    ``component_type``'s class-level ``apply_component_action``."""

    component_type: ComponentType = field(kw_only=True)


@dataclass(frozen=True)
class FreeComponentDeviceSlot(ComponentAction):
    """Free only the given ``component_type``'s device KV slots."""

    indices: list[torch.Tensor]


@dataclass(frozen=True)
class FreeComponentHostSlot(ComponentAction):
    """Free the given ``component_type``'s host KV pages."""

    host_indices: list[torch.Tensor]


class BackupKV(NamedTuple):
    """Back up node_ids device->host in order, stopping at the first failure; write-through
    ids form a contiguous root-first parent chain (each id's parent precedes it), write-back
    actions carry a single eviction victim."""

    node_ids: list[NodeId]


@dataclass(frozen=True)
class MambaEvictExcessPathStates(ComponentAction):
    """Deferred per-path Mamba state-cap eviction from the tail's root path;
    ordered after the insert's BackupKVs so in-flight write-through locks
    shield the pending backup chain."""

    tail_node_id: NodeId
    component_type: ComponentType = field(default=ComponentType.MAMBA, kw_only=True)


@dataclass(frozen=True)
class RebuildFullToSWAMapping(ComponentAction):
    """Rebuild the SWA allocator's full->swa index mapping for loaded chunks."""

    full_indices: list[torch.Tensor]
    swa_indices: list[torch.Tensor]
    component_type: ComponentType = field(default=ComponentType.SWA, kw_only=True)


@dataclass(frozen=True)
class RecoverSWAWithLockedFull(ComponentAction):
    """Recover an SWA tombstone whose full is locked: keep the locked full, remap it
    onto the incoming full's SWA translation, and free only the incoming full."""

    node_id: NodeId
    kept_full: torch.Tensor
    incoming_full: torch.Tensor
    component_type: ComponentType = field(default=ComponentType.SWA, kw_only=True)


@dataclass(frozen=True)
class SWARebuild(ComponentAction):
    """Rebuild a node's SWA value by translating its source full value, then store it."""

    node_id: NodeId
    source_value: torch.Tensor
    component_type: ComponentType = field(default=ComponentType.SWA, kw_only=True)


# Cache-owned actions, applied by UnifiedRadixCache itself.
CacheAction = ReplaceWriteThroughOnNodeSplit | FreeDeviceKV | BackupKV
